import os
import re
import numpy as np
import faiss
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# -------------------- 初始設定 --------------------
load_dotenv()

# OpenAI client
def setup_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=api_key)

openai_client = setup_openai()

# LINE Bot client
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
if not LINE_TOKEN or not LINE_SECRET:
    raise RuntimeError("Missing LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# -------------------- TDEE 計算及狀態管理 --------------------
user_states = {}  # { user_id: { 'stage': str, 'data': {} } }

def calculate_tdee(sex, age, height, weight, activity, goal):
    if sex == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    factors = {
        "靜態": 1.20,
        "輕度": 1.375,
        "中度": 1.55,
        "高度": 1.725,
        "劇烈": 1.90
    }
    tdee = bmr * factors.get(activity, 1.20)
    if goal == "減重":
        tdee -= 500
    elif goal == "增重":
        tdee += 300
    return round(tdee)

# -------------------- RAG 函式 --------------------
def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split("####")
    partitions = {}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        header, body = sec.split("\n", 1) if "\n" in sec else (sec, "")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", ".", "。"]
        )
        docs = [Document(page_content=body)]
        splits = splitter.split_documents(docs)
        segments = [f"{header}\n{d.page_content}" for d in splits]
        partitions[header] = segments
    return partitions

def initialize_rag(partitions, model_name="intfloat/multilingual-e5-base"):
    device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
    embed_model = SentenceTransformer(model_name, device=device)
    indexes, segments_map = {}, {}
    for sec, texts in partitions.items():
        embs = embed_model.encode(texts, show_progress_bar=True)
        dim = embs.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(np.array(embs, dtype="float32"))
        indexes[sec] = idx
        segments_map[sec] = texts
    return embed_model, indexes, segments_map

def query_rag(query, embed_model, indexes, segments_map, top_k=3, threshold=1.5):
    q_emb = embed_model.encode([query])
    results = []
    for sec, idx in indexes.items():
        dists, ids = idx.search(np.array(q_emb, dtype="float32"), k=top_k)
        for dist, i in zip(dists[0], ids[0]):
            if i >= 0 and dist < threshold:
                results.append(segments_map[sec][i])
    return results

# -------------------- 智能回覆＋自動續寫 --------------------
def generate_answer(openai_client, prompt, contexts,
                    model_name="gpt-4.1", max_tokens=200):
    context = "\n".join(contexts)[:2000]
    messages = [
        {"role": "system", "content": (
            "你是專業食物推薦助理，只能根據使用者提供的 TDEE 或查詢內容推薦，"
            "不要進行隨機建議。"
        )},
        {"role": "user", "content": f"{prompt}\n上下文片段:\n{context}"}
    ]
    # 第一次呼叫
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens
    )
    text = resp.choices[0].message.content
    # 如果被截斷，續寫剩餘內容
    if resp.choices[0].finish_reason == "length":
        cont = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "延續上文，不要重複前面內容，只接著說完。"},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        text += cont.choices[0].message.content
    return text

# -------------------- 意圖分類函式 --------------------
def detect_intent(openai_client, text):
    messages = [
        {"role": "system", "content": (
            "你是一個意圖分類助理，"
            "請判斷使用者輸入是在要求「計算 TDEE」還是「查食譜／一般問答」（RAG）。"
            "回應格式只要單字：TDEE 或 RAG。"
        )},
        {"role": "user", "content": text}
    ]
    resp = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# 資料索引初始化
partitions = load_and_partition_text("recipes.txt")
embed_model, indexes, segments_map = initialize_rag(partitions)

# -------------------- Flask + LINE Callback --------------------
app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    state = user_states.get(user_id)

    # 若正在進行 TDEE 流程，先處理多階段收集
    if state:
        data = state["data"]
        stage = state["stage"]

        if stage == "ask_sex":
            if text in ["男", "女"]:
                data["sex"] = "male" if text == "男" else "female"
                user_states[user_id]["stage"] = "ask_age"
                reply = "請輸入年齡（歲）"
            else:
                reply = "性別請輸入 男 或 女"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if stage == "ask_age":
            try:
                data["age"] = int(text)
                user_states[user_id]["stage"] = "ask_height"
                reply = "請輸入身高（公分）"
            except ValueError:
                reply = "年齡請輸入整數"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if stage == "ask_height":
            try:
                data["height"] = float(text)
                user_states[user_id]["stage"] = "ask_weight"
                reply = "請輸入體重（公斤）"
            except ValueError:
                reply = "身高請輸入數字"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if stage == "ask_weight":
            try:
                data["weight"] = float(text)
                user_states[user_id]["stage"] = "ask_activity"
                reply = "選擇活動程度: 靜態/輕度/中度/高度/劇烈"
            except ValueError:
                reply = "體重請輸入數字"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if stage == "ask_activity":
            if text in ["靜態", "輕度", "中度", "高度", "劇烈"]:
                data["activity"] = text
                user_states[user_id]["stage"] = "ask_goal"
                reply = "最後告訴我目標: 維持/減重/增重"
            else:
                reply = "請輸入正確活動程度"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if stage == "ask_goal":
            if text in ["維持", "減重", "增重"]:
                data["goal"] = text
                # 計算並回傳 TDEE
                tdee = calculate_tdee(
                    sex=data["sex"],
                    age=data["age"],
                    height=data["height"],
                    weight=data["weight"],
                    activity=data["activity"],
                    goal=data["goal"]
                )
                msg1 = TextSendMessage(text=f"你的 TDEE 約為 {tdee} 大卡")
                # 根據 TDEE 推薦三餐
                segs = query_rag(f"{tdee} 大卡", embed_model, indexes, segments_map)
                prompt = (
                    f"請根據我每日需求 {tdee} 大卡，推薦一天三餐的食物組合，"
                    f"並註明每餐熱量，總和不超過 {tdee} 大卡。"
                )
                msg2 = TextSendMessage(text=generate_answer(openai_client, prompt, segs))
                line_bot_api.reply_message(event.reply_token, [msg1, msg2])
                user_states.pop(user_id, None)
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="目標請輸入 維持、減重 或 增重")
                )
            return

    # 不在 TDEE 流程，先做意圖分類
    intent = detect_intent(openai_client, text)
    if intent == "TDEE":
        user_states[user_id] = {"stage": "ask_sex", "data": {}}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請告訴我性別（男／女）")
        )
        return

    # RAG + GPT 回答（自動續寫）
    if text.startswith("食物") or re.match(r"^\d+\s*大卡", text):
        segs = query_rag(text, embed_model, indexes, segments_map)
        ans = generate_answer(openai_client, f"請提供食譜詳情: {text}", segs)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ans))
        return

    segs = query_rag(text, embed_model, indexes, segments_map)
    ans = generate_answer(openai_client, text, segs)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ans))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
