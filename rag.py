import os
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from dotenv import load_dotenv, find_dotenv
# 新增 LINE Bot SDK
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

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
LINE_TOKEN  = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
if not LINE_TOKEN or not LINE_SECRET:
    raise RuntimeError("Missing LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_TOKEN)
handler      = WebhookHandler(LINE_SECRET)

# -------------------- RAG 函式 --------------------
def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split("####")
    partitions = {}
    for sec in sections:
        sec = sec.strip()
        if not sec: continue
        header, body = sec.split("\n", 1) if "\n" in sec else (sec, "")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n", ".", "。"]
        )
        docs   = [Document(page_content=body)]
        splits = splitter.split_documents(docs)
        segments = [f"{header}\n{d.page_content}" for d in splits]
        partitions[header] = segments
    return partitions

def initialize_rag(partitions, model_name="intfloat/multilingual-e5-base"):
    device     = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
    embed_model = SentenceTransformer(model_name, device=device)
    indexes, segments_map = {}, {}
    for sec, texts in partitions.items():
        embs = embed_model.encode(texts, show_progress_bar=True)
        dim  = embs.shape[1]
        idx  = faiss.IndexFlatL2(dim)
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

def generate_answer(openai_client, query, contexts, model_name="gpt-4.1", max_tokens=200):
    context = "\n".join(contexts)[:2000]
    messages = [
        {"role":"system","content":"你是食物推薦助理，請根據上下文回答問題，不超過200字。"},
        {"role":"user","content":f"問題: {query}\n上下文: {context}"}
    ]
    resp = openai_client.chat.completions.create(
        model=model_name, messages=messages,
        temperature=0.7, max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# 載入資料並建立索引
partitions       = load_and_partition_text("recipes.txt")
embed_model, indexes, segments_map = initialize_rag(partitions)

# -------------------- Flask + LINE Callback --------------------
app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body      = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_q = event.message.text
    segs   = query_rag(user_q, embed_model, indexes, segments_map)
    answer = generate_answer(openai_client, user_q, segs)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )

if __name__ == "__main__":
    # Flask 監聽 8000 埠號
    app.run(host="0.0.0.0", port=8000)
