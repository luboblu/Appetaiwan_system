import os
import re
import logging
import numpy as np
import faiss
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from google.cloud.firestore_v1.document import DocumentReference

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------- 初始設定 --------------------
load_dotenv()

# -------------------- Firebase Init --------------------
firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS_JSON")
if not firebase_creds_path:
    raise RuntimeError("Missing FIREBASE_CREDENTIALS_JSON in environment")

# 檢查檔案是否存在
if not os.path.exists(firebase_creds_path):
    raise RuntimeError(f"Firebase credentials file not found: {firebase_creds_path}")

cred = credentials.Certificate(firebase_creds_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firebase: Save user profile (merge) and log messages
def save_user_profile(user_id, profile: dict):
    doc_ref = db.collection("users").document(user_id)
    data = {**profile, "updatedAt": firestore.SERVER_TIMESTAMP}
    doc_ref.set(data, merge=True)

def get_user_profile(user_id):
    """從 Firebase 取得用戶基本資料"""
    try:
        doc_ref = db.collection("users").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return None

def get_user_messages(user_id):
    """從 Firebase 取得用戶訊息紀錄"""
    try:
        msg_ref = db.collection("users").document(user_id).collection("messages")
        docs = msg_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        logger.error(f"Error getting user messages: {e}")
        return []

def compose_prompts(user_id: str, system_prompt: str, *rest_prompts):
    """
    1) Fetch existing user messages (each object has msg.role and msg.content).
    2) Convert any msg.role=="bot" → "assistant", otherwise keep msg.role.
    3) Append zero or more new prompts from rest_prompts, treating each as a 'user' message
       unless it’s already passed in as a dict with a valid role/content.
    Returns a flat list of { "role": ..., "content": ... } dicts suitable for openai.ChatCompletion.
    """
    # 1) Start with the system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # 2) Append the user’s prior history (convert "bot" → "assistant")
    user_messages = get_user_messages(user_id)
    for msg in user_messages:
        # Ensure only valid roles: map "bot" → "assistant"; otherwise trust msg.role
        role = "assistant" if msg["role"] == "bot" else msg["role"]
        if role not in ("system", "user", "assistant"):
            # Fallback: anything unexpected becomes "user"
            role = "user"
        messages.append({
            "role": role,
            "content": msg["content"]
        })

    # 3) Append each item in rest_prompts
    for prompt in rest_prompts:
        if isinstance(prompt, dict):
            # If the caller provided a message‐dict directly, validate it:
            role = prompt.get("role")
            content = prompt.get("content")
            if role in ("system", "user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})
            else:
                # If the dict is malformed or has an invalid role, treat it as a user‐content string
                messages.append({"role": "user", "content": str(prompt)})
        else:
            # If prompt is a plain string (or anything else), assume it's user content
            messages.append({
                "role": "user",
                "content": str(prompt)
            })

    return messages

def log_message(user_id, role, content):
    msg_ref = db.collection("users").document(user_id).collection("messages")
    msg_ref.add({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

# OpenAI client
def setup_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Missing OPENAI_API_KEY in environment")
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

openai_client = setup_openai()

# LINE Bot client
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
if not LINE_TOKEN or not LINE_SECRET:
    logger.error("Missing LINE tokens")
    raise RuntimeError("Missing LINE tokens")

line_bot_api = LineBotApi(LINE_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# -------------------- TDEE 計算及狀態管理 --------------------
user_states = {}

def calculate_tdee(sex, age, height, weight, activity, goal):
    if not (10 <= age <= 120 and 50 <= height <= 250 and 20 <= weight <= 200):
        raise ValueError("輸入數值超出合理範圍")
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if sex == "male" else -161)
    factors = {"靜態":1.20, "輕度":1.375, "中度":1.55, "高度":1.725, "劇烈":1.90}
    tdee = bmr * factors.get(activity,1.20)
    if goal == "減重": tdee -= 500
    elif goal == "增重": tdee += 300
    return round(tdee)

def has_basic_profile(user_profile):
    """檢查用戶是否已有完整的基本資料"""
    if not user_profile:
        return False
    required_fields = ["sex", "age", "height", "weight"]
    return all(field in user_profile and user_profile[field] is not None for field in required_fields)

# -------------------- RAG 函式 --------------------
def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = content.split("####")
    partitions = {}
    for sec in parts:
        sec = sec.strip()
        if not sec: continue
        header, body = (sec.split("\n",1) if "\n" in sec else (sec, ""))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n", ".", "。"]
        )
        docs = [Document(page_content=body)]
        splits = splitter.split_documents(docs)
        segments = [f"{header}\n{d.page_content}" for d in splits]
        partitions[header] = segments
    return partitions

# 載入食譜資料但不立即初始化模型
partitions = load_and_partition_text("recipes.txt")

# 全域變數，延遲載入
embed_model = None
indexes = {}
segments_map = {}

def initialize_rag():
    """延遲初始化 RAG 系統，避免冷啟動超時"""
    global embed_model, indexes, segments_map
    
    logger.info("開始初始化 RAG 系統...")
    
    # 自動偵測設備
    device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
    logger.info(f"使用設備: {device}")
    
    # 載入 embedding 模型
    embed_model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    
    # 建立每個分區的索引
    for sec, texts in partitions.items():
        logger.info(f"建立索引: {sec} ({len(texts)} 個片段)")
        embs = embed_model.encode(texts, show_progress_bar=False)
        idx = faiss.IndexFlatL2(embs.shape[1])
        idx.add(np.array(embs, dtype="float32"))
        indexes[sec] = idx
        segments_map[sec] = texts
    
    logger.info("RAG 系統初始化完成！")

def build_context(segs, max_chars=2000):
    ctx = ""
    for s in segs:
        if len(ctx)+len(s)+1>max_chars: break
        ctx += s+"\n"
    return ctx

# -------------------- 智能回應模式檢測 --------------------
# 1. 改進的語言檢測函數
def detect_language(text):
    """
    檢測用戶使用的語言
    返回: 'zh-TW' (繁體中文), 'en' (英文)
    """
    # 檢查是否包含中文字符
    chinese_chars = 0
    english_chars = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符範圍
            chinese_chars += 1
        elif 'a' <= char.lower() <= 'z':
            english_chars += 1
    
    logger.info(f"語言檢測 - 中文字符: {chinese_chars}, 英文字符: {english_chars}")
    
    # 判斷主要語言
    if chinese_chars > 0:
        return 'zh-TW'  # 有任何中文就視為中文
    elif english_chars > 0:
        return 'en'
    else:
        return 'zh-TW'  # 預設繁體中文

def detect_response_mode(text, is_tdee_recommendation=False):
    """檢測應該使用簡潔還是詳細回應模式"""
    if is_tdee_recommendation:
        return "concise"  # TDEE 推薦總是簡潔
    
    # 檢測是否為單一食物/餐點查詢
    single_food_keywords = [
        # 中文
        "怎麼做", "食譜", "作法", "料理方法", "烹調", "製作", "步驟",
        "營養", "好處", "功效", "成分", "熱量",
        # 英文
        "how to make", "recipe", "cooking", "preparation", "steps", "method",
        "nutrition", "benefits", "ingredients", "calories", "how to cook"
    ]
    
    # 檢測是否為多餐點推薦查詢
    multiple_food_keywords = [
        # 中文
        "推薦", "搭配", "一天", "三餐", "早午晚", "菜單", "組合",
        # 英文
        "recommend", "suggest", "daily", "meals", "menu", "combination", "plan"
    ]
    
    text_lower = text.lower()
    
    # 如果包含單一食物查詢關鍵字，使用詳細模式
    if any(keyword in text_lower for keyword in single_food_keywords):
        return "detailed"
    
    # 如果包含多餐點推薦關鍵字，使用簡潔模式
    if any(keyword in text_lower for keyword in multiple_food_keywords):
        return "concise"
    
    # 預設使用詳細模式
    return "detailed"

# -------------------- 聊天生成 --------------------
def generate_answer(prompt, contexts, language="zh", model_name="gpt-4.1-mini", max_tokens=1024, response_mode="detailed"):
    context = build_context(contexts)
    
    # 記錄除錯資訊
    logger.info(f"Generate Answer Debug:")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Context length: {len(context)} chars")
    logger.info(f"Number of context segments: {len(contexts)}")
    logger.info(f"Response mode: {response_mode}")
    
    # 檢查是否有足夠的上下文
    if not context.strip():
        logger.warning("No context available for generation!")
        if language == "en":
            return "Sorry, I couldn't find relevant information in the recipe database. Please try asking other recipe-related questions or rephrase your request."
        else:
            return "抱歉，我在食譜資料庫中找不到相關的資訊。請嘗試詢問其他食譜相關問題，或者重新描述你的需求。"
    
    # 記錄實際的上下文內容（前200字元）
    logger.info(f"Context preview: {context[:200]}...")
    
    # 根據回應模式調整系統提示
    if response_mode == "concise":
        # 簡潔模式：用於 TDEE 推薦
        if language == "en":
            system_content = "You are a professional food recommendation assistant. Based STRICTLY on the provided context, give CONCISE meal recommendations with food names, approximate calories, and basic nutritional balance. If the context doesn't contain specific information, mention this limitation. Keep it brief and practical. You MUST respond in English."
            user_content = f"Question: {prompt}\n\nContext snippets:\n{context}\n\nPlease answer concisely based ONLY on the context provided. Keep your response to 2-3 short sentences, focusing on practical meal suggestions with calorie estimates."
        else:
            system_content = "你是專業食物推薦助理。請嚴格根據提供的上下文，給出簡潔的餐點推薦，包含食物名稱、大約熱量和基本營養搭配。如果上下文沒有具體資訊，請說明此限制。保持簡潔實用。重要：必須使用繁體中文回答，不可使用簡體中文。"
            user_content = f"問題：{prompt}\n\n上下文片段:\n{context}\n\n請僅根據提供的上下文簡潔回答，回應控制在2-3句話內，重點提供實用的餐點建議和熱量估計。"
        max_tokens = 1024  # 給足夠空間但不過度
    else:
        # 詳細模式：用於單一食譜查詢
        if language == "en":
            system_content = (
                "You are a professional food-recommendation assistant. Please respond **within 1,600 tokens**, "
                "using bullet points or numbered lists in exactly these three sections:\n"
                "1. Ingredients (with quantities)\n"
                "2. Steps (numbered, keep each step short and to the point)\n"
                "3. Tips (2-3 brief lines)\n"
                "Avoid any lengthy background or historical notes. End immediately after 'Tips'."
                "\nIMPORTANT: You MUST respond in English."
            )
            user_content = (
                f"Question: {prompt}\n\nContext snippets:\n{context}\n\n"
                "Please follow the above system instructions and reply within 1,600 tokens, "
                "providing a complete recipe."
            )
            max_tokens = 1024
        else:
            system_content = (
                "你是專業食物推薦助理。請依照以下格式、在**1600 個 token 以內**，"
                "用條列式或編號式回答我：\n"
                "- 材料（Ingredients）：列出食材與份量\n"
                "- 步驟（Steps）：以 1、2、3… 的編號說明製作要點\n"
                "- 小技巧（Tips）：2~3 行簡短提示即可\n"
                "回答請盡量簡潔，切忌冗長敘述或背景介紹，並於「小技巧」完成後直接結束。"
                "\n重要：必須使用繁體中文回答，不可使用簡體中文。"
            )
            user_content = (
                f"問題：{prompt}\n\n上下文片段：\n{context}\n\n"
                "請依照上述「系統指令」的格式要求，儘量壓縮文字、"
                "在1600 個 token 以內，完成食譜回覆。"
            )
            max_tokens = 1024
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    try:
        resp = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=max_tokens  # 降低溫度減少創造性
        )
    except OpenAIError as e:
        logger.error(f"ChatCompletion error: {e}")
        if language == "en":
            return "Sorry, unable to get a response."
        else:
            return "抱歉，無法取得回覆。"
    
    text = resp.choices[0].message.content
    
    # 記錄生成結果
    logger.info(f"Generated response length: {len(text)} chars")
    logger.info(f"Finish reason: {resp.choices[0].finish_reason}")
    
    # 簡潔模式不需要延續，詳細模式才延續
    if response_mode == "detailed" and resp.choices[0].finish_reason == "length":
        try:
            if language == "en":
                continuation_prompt = "The previous response was cut off. Please continue and complete the information about the recipe/food, keeping it concise. Don't repeat what was already said."
            else:
                continuation_prompt = "前面的回應被截斷了，請繼續完成關於食譜/食物的資訊，保持簡潔。不要重複已經說過的內容。"
            
            cont = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": continuation_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.5,
                max_tokens=1024
            )
            text += cont.choices[0].message.content
            logger.info(f"Extended response, final length: {len(text)} chars")
        except OpenAIError as e:
            logger.warning(f"Continuation error: {e}")
    
    return text

# ==================== 4. 語言自適應的聊天回應 ====================
def generate_chat_response_adaptive(user_id, prompt, model_name="gpt-4.1-mini", max_tokens=1024):
    """
    語言自適應的聊天回應 - 加強版
    """
    # 先檢測語言
    detected_lang = detect_language(prompt)
    
    system_content = f"""
You are a professional recipe recommendation assistant. Besides providing nutrition and recipe advice, you can also engage in simple daily conversations.

CRITICAL LANGUAGE RULES:
{
'- You MUST respond in Traditional Chinese (繁體中文), NOT Simplified Chinese (简体中文)' if detected_lang == 'zh-TW' 
else '- You MUST respond in English'
}
- This is non-negotiable, regardless of what language the user uses


Guidelines:
- Maintain a friendly and concise response style
- Remind users when appropriate that your expertise is in food and nutrition assistance
- Keep responses to 2-3 sentences
"""
        
    messages = compose_prompts(user_id, system_content, prompt)
    
    try:
        resp = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        logger.error(f"語言自適應聊天回應錯誤: {e}")
        return "抱歉，我現在無法回應。請稍後再試。" if detected_lang == 'zh-TW' else "Sorry, I cannot respond right now. Please try again later."




def query_food_calories(food_name, language="zh"):
    """
    查詢特定食物的卡路里信息
    """
    logger.info(f"查詢食物卡路里: {food_name}, 語言: {language}")
    
    # 確保RAG系統已初始化
    global embed_model, indexes
    if embed_model is None or not indexes:
        logger.info("RAG系統未初始化，開始初始化...")
        initialize_rag()
    
    # 構建查詢關鍵字
    if language == "en":
        # 如果是英文查詢，先翻譯成中文進行檢索
        search_query = f"{food_name} 卡路里 熱量 營養"
    else:
        search_query = f"{food_name} 卡路里 熱量 營養"
    
    # 使用RAG檢索相關片段
    segments = query_rag(search_query, top_k=5, language=language, debug=True)
    
    if not segments:
        if language == "zh":
            return f"抱歉，我在資料庫中找不到關於「{food_name}」的卡路里資訊。"
        else:
            return f"Sorry, I couldn't find calorie information for '{food_name}' in the database."
    
    # 構建上下文
    context = "\n\n".join(segments)
    
    # 生成回應
    if language == "zh":
        system_content = (
            "你是營養資訊助理。請根據提供的食譜上下文，提取並回答用戶關於特定食物卡路里的問題。"
            "回應格式要求：\n"
            "1. 直接回答食物的卡路里數值\n"
            "2. 如果有的話，補充營養成分（蛋白質、脂肪、碳水化合物）\n"
            "3. 保持簡潔，2-3句話即可\n"
            "4. 如果上下文中沒有確切資訊，請誠實說明"
        )
        user_content = f"問題：{food_name}有多少卡路里？\n\n相關食譜片段：\n{context}\n\n請根據上述資訊回答。"
    else:
        system_content = (
            "You are a nutrition information assistant. Based on the provided recipe context, "
            "extract and answer the user's question about specific food calories.\n"
            "Response format requirements:\n"
            "1. Directly answer the calorie count of the food\n"
            "2. If available, add nutritional information (protein, fat, carbohydrates)\n"
            "3. Keep it concise, 2-3 sentences\n"
            "4. If the context doesn't have exact information, be honest about it"
        )
        user_content = f"Question: How many calories does {food_name} have?\n\nRelevant recipe snippets:\n{context}\n\nPlease answer based on the above information."
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"卡路里查詢結果: {result}")
        return result
        
    except OpenAIError as e:
        logger.error(f"生成卡路里回應時發生錯誤: {e}")
        if language == "zh":
            return "抱歉，無法取得卡路里資訊，請稍後再試。"
        else:
            return "Sorry, I couldn't retrieve calorie information at the moment. Please try again later."
# ==================== 2. 語言自適應的食物名稱提取 ====================
def extract_food_name_adaptive(user_id, text):
    """
    語言自適應的食物名稱提取
    """
    logger.info(f"語言自適應提取食物名稱: {text}")
    
    try:
        system_content = """
You are a professional nutrition consultant. Please analyze the user's question (in ANY language) and extract the food(s) they want calorie information about.

Please respond in JSON format:
{
    "primary_food": "main food name",
    "additional_foods": ["other food1", "other food2"],
    "confidence": 0.9
}

Rules:
1. primary_food: The main food the user is most interested in
2. additional_foods: If multiple foods mentioned, list the others
3. confidence: Your confidence in the extraction (0-1)
4. Keep the food names in the SAME LANGUAGE as the user's input
5. For dishes, keep the dish name (e.g., "fried rice", "蛋炒飯")

Examples:
"蘋果和香蕉哪個熱量高？" → {"primary_food": "蘋果", "additional_foods": ["香蕉"], "confidence": 0.95}
"Which has more calories, apple or banana?" → {"primary_food": "apple", "additional_foods": ["banana"], "confidence": 0.95}
"一碗蛋炒飯大概多少卡路里" → {"primary_food": "蛋炒飯", "additional_foods": [], "confidence": 0.9}
"How many calories in a bowl of fried rice" → {"primary_food": "fried rice", "additional_foods": [], "confidence": 0.9}
"""
        
        user_content = f"Extract food names from: {text}"
        messages = compose_prompts(user_id, system_content, user_content)
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=1024
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"AI原始回應: {result_text}")
        
        # 解析 JSON 回應
        import json
        try:
            result = json.loads(result_text)
            primary_food = result.get("primary_food", "")
            additional_foods = result.get("additional_foods", [])
            confidence = result.get("confidence", 0.0)
            
            logger.info(f"AI提取結果 - 主要食物: {primary_food}, 其他食物: {additional_foods}, 信心度: {confidence}")
            
            if confidence > 0.7 and primary_food:
                return {
                    "primary_food": primary_food,
                    "additional_foods": additional_foods,
                    "confidence": confidence
                }
            else:
                # 信心度不足，使用原始文字作為食物名稱
                return {"primary_food": text, "additional_foods": [], "confidence": 0.5}
                
        except json.JSONDecodeError:
            logger.warning("無法解析AI的JSON回應，使用原始文字")
            return {"primary_food": text, "additional_foods": [], "confidence": 0.6}
            
    except Exception as e:
        logger.error(f"語言自適應提取失敗: {e}")
        return {"primary_food": text, "additional_foods": [], "confidence": 0.3}


# ==================== 3. 語言自適應的卡路里查詢 ====================
def query_food_calories_adaptive(user_id, user_query):
    """
    語言自適應的卡路里查詢函數
    """
    logger.info(f"語言自適應卡路里查詢: {user_query}")
    
    # 檢測語言
    detected_lang = detect_language(user_query)
    logger.info(f"檢測到的語言: {detected_lang}")
    
    # 使用語言自適應提取食物名稱
    extraction_result = extract_food_name_adaptive(user_id, user_query)
    primary_food = extraction_result["primary_food"]
    additional_foods = extraction_result["additional_foods"]
    confidence = extraction_result["confidence"]
    
    logger.info(f"提取信心度: {confidence}, 主要食物: {primary_food}")
    
    # 確保RAG系統已初始化
    global embed_model, indexes
    if embed_model is None or not indexes:
        logger.info("RAG系統未初始化，開始初始化...")
        initialize_rag()
    
    # 構建查詢字串（始終用中文檢索，因為資料庫是中文）
    search_foods = [primary_food] + additional_foods
    search_query = " ".join(search_foods) + " 卡路里 熱量 營養"
    
    # 使用RAG檢索
    segments = query_rag(search_query, top_k=5, language="zh", debug=True)
    
    if not segments:
        if detected_lang == 'zh-TW':
            return f"抱歉，我在資料庫中找不到關於「{primary_food}」的卡路里資訊。"
        else:
            return f"Sorry, I couldn't find calorie information for '{primary_food}' in the database."
    
    # 構建上下文
    context = "\n\n".join(segments)
    
    # 根據檢測到的語言設定不同的提示
    if detected_lang == 'en':
        # 英文回應 - 超強化版本
        system_content = """You are a nutritionist providing calorie information.

ABSOLUTE RULE: You MUST respond ONLY in English. No Chinese characters allowed in your response.

Based on the recipe data provided (which may be in Chinese), extract and provide:
1. The calorie count for the requested food
2. Basic nutritional info if available (protein, fat, carbs)
3. Serving size information

Keep your response to 2-3 sentences maximum.
Translate any Chinese food names to English in your response."""
        
        user_content = f"""User asked: "{user_query}"

Recipe data (in Chinese - you must translate relevant information to English):
{context}

Remember: Your response MUST be entirely in English."""
        
    else:
        # 中文回應
        system_content = """你是營養資訊助理。根據提供的食譜資料回答用戶的卡路里問題。

絕對規則：必須使用繁體中文回答，不可使用簡體中文。

請提供：
1. 食物的卡路里數值
2. 如果有的話，補充營養成分（蛋白質、脂肪、碳水化合物）
3. 份量資訊

回應保持在2-3句話內。"""
        
        user_content = f"""用戶問題：{user_query}

相關食譜資料：
{context}

記住：必須用繁體中文回答。"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages = compose_prompts(user_id, system_content, user_content),
            temperature=0.1,  # 極低溫度增加一致性
            max_tokens=1024
        )
        
        result = response.choices[0].message.content.strip()
        
        # 檢查回應語言是否正確
        response_lang = detect_language(result)
        logger.info(f"AI回應語言: {response_lang}, 期望語言: {detected_lang}")
        
        # 如果語言不匹配，提供備用回應
        if detected_lang == 'en' and response_lang != 'en':
            logger.warning("AI未遵守英文指示，使用備用回應")
            # 嘗試簡單的備用英文回應
            if "大卡" in result or "熱量" in result:
                # 嘗試提取數字
                import re
                numbers = re.findall(r'\d+', result)
                if numbers:
                    calories = numbers[0]
                    return f"{primary_food} contains approximately {calories} calories per serving."
            return f"Based on the recipe database, {primary_food} is a dish in our collection. Please check the nutritional information for specific calorie details."
        
        return result
        
    except Exception as e:
        logger.error(f"生成卡路里回應時發生錯誤: {e}")
        if detected_lang == 'zh-TW':
            return "抱歉，無法取得卡路里資訊，請稍後再試。"
        else:
            return "Sorry, I couldn't retrieve calorie information at the moment. Please try again later."
# -------------------- RAG 查詢與除錯 --------------------
def query_rag(query, top_k=3, threshold=0.4, language="zh", debug=False):
    global embed_model, indexes, segments_map
    
    # 如果還沒初始化，就在這邊延遲載入
    if embed_model is None or not indexes:
        initialize_rag()
    
    # 如果是英文查詢，先嘗試翻譯成中文來搜尋
    search_query = query
    if language == "en":
        try:
            translation_messages = [
                {"role": "system", "content": "Translate the following English food/recipe query to Traditional Chinese. Only return the translation, no explanations."},
                {"role": "user", "content": query}
            ]
            resp = openai_client.chat.completions.create(
                model="gpt-4.1-mini", 
                messages=translation_messages, 
                temperature=0, 
                max_tokens=1024
            )
            search_query = resp.choices[0].message.content.strip()
            if debug:
                logger.info(f"Translated query: '{query}' -> '{search_query}'")
        except OpenAIError as e:
            logger.warning(f"Translation failed: {e}, using original query")
            # 如果翻譯失敗，仍使用原查詢
    
    # 編碼查詢（使用翻譯後的中文或原始查詢）
    q_emb = embed_model.encode([search_query])
    
    all_results = []
    # 搜尋所有分區
    for sec, idx in indexes.items():
        dists, ids = idx.search(np.array(q_emb, dtype="float32"), top_k)
        for dist, i in zip(dists[0], ids[0]):
            # 過濾有效結果和相似度閾值
            if i >= 0 and dist < threshold:
                all_results.append((dist, segments_map[sec][i], sec))
    
    # 按相似度排序並取前 top_k 筆
    all_results.sort(key=lambda x: x[0])
    retrieved_segments = [seg for _, seg, _ in all_results[:top_k]]
    
    # 除錯資訊
    if debug or len(retrieved_segments) == 0:
        logger.info(f"RAG Debug Info:")
        logger.info(f"Original query: {query}")
        logger.info(f"Search query: {search_query}")
        logger.info(f"Total results found: {len(all_results)}")
        logger.info(f"Results after filtering: {len(retrieved_segments)}")
        logger.info(f"Threshold: {threshold}")
        
        if len(all_results) > 0:
            logger.info(f"Top 5 similarity scores: {[round(dist, 3) for dist, _, _ in all_results[:5]]}")
            logger.info(f"Sections found: {[sec for _, _, sec in all_results[:5]]}")
        
        if len(retrieved_segments) == 0:
            logger.warning("No segments retrieved! Trying with higher threshold...")
            # 嘗試放寬閾值
            relaxed_results = []
            for sec, idx in indexes.items():
                dists, ids = idx.search(np.array(q_emb, dtype="float32"), top_k)
                for dist, i in zip(dists[0], ids[0]):
                    if i >= 0:  # 移除閾值限制
                        relaxed_results.append((dist, segments_map[sec][i], sec))
            
            relaxed_results.sort(key=lambda x: x[0])
            retrieved_segments = [seg for _, seg, _ in relaxed_results[:top_k]]
            logger.info(f"With relaxed threshold, found: {len(retrieved_segments)} segments")
            if len(relaxed_results) > 0:
                logger.info(f"Best similarity scores: {[round(dist, 3) for dist, _, _ in relaxed_results[:3]]}")
    
    return retrieved_segments
# -------------------- Flask + LINE Callback --------------------
app=Flask(__name__)

@app.route("/callback",methods=["POST"])
def callback():
    sig=request.headers.get("X-Line-Signature","")
    body=request.get_data(as_text=True)
    try:
        handler.handle(body,sig)
    except InvalidSignatureError:
        logger.warning("Invalid signature.")
        abort(400)
    return "OK"



@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    
    # 加入語言檢測和記錄
    user_language = detect_language(text)
    
    # 詳細的debug日誌
    logger.info(f"=== 新訊息處理開始 ===")
    logger.info(f"User ID: {user_id}")
    logger.info(f"訊息內容: '{text}'")
    logger.info(f"訊息長度: {len(text)}")
    logger.info(f"檢測到的語言: {user_language}")
    
    # 分開計算避免引號問題
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_english = any('a' <= c.lower() <= 'z' for c in text)
    
    logger.info(f"訊息包含中文: {'是' if has_chinese else '否'}")
    logger.info(f"訊息包含英文: {'是' if has_english else '否'}")
    
    # 記錄使用者訊息
    try:
        log_message(user_id, "user", text)
        logger.info("用戶訊息已記錄到Firebase")
    except Exception as e:
        logger.error(f"記錄用戶訊息失敗: {e}")

    # 檢查TDEE狀態機
    state = user_states.get(user_id)
    logger.info(f"用戶狀態: {state}")
    
    if state:
        logger.info("=== 處理TDEE狀態流程 ===")
        data = state["data"]
        stage = state["stage"]
        # ✅ 修復：從狀態中取得語言，如果沒有就預設為 "zh"
        language = user_language

        # 用於檢查這一輪 TDEE 輸入是否有效
        is_valid_tdee_input = False

        # 依照 stage 進行 TDEE 流程的多階段交互
        if stage == "ask_sex":
            if language == "en":
                if text.lower() in ["male", "m", "female", "f"]:
                    is_valid_tdee_input = True
                    data["sex"] = "male" if text.lower() in ["male", "m"] else "female"
                    user_states[user_id]["stage"] = "ask_age"
                    reply = "Please tell me your age"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    log_message(user_id, "assistant", reply)
                    return
            else:
                if text in ["男", "女"]:
                    is_valid_tdee_input = True
                    data["sex"] = "male" if text == "男" else "female"
                    user_states[user_id]["stage"] = "ask_age"
                    reply = "請告訴我年齡"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    log_message(user_id, "assistant", reply)
                    return

        elif stage == "ask_age":
            try:
                age = int(text)
                if 10 <= age <= 120:
                    is_valid_tdee_input = True
                    data["age"] = age
                    user_states[user_id]["stage"] = "ask_height"
                    reply = "Please tell me your height (cm)" if language == "en" else "請告訴我身高（公分）"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    log_message(user_id, "assistant", reply)
                    return
            except ValueError:
                pass

        elif stage == "ask_height":
            try:
                height = float(text)
                if 50 <= height <= 250:
                    is_valid_tdee_input = True
                    data["height"] = height
                    user_states[user_id]["stage"] = "ask_weight"
                    reply = "Please tell me your weight (kg)" if language == "en" else "請告訴我體重（公斤）"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    log_message(user_id, "assistant", reply)
                    return
            except ValueError:
                pass

        elif stage == "ask_weight":
            try:
                weight = float(text)
                if 20 <= weight <= 200:
                    is_valid_tdee_input = True
                    data["weight"] = weight
                    save_user_profile(user_id, {
                        "sex": data["sex"],
                        "age": data["age"],
                        "height": data["height"],
                        "weight": data["weight"]
                    })
                    user_states[user_id]["stage"] = "ask_activity"
                    reply = (
                        "Please select activity level: sedentary, light, moderate, active, very active"
                        if language == "en"
                        else "基本資料已儲存！請選擇活動量：靜態、輕度、中度、高度、劇烈"
                    )
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    log_message(user_id, "assistant", reply)
                    return
            except ValueError:
                pass

        elif stage == "ask_activity":
            activity_mapping = {
                "靜態": "靜態", "輕度": "輕度", "中度": "中度", "高度": "高度", "劇烈": "劇烈",
                "sedentary": "靜態", "light": "輕度", "moderate": "中度", "active": "高度", "very active": "劇烈"
            }
            activity_key = text.lower() if language == "en" else text
            if activity_key in activity_mapping:
                is_valid_tdee_input = True
                data["activity"] = activity_mapping[activity_key]
                user_states[user_id]["stage"] = "ask_goal"
                reply = (
                    "Please select your goal: maintain, lose weight, gain weight"
                    if language == "en"
                    else "請選擇目標：維持、減重、增重"
                )
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                log_message(user_id, "assistant", reply)
                return

        elif stage == "ask_goal":
            goal_mapping = {
                "維持": "維持", "減重": "減重", "增重": "增重",
                "maintain": "維持", "lose weight": "減重", "gain weight": "增重",
                "lose": "減重", "gain": "增重"
            }
            goal_key = text.lower() if language == "en" else text
            if goal_key in goal_mapping:
                is_valid_tdee_input = True
                data["goal"] = goal_mapping[goal_key]
                try:
                    tdee = calculate_tdee(
                        data["sex"], data["age"], data["height"], data["weight"],
                        data["activity"], data["goal"]
                    )
                    save_user_profile(user_id, {
                        "activity": data["activity"],
                        "goal": data["goal"],
                        "tdee": tdee
                    })

                    # ✅ 修復：使用語言自適應回應
                    if language == "en":
                        msg1 = TextSendMessage(text=f"Your TDEE is approximately {tdee} calories")
                        prompt = f"Based on my daily requirement of {tdee} calories, please recommend a daily three-meal food combination with calorie information for each meal, totaling no more than {tdee} calories."
                    else:
                        msg1 = TextSendMessage(text=f"你的 TDEE 約為 {tdee} 大卡")
                        prompt = f"請根據我每日需求 {tdee} 大卡，推薦一天三餐的食物組合，並註明每餐熱量，總和不超過 {tdee} 大卡。"

                    # ✅ 修復：使用語言自適應推薦
                    ans = calorie_recommend_adaptive(user_id, tdee, prompt)
                    msg2 = TextSendMessage(text=ans)

                    line_bot_api.reply_message(event.reply_token, [msg1, msg2])
                    log_message(user_id, "assistant", msg1.text)
                    log_message(user_id, "assistant", msg2.text)

                    user_states.pop(user_id)
                    return

                except ValueError as e:
                    reply = f"Calculation error: {str(e)}" if language == "en" else f"計算錯誤：{str(e)}"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                    log_message(user_id, "assistant", reply)
                    user_states.pop(user_id)
                    return

        # 如果輸入不屬於任何有效 TDEE 階段，清除狀態
        if not is_valid_tdee_input:
            logger.info("TDEE輸入無效，清除狀態")
            user_states.pop(user_id)

    # ✅ 修復：使用語言自適應意圖分類
    logger.info("=== 開始語言自適應意圖分類 ===")
    try:
        intent = detect_intent_adaptive(user_id, text)
        logger.info(f"意圖分類結果: intent='{intent}'")
    except Exception as e:
        logger.error(f"意圖分類發生錯誤: {e}")
        intent = "RECIPE"
        logger.info(f"使用預設值: intent='{intent}'")

    # ✅ 修復：TDEE 請求處理 - 使用語言自適應
    if intent == "TDEE":
        logger.info("=== 處理TDEE請求 ===")
        try:
            user_profile = get_user_profile(user_id)
            logger.info(f"用戶資料: {user_profile}")
            
            if has_basic_profile(user_profile):
                logger.info("用戶已有完整資料，直接詢問活動量")
                user_states[user_id] = {
                    "stage": "ask_activity",
                    "language": "zh",  # 預設中文，讓AI自適應
                    "data": {
                        "sex": user_profile["sex"],
                        "age": user_profile["age"],
                        "height": user_profile["height"],
                        "weight": user_profile["weight"]
                    }
                }
                
                # 語言自適應回應
                system_content = """
The user already has basic profile information and wants to calculate TDEE. 
Please respond in the SAME LANGUAGE as their question, asking for their activity level.

If Chinese: mention activity levels as 靜態、輕度、中度、高度、劇烈
If English: mention activity levels as sedentary, light, moderate, active, very active
"""
                try:
                    messages = compose_prompts(user_id, system_content, f"User said: '{text}'. They want to calculate TDEE...")
                    response = openai_client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1024
                    )
                    reply = response.choices[0].message.content.strip()
                except:
                    reply = "我已有你的基本資料。請告訴我你的活動量：靜態、輕度、中度、高度、劇烈"
            else:
                logger.info("用戶沒有基本資料，開始收集")
                user_states[user_id] = {"stage": "ask_sex", "language": user_language, "data": {}}
                
                # 語言自適應回應
                system_content = """
The user wants to calculate TDEE but doesn't have basic information yet.
Please respond in the SAME LANGUAGE as their question, asking for their gender first.

If Chinese: ask for 性別：男 或 女
If English: ask for gender: male or female
"""
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": f"User said: '{text}'. They want to calculate TDEE but have no profile."}
                        ],
                        temperature=0.3,
                        max_tokens=1024
                    )
                    reply = response.choices[0].message.content.strip()
                except:
                    reply = "我來幫你計算TDEE！首先，請告訴我你的性別：男 或 女"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            log_message(user_id, "assistant", reply)
            logger.info("TDEE請求處理完成")
            return
            
        except Exception as e:
            logger.error(f"處理TDEE請求時發生錯誤: {e}")
            # 語言自適應錯誤回應
            system_content = """
There was an error processing the TDEE request.
Please respond in the SAME LANGUAGE as the user's question with an appropriate error message.
"""
            messages = compose_prompts(user_id, system_content, f"User said: '{text}'. There was an error processing TDEE.")
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                error_reply = response.choices[0].message.content.strip()
            except:
                error_reply = "抱歉，處理TDEE請求時發生錯誤。"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_reply))
            log_message(user_id, "assistant", error_reply)
            return

    # ✅ 修復：熱量預算推薦 - 使用語言自適應
    calorie_pattern = re.compile(r"(\d+)\s*(大卡|calories)", re.IGNORECASE)
    if intent == "RECIPE" and calorie_pattern.search(text):
        logger.info("=== 處理語言自適應熱量預算推薦 ===")
        try:
            match = calorie_pattern.search(text)
            wanted_cal = int(match.group(1))
            logger.info(f"需求熱量: {wanted_cal}")
            
            reply = calorie_recommend_adaptive(user_id, wanted_cal, text)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            log_message(user_id, "assistant", reply)
            logger.info("語言自適應熱量推薦處理完成")
            return
            
        except Exception as e:
            logger.error(f"語言自適應熱量推薦錯誤: {e}")
            # 語言自適應錯誤回應
            system_content = """
There was an error processing the calorie recommendation request.
Please respond in the SAME LANGUAGE as the user's question.
"""
            try:
                messages = compose_prompts(user_id, system_content, f"User said: '{text}'. Error processing calorie recommendation.")
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                error_msg = response.choices[0].message.content.strip()
            except:
                error_msg = "抱歉，處理您的請求時發生錯誤。"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
            log_message(user_id, "assistant", error_msg)
            return

    # ✅ 修復：卡路里查詢 - 使用語言自適應
    if intent == "CALORIE_QUERY":
        logger.info("=== 處理語言自適應卡路里查詢 ===")
        try:
            ans = query_food_calories_adaptive(user_id, text)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ans))
            log_message(user_id, "assistant", ans)
            logger.info("語言自適應卡路里查詢處理完成")
            return
        except Exception as e:
            logger.error(f"語言自適應卡路里查詢錯誤: {e}")
            # 語言自適應錯誤回應
            system_content = """
There was an error processing the calorie query.
Please respond in the SAME LANGUAGE as the user's question.
"""
            try:
                messages = compose_prompts(user_id, system_content, f"User said: '{text}'. Error processing calorie query.")
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                error_msg = response.choices[0].message.content.strip()
            except:
                error_msg = "抱歉，處理您的卡路里查詢時發生錯誤。"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
            log_message(user_id, "assistant", error_msg)
            return

    # ✅ 修復：聊天回應 - 使用語言自適應
    if intent == "CHAT":
        logger.info("=== 處理語言自適應聊天 ===")
        try:
            ans = generate_chat_response_adaptive(user_id, text)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ans))
            log_message(user_id, "assistant", ans)
            logger.info("語言自適應聊天處理完成")
            return
        except Exception as e:
            logger.error(f"語言自適應聊天錯誤: {e}")

    # ✅ 修復：食譜查詢 - 使用語言自適應
    if intent == "RECIPE":
        logger.info("=== 處理語言自適應食譜查詢 ===")
        try:
            # 確保RAG系統已初始化
            global embed_model, indexes
            if embed_model is None or not indexes:
                logger.info("RAG系統未初始化，開始初始化...")
                initialize_rag()
            
            segs = query_rag(text, language="zh", debug=True)  # 仍用中文檢索
            logger.info(f"RAG檢索到 {len(segs)} 個片段")
            
            if segs:
                context = "\n\n".join(segs)
                system_content = """
You are a professional recipe assistant. Please provide recipe information based on the context provided.

IMPORTANT: Please respond in the SAME LANGUAGE as the user's question.

Provide detailed recipes with:
1. Ingredients (with quantities)  
2. Steps (numbered, clear instructions)
3. Tips (2-3 brief suggestions)

Keep the response well-structured and practical.
"""
                
                user_content = f"User question: {text}"
                messages = compose_prompts(user_id, system_content, f"Recipe context:\n{context}", user_content)
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1024
                )
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1024
                )
                ans = response.choices[0].message.content.strip()
            else:
                # 沒找到相關資料的語言自適應回應
                system_content = """
No relevant recipe information was found for the user's query.
Please respond in the SAME LANGUAGE as the user's question, suggesting they try other recipe-related questions.
"""
                try:
                    messages = compose_prompts(user_id, system_content, f"User asked: '{text}'. No recipe data found.")
                    response = openai_client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1024
                    )
                    ans = response.choices[0].message.content.strip()
                except:
                    ans = "抱歉，我在食譜資料庫中找不到相關的資訊。請嘗試詢問其他食譜相關問題。"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ans))
            log_message(user_id, "assistant", ans)
            logger.info("語言自適應食譜查詢處理完成")
            return
        except Exception as e:
            logger.error(f"語言自適應食譜查詢錯誤: {e}")

    # ✅ 修復：Fallback - 使用語言自適應
    logger.warning("=== 進入語言自適應Fallback處理 ===")
    logger.warning(f"意圖: {intent}, 但沒有被正確處理")
    
    system_content = """
The user's request couldn't be processed properly. 
Please respond in the SAME LANGUAGE as the user's question, politely saying you're not sure how to help and suggest asking about recipes or nutrition.
"""
    try:
        messages = compose_prompts(user_id, system_content, f"User said: '{text}'. Couldn't process their request.")
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        fallback_reply = response.choices[0].message.content.strip()
    except:
        fallback_reply = "抱歉，我不確定該如何回應。請嘗試詢問食譜或營養相關問題。"
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=fallback_reply))
    log_message(user_id, "assistant", fallback_reply)
    logger.info("=== 語言自適應訊息處理結束 ===")


def detect_intent_adaptive(user_id, text):
    """
    語言自適應的意圖分類 - 不預先判斷語言
    """
    logger.info(f"=== 語言自適應意圖檢測開始 ===")
    logger.info(f"輸入文本: '{text}'")

    # 先檢查明確的TDEE關鍵字（快速路徑）
    if re.search(r"\btdee\b", text, re.IGNORECASE):
        logger.info("檢測到TDEE關鍵字，直接返回TDEE")
        return "TDEE"

    # 使用語言自適應的 GPT 分類
    logger.info("使用語言自適應GPT進行意圖分類...")
    
    system_content = """
You are an intent classification assistant. You will receive a user query in ANY language (Chinese, English, etc.) and must classify it into one of four categories: TDEE, RECIPE, CALORIE_QUERY, or CHAT.

Please respond with exactly ONE word: TDEE, RECIPE, CALORIE_QUERY, or CHAT.

Category descriptions:
- TDEE: Questions about calculating daily caloric needs, basal metabolic rate, or TDEE
- RECIPE: Requests for recipes, cooking instructions, meal recommendations, food preparation
- CALORIE_QUERY: Questions about calorie content of specific foods
- CHAT: General conversation, greetings, unrelated topics

Examples in multiple languages:
TDEE:
- "What's my TDEE?" 
- "我的TDEE是多少？"
- "Calculate my daily calorie needs"
- "請計算我的基礎代謝率"

RECIPE:
- "How do I make pancakes?"
- "如何做鬆餅？"
- "500 calories meal ideas"
- "推薦我晚餐食譜"

CALORIE_QUERY:
- "How many calories in an apple?"
- "蘋果有多少卡路里？"
- "Calories in rice"
- "雞胸肉的熱量"

CHAT:
- "Hi, how are you?"
- "你好嗎？"
- "Thank you"
- "謝謝"
"""

    messages = compose_prompts(user_id, system_content, f"Classify this query: {text}")
    
    try:
        logger.info("正在呼叫OpenAI API...")
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=1024
        )
        
        raw_response = resp.choices[0].message.content.strip().upper()
        logger.info(f"GPT-4原始回應: '{raw_response}'")
        
        # 從回應中提取意圖
        m = re.search(r"\b(TDEE|RECIPE|CALORIE_QUERY|CHAT)\b", raw_response)
        if m:
            intent = m.group(1)
            logger.info(f"成功提取意圖: {intent}")
        else:
            logger.warning(f"無法從GPT-4回應中提取有效意圖，預設為RECIPE")
            intent = "RECIPE"
        
        logger.info(f"=== 語言自適應意圖檢測完成: {intent} ===")
        return intent

    except OpenAIError as e:
        logger.error(f"OpenAI API 呼叫失敗: {e}")
        intent = "RECIPE"
        logger.info(f"使用備用意圖: {intent}")
        return intent
    
    except Exception as e:
        logger.error(f"意圖檢測發生未預期錯誤: {e}")
        intent = "RECIPE"
        logger.info(f"使用備用意圖: {intent}")
        return intent



# ────────────────────────────────────────────────────────────────────────────────────────
# 以下新增 calorie_recommend_v2 函式，請貼在 handle_text 後面（與其他函式同一縮排層級）
# ────────────────────────────────────────────────────────────────────────────────────────
# ==================== 5. 語言自適應的卡路里推薦 ====================
def calorie_recommend_adaptive(user_id, wanted_cal, original_text):
    """
    語言自適應的熱量推薦函數
    """
    logger.info(f"語言自適應熱量推薦 - 目標卡路里: {wanted_cal}")
    
    # 檢測語言
    detected_lang = detect_language(original_text)
    logger.info(f"推薦查詢語言: {detected_lang}")
    
    # RAG 檢索 - 始終用中文檢索（因為資料庫是中文）
    rag_query = f"{wanted_cal} 大卡 推薦 食譜 名稱"
    segments = query_rag(rag_query, top_k=5, language="zh")
    
    if not segments:
        if detected_lang == 'zh-TW':
            return f"抱歉，我找不到符合 {wanted_cal} 大卡的餐點推薦。請試試其他熱量範圍。"
        else:
            return f"Sorry, I couldn't find any meal recommendations for {wanted_cal} calories. Please try a different calorie range."
    
    # 拼接上下文片段
    context_text = "\n\n".join(segments)
    
    # 根據語言設定不同提示
    if detected_lang == 'en':
        # 英文提示 - 加強版
        system_content = f"""You are a nutrition assistant recommending meals.

ABSOLUTE RULE: Respond ONLY in English. No Chinese characters in your response.

Task: From the Chinese recipe data provided, select and recommend 3 meals closest to {wanted_cal} calories.

Format your response as:
- [Meal name in English]: [X] calories
- [Meal name in English]: [X] calories  
- [Meal name in English]: [X] calories

Translate Chinese dish names to English."""
        
        user_content = f"""User requested: "{original_text}"

Available recipes (in Chinese - extract calorie info and translate names):
{context_text}

Remember: Output must be entirely in English."""
        
    else:
        # 中文提示
        system_content = f"""你是營養助理，為用戶推薦 {wanted_cal} 大卡左右的餐點。

規則：必須使用繁體中文，不可使用簡體中文。

從提供的食譜中選出3個最接近 {wanted_cal} 大卡的餐點，格式：
- 餐點名稱：XXX 大卡
- 餐點名稱：XXX 大卡
- 餐點名稱：XXX 大卡"""
        
        user_content = f"""用戶需求：{original_text}

可選食譜：
{context_text}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=compose_prompts(user_id, system_content, {"role": "user", "content": user_content}),
            temperature=0.1,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content.strip()
        
        # 檢查語言
        response_lang = detect_language(result)
        logger.info(f"推薦回應語言: {response_lang}, 期望: {detected_lang}")
        
        # 語言不匹配時的備用方案
        if detected_lang == 'en' and response_lang != 'en':
            logger.warning("AI未遵守英文指示，使用備用推薦")
            return f"Here are meal recommendations around {wanted_cal} calories:\n- Japanese Curry Rice: 600 calories\n- Fried Rice: 600 calories\n- Tomato Pasta: 550 calories"
        
        return result
        
    except Exception as e:
        logger.error(f"生成推薦時發生錯誤: {e}")
        if detected_lang == 'zh-TW':
            return "抱歉，無法取得推薦。請稍後再試。"
        else:
            return "Sorry, I couldn't retrieve recommendations. Please try again later."
