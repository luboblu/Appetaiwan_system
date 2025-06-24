"""
重構後的營養諮詢 LINE Bot
主要改進：
1. 模組化設計，分離關注點
2. 統一的錯誤處理機制
3. 改進的語言檢測和回應系統
4. 更清晰的狀態管理
5. 優化的 RAG 系統初始化
"""

import os
import re
import logging
import numpy as np
import faiss
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime


# ==================== 配置和常數 ====================

class Language(Enum):
    CHINESE = "zh-TW"
    ENGLISH = "en"

class Intent(Enum):
    TDEE = "TDEE"
    RECIPE = "RECIPE"
    CALORIE_QUERY = "CALORIE_QUERY"
    CALORIE_RECOMMENDATION = "CALORIE_RECOMMENDATION"
    CHAT = "CHAT"
    HELP = "HELP"  # 新增這行

class ResponseMode(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"

@dataclass
class UserProfile:
    sex: Optional[str] = None
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    activity: Optional[str] = None
    goal: Optional[str] = None
    tdee: Optional[int] = None
    updated_at: Optional[datetime] = None

@dataclass
class TDEEState:
    stage: str
    language: Language
    data: Dict[str, Any]


# ==================== 日誌設置 ====================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# ==================== 配置管理 ====================

class Config:
    def __init__(self):
        load_dotenv()
        self.validate_environment()
        
    def validate_environment(self):
        required_vars = [
            "FIREBASE_CREDENTIALS_JSON",
            "OPENAI_API_KEY", 
            "LINE_CHANNEL_ACCESS_TOKEN",
            "LINE_CHANNEL_SECRET"
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise RuntimeError(f"Missing environment variables: {missing}")
            
        firebase_path = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if not os.path.exists(firebase_path):
            raise RuntimeError(f"Firebase credentials file not found: {firebase_path}")
    
    @property
    def firebase_credentials_path(self) -> str:
        return os.getenv("FIREBASE_CREDENTIALS_JSON")
    
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def line_access_token(self) -> str:
        return os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    
    @property
    def line_secret(self) -> str:
        return os.getenv("LINE_CHANNEL_SECRET")


# ==================== 語言檢測模組 ====================

class LanguageDetector:
    @staticmethod
    def detect(text: str) -> Language:
        """檢測文本語言"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        logger.info(f"語言檢測 - 中文字符: {chinese_chars}, 英文字符: {english_chars}")
        
        if chinese_chars > 0:
            return Language.CHINESE
        elif english_chars > 0:
            return Language.ENGLISH
        else:
            return Language.CHINESE  # 預設中文


# ==================== Firebase 服務 ====================

class FirebaseService:
    def __init__(self, credentials_path: str):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
    
    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """儲存用戶資料"""
        try:
            doc_ref = self.db.collection("users").document(user_id)
            data = {**profile_data, "updatedAt": firestore.SERVER_TIMESTAMP}
            doc_ref.set(data, merge=True)
            logger.info(f"User profile saved for {user_id}")
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            raise
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """取得用戶資料"""
        try:
            doc_ref = self.db.collection("users").document(user_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                return UserProfile(**{k: v for k, v in data.items() 
                                   if k in UserProfile.__annotations__})
            return None
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def get_user_messages(self, user_id: str) -> List[Dict[str, Any]]:
        """取得用戶訊息歷史"""
        try:
            msg_ref = self.db.collection("users").document(user_id).collection("messages")
            docs = msg_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error getting user messages: {e}")
            return []
    
    def log_message(self, user_id: str, role: str, content: str):
        """記錄訊息"""
        try:
            msg_ref = self.db.collection("users").document(user_id).collection("messages")
            msg_ref.add({
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error logging message: {e}")


# ==================== TDEE 計算服務 ====================

class TDEECalculator:
    ACTIVITY_FACTORS = {
        "靜態": 1.20, "輕度": 1.375, "中度": 1.55, 
        "高度": 1.725, "劇烈": 1.90
    }
    
    ACTIVITY_MAPPING = {
        # 中文
        "靜態": "靜態", "輕度": "輕度", "中度": "中度", "高度": "高度", "劇烈": "劇烈",
        # 英文
        "sedentary": "靜態", "light": "輕度", "moderate": "中度", 
        "active": "高度", "very active": "劇烈"
    }
    
    GOAL_MAPPING = {
        # 中文
        "維持": "維持", "減重": "減重", "增重": "增重",
        # 英文
        "maintain": "維持", "lose weight": "減重", "gain weight": "增重",
        "lose": "減重", "gain": "增重"
    }
    
    @staticmethod
    def calculate(sex: str, age: int, height: float, weight: float, 
                  activity: str, goal: str) -> int:
        """計算 TDEE"""
        if not (10 <= age <= 120 and 50 <= height <= 250 and 20 <= weight <= 200):
            raise ValueError("輸入數值超出合理範圍")
        
        # 計算 BMR
        bmr = 10 * weight + 6.25 * height - 5 * age + (5 if sex == "male" else -161)
        
        # 計算 TDEE
        factor = TDEECalculator.ACTIVITY_FACTORS.get(activity, 1.20)
        tdee = bmr * factor
        
        # 根據目標調整
        if goal == "減重":
            tdee -= 500
        elif goal == "增重":
            tdee += 300
            
        return round(tdee)
    
    @staticmethod
    def has_complete_profile(profile: Optional[UserProfile]) -> bool:
        """檢查是否有完整的基本資料"""
        if not profile:
            return False
        return all([profile.sex, profile.age, profile.height, profile.weight])


# ==================== RAG 系統 ====================

class RAGSystem:
    def __init__(self, recipe_file: str = "recipes.txt"):
        self.recipe_file = recipe_file
        self.embed_model = None
        self.indexes = {}
        self.segments_map = {}
        self.partitions = None
        self._initialized = False
    
    def _load_and_partition_text(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """載入並分割文本"""
        with open(self.recipe_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parts = content.split("####")
        partitions = {}
        
        for sec in parts:
            sec = sec.strip()
            if not sec:
                continue
                
            header, body = (sec.split("\n", 1) if "\n" in sec else (sec, ""))
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
    
    def initialize(self):
        """延遲初始化 RAG 系統"""
        if self._initialized:
            return
            
        logger.info("開始初始化 RAG 系統...")
        
        # 載入分割文本
        self.partitions = self._load_and_partition_text()
        
        # 自動偵測設備
        device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
        logger.info(f"使用設備: {device}")
        
        # 載入 embedding 模型
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
        
        # 建立每個分區的索引
        for sec, texts in self.partitions.items():
            logger.info(f"建立索引: {sec} ({len(texts)} 個片段)")
            embs = self.embed_model.encode(texts, show_progress_bar=False)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs, dtype="float32"))
            self.indexes[sec] = idx
            self.segments_map[sec] = texts
        
        self._initialized = True
        logger.info("RAG 系統初始化完成！")
    
    def query(self, query: str, top_k: int = 3, threshold: float = 0.4, 
              language: Language = Language.CHINESE, debug: bool = False) -> List[str]:
        """查詢相關片段"""
        if not self._initialized:
            self.initialize()
        
        search_query = query
        # 如果是英文查詢，可以考慮翻譯（這裡簡化處理）
        
        # 編碼查詢
        q_emb = self.embed_model.encode([search_query])
        
        all_results = []
        # 搜尋所有分區
        for sec, idx in self.indexes.items():
            dists, ids = idx.search(np.array(q_emb, dtype="float32"), top_k)
            for dist, i in zip(dists[0], ids[0]):
                if i >= 0 and dist < threshold:
                    all_results.append((dist, self.segments_map[sec][i], sec))
        
        # 按相似度排序並取前 top_k 筆
        all_results.sort(key=lambda x: x[0])
        retrieved_segments = [seg for _, seg, _ in all_results[:top_k]]
        
        # 除錯資訊
        if debug or len(retrieved_segments) == 0:
            logger.info(f"RAG Debug - Query: {query}, Found: {len(retrieved_segments)} segments")
            if len(all_results) > 0:
                logger.info(f"Top similarities: {[round(dist, 3) for dist, _, _ in all_results[:3]]}")
        
        return retrieved_segments


# ==================== 意圖檢測服務 ====================

class IntentDetector:
    def __init__(self, openai_client: OpenAI, firebase_service: FirebaseService):
        self.openai_client = openai_client
        self.firebase_service = firebase_service
    
    def detect(self, user_id: str, text: str) -> Intent:
        """使用純 GPT 進行意圖檢測"""
        logger.info(f"=== 開始 GPT 意圖檢測 ===")
        logger.info(f"輸入文本: '{text}'")
        
        return self._gpt_classification(user_id, text)
    
    def _gpt_classification(self, user_id: str, text: str) -> Intent:
        """使用 GPT 進行精細的意圖分類"""
        system_content = """
你是專業的意圖分類助理。請將用戶查詢精確分類為以下其中一種意圖：TDEE, RECIPE, CALORIE_QUERY, CALORIE_RECOMMENDATION, CHAT, HELP

重要：請只回答一個詞，必須是上述六個選項之一。

== 意圖定義及範例 ==

**TDEE**
計算每日熱量需求、基礎代謝率、TDEE相關
範例：「計算我的TDEE」、「我的基礎代謝率」、「每日熱量需求」

**CALORIE_RECOMMENDATION** 
根據特定卡路里目標或範圍要求餐點推薦、食物建議
關鍵特徵：數字+卡路里+推薦/建議語境，或減重/增重餐點建議
範例：
- 「請給我2000大卡以內的食物推薦」
- 「500大卡的早餐有什麼推薦」
- 「減重期間適合吃什麼」

**RECIPE**
具體食譜查詢、烹飪方法、製作步驟
範例：「蛋炒飯怎麼做」、「How to make pancakes」、「雞肉料理食譜」

**CALORIE_QUERY**
詢問特定食物的卡路里含量、營養成分
範例：「蘋果有多少卡路里」、「How many calories in rice」、「雞胸肉熱量」

**HELP**
詢問機器人功能、使用說明、能做什麼
範例：「你能做什麼」、「功能介紹」、「怎麼使用」、「What can you do」、「Help」、「使用說明」

**CHAT**
一般對話、問候、感謝、無關話題
範例：「你好」、「謝謝」、「Hello」、「How are you」

== 分類指引 ==
1. 重點看整體意圖，不只是關鍵字
2. 推薦 vs 食譜區別：
   - CALORIE_RECOMMENDATION：「給我低卡餐點推薦」（要建議）
   - RECIPE：「低卡餐點怎麼做」（要做法）
3. 查詢 vs 推薦區別：
   - CALORIE_QUERY：「雞肉有多少卡路里」（要資訊）
   - CALORIE_RECOMMENDATION：「推薦低卡雞肉料理」（要建議）
4. 功能詢問優先：如果用戶詢問機器人能做什麼或需要幫助，優先選擇 HELP

記住：只回答一個詞：TDEE, RECIPE, CALORIE_QUERY, CALORIE_RECOMMENDATION, CHAT, 或 HELP
"""
        
        messages = self._compose_prompts(user_id, system_content, f"請分類這個查詢：{text}")
        
        try:
            logger.info("正在呼叫 GPT 進行意圖分類...")
            resp = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=10,
                top_p=0.9
            )
            
            raw_response = resp.choices[0].message.content.strip().upper()
            logger.info(f"GPT 原始回應: '{raw_response}'")
            
            # 提取有效意圖
            valid_intents = [intent.value for intent in Intent]
            
            # 嘗試直接匹配
            if raw_response in valid_intents:
                detected_intent = Intent(raw_response)
                logger.info(f"✅ 直接匹配成功: {detected_intent.value}")
                return detected_intent
            
            # 嘗試在回應中查找有效意圖
            for intent_value in valid_intents:
                if intent_value in raw_response:
                    detected_intent = Intent(intent_value)
                    logger.info(f"✅ 部分匹配成功: {detected_intent.value}")
                    return detected_intent
            
            # 如果都沒匹配到，記錄警告並使用預設值
            logger.warning(f"⚠️ 無法從 GPT 回應中提取有效意圖: '{raw_response}'")
            logger.info("使用預設意圖: RECIPE")
            return Intent.RECIPE
                
        except Exception as e:
            logger.error(f"❌ GPT 意圖分類錯誤: {e}")
            return Intent.RECIPE
    
    def _compose_prompts(self, user_id: str, system_prompt: str, *rest_prompts) -> List[Dict[str, str]]:
        """組合提示詞"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # 加入歷史訊息
        user_messages = self.firebase_service.get_user_messages(user_id)
        for msg in user_messages:
            role = "assistant" if msg["role"] == "bot" else msg["role"]
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": msg["content"]})
        
        # 加入新的提示
        for prompt in rest_prompts:
            if isinstance(prompt, dict):
                if prompt.get("role") in ("system", "user", "assistant"):
                    messages.append(prompt)
                else:
                    messages.append({"role": "user", "content": str(prompt)})
            else:
                messages.append({"role": "user", "content": str(prompt)})
        
        return messages


# ==================== 回應生成服務 ====================

class ResponseGenerator:
    def __init__(self, openai_client: OpenAI, firebase_service: FirebaseService, rag_system: RAGSystem):
        self.openai_client = openai_client
        self.firebase_service = firebase_service
        self.rag_system = rag_system
        self.intent_detector = IntentDetector(openai_client, firebase_service)
    
    def generate_chat_response(self, user_id: str, text: str, language: Language) -> str:
        """生成聊天回應"""
        if language == Language.CHINESE:
            system_content = """
你是專業食物推薦助理。除了提供營養和食譜建議，你也可以進行簡單的日常對話。

重要語言規則：
- 必須使用繁體中文回答，不可使用簡體中文
- 這是不可協商的，無論用戶使用什麼語言

翻譯指引：
- 如果提到食物名稱，請使用常見的中文名稱
- 保持對話自然和友善

指導原則：
- 保持友善和簡潔的回應風格
- 適當時提醒用戶你的專長是食物和營養協助
- 回應控制在2-3句話內
"""
        else:
            system_content = """
You are a professional recipe recommendation assistant. Besides providing nutrition and recipe advice, you can also engage in simple daily conversations.

CRITICAL LANGUAGE RULES:
- You MUST respond in English
- This is non-negotiable, regardless of what language the user uses

Translation Guidelines:
- If mentioning food names, use common English food names
- Keep conversation natural and friendly

Guidelines:
- Maintain a friendly and concise response style
- Remind users when appropriate that your expertise is in food and nutrition assistance
- Keep responses to 2-3 sentences
"""
        
        messages = self.intent_detector._compose_prompts(user_id, system_content, text)
        
        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"聊天回應錯誤: {e}")
            if language == Language.CHINESE:
                return "抱歉，我現在無法回應。請稍後再試。"
            else:
                return "Sorry, I cannot respond right now. Please try again later."
    
    def generate_calorie_query_response(self, user_id: str, text: str, language: Language) -> str:
        """生成卡路里查詢回應"""
        self.rag_system.initialize()
        
        # 提取食物名稱（簡化版）
        food_extraction = self._extract_food_name(user_id, text)
        primary_food = food_extraction.get("primary_food", text)
        
        # RAG 檢索
        search_query = f"{primary_food} 卡路里 熱量 營養"
        segments = self.rag_system.query(search_query, top_k=5)
        
        if not segments:
            if language == Language.CHINESE:
                return f"抱歉，我在資料庫中找不到關於「{primary_food}」的卡路里資訊。"
            else:
                return f"Sorry, I couldn't find calorie information for '{primary_food}'."
        
        # 生成回應
        context = "\n\n".join(segments)
        system_content = self._get_calorie_system_prompt(language, primary_food)
        user_content = f"Question: {text}\n\nContext:\n{context}"
        
        messages = self.intent_detector._compose_prompts(user_id, system_content, user_content)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Calorie query error: {e}")
            if language == Language.CHINESE:
                return "抱歉，無法取得卡路里資訊，請稍後再試。"
            else:
                return "Sorry, I couldn't retrieve calorie information. Please try again later."
    
    def generate_recipe_response(self, user_id: str, text: str, language: Language) -> str:
        """生成食譜回應"""
        self.rag_system.initialize()
        
        # RAG 檢索
        segments = self.rag_system.query(text, top_k=5, debug=True)
        
        if not segments:
            if language == Language.CHINESE:
                return "抱歉，我在食譜資料庫中找不到相關的資訊。請嘗試詢問其他食譜相關問題。"
            else:
                return "Sorry, I couldn't find relevant recipe information. Please try other recipe-related questions."
        
        # 生成回應
        context = "\n\n".join(segments)
        system_content = self._get_recipe_system_prompt(language)
        user_content = f"User question: {text}\n\nRecipe context:\n{context}"
        
        messages = self.intent_detector._compose_prompts(user_id, system_content, user_content)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.5,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Recipe response error: {e}")
            if language == Language.CHINESE:
                return "抱歉，無法生成食譜回應，請稍後再試。"
            else:
                return "Sorry, I couldn't generate a recipe response. Please try again later."
    
    def generate_calorie_recommendation_advanced(self, user_id: str, original_text: str, 
                                               preferences: Dict[str, Any], language: Language) -> str:
        """基於 GPT 提取的偏好生成進階卡路里推薦"""
        logger.info(f"🎯 生成進階卡路里推薦")
        logger.info(f"用戶偏好: {preferences}")
        
        self.rag_system.initialize()
        
        # 根據偏好構建 RAG 查詢
        rag_query = self._build_rag_query_from_preferences(preferences, language)
        logger.info(f"構建的 RAG 查詢: {rag_query}")
        
        # RAG 檢索
        segments = self.rag_system.query(rag_query, top_k=10, threshold=0.5)
        
        if not segments:
            # 備用查詢
            fallback_query = "推薦 餐點 健康" if language == Language.CHINESE else "recommend meal healthy"
            segments = self.rag_system.query(fallback_query, top_k=5, threshold=0.6)
        
        if not segments:
            if language == Language.CHINESE:
                return "抱歉，我找不到符合您需求的餐點推薦。請試試重新描述您的需求。"
            else:
                return "Sorry, I couldn't find meal recommendations that match your needs. Please try rephrasing your request."
        
        # 構建上下文
        context = "\n\n".join(segments)
        
        # 生成系統提示
        system_content = self._get_advanced_recommendation_prompt(preferences, language)
        
        if language == Language.CHINESE:
            user_content = f"""用戶原始請求：{original_text}

解析的需求偏好：
{self._format_preferences_for_display(preferences, language)}

可選食譜資料：
{context}

請根據用戶的具體需求和偏好，提供個性化的餐點推薦。"""
        else:
            user_content = f"""User's original request: {original_text}

Parsed preferences:
{self._format_preferences_for_display(preferences, language)}

Available recipe data:
{context}

Please provide personalized meal recommendations based on the user's specific needs and preferences."""
        
        messages = self.intent_detector._compose_prompts(user_id, system_content, user_content)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"✅ 進階推薦生成成功，長度: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 生成進階推薦錯誤: {e}")
            # 降級到基礎推薦
            return self.generate_calorie_recommendation(user_id, 0, original_text, language)
    def generate_help_response(self, user_id: str, text: str, language: Language) -> str:
        """生成功能介紹回應"""
    
        if language == Language.CHINESE:
            return """🤖 **營養諮詢助理功能介紹**

我是你的專業營養助理，可以幫你：

🔥 **TDEE 計算**
- 計算每日熱量需求
- 個人化基礎代謝率分析
- 根據目標調整熱量建議
範例：「計算我的TDEE」

🍳 **食譜查詢**
- 各種料理製作方法
- 詳細烹飪步驟指導
- 料理小技巧分享
範例：「蛋炒飯怎麼做」

📊 **食物熱量查詢**
- 查詢特定食物卡路里
- 營養成分分析
- 食物比較建議
範例：「蘋果有多少卡路里」

🥗 **個人化餐點推薦**
- 依熱量需求推薦餐點
- 客製化飲食建議
- 減重/增重餐點組合
範例：「推薦500大卡的早餐」

💬 **營養諮詢對話**
- 回答營養相關問題
- 飲食建議和指導
- 友善互動交流

---
**使用小提示：**
- 直接輸入你的問題即可
- 支援中文和英文
- 可以進行多輪對話

有什麼營養問題都可以問我！ 😊"""
    
        else:
            return """🤖 **Nutrition Assistant Features**

I'm your professional nutrition assistant, here to help with:

🔥 **TDEE Calculation**
- Calculate daily calorie requirements
- Personalized basal metabolic rate analysis
- Goal-adjusted calorie recommendations
Example: "Calculate my TDEE"

🍳 **Recipe Queries**
- Various cooking methods
- Detailed step-by-step instructions
- Cooking tips and tricks
Example: "How to make fried rice"

📊 **Food Calorie Queries**
- Check specific food calories
- Nutritional content analysis
- Food comparison suggestions
Example: "How many calories in an apple"

🥗 **Personalized Meal Recommendations**
- Meal suggestions based on calorie needs
- Customized dietary advice
- Weight loss/gain meal combinations
Example: "Recommend 500-calorie breakfast"

💬 **Nutrition Consultation**
- Answer nutrition-related questions
- Dietary guidance and advice
- Friendly interactive conversations

---
**Usage Tips:**
- Simply type your questions
- Supports Chinese and English
- Multi-turn conversations available

Feel free to ask me any nutrition questions! 😊"""
    def _build_rag_query_from_preferences(self, preferences: Dict[str, Any], language: Language) -> str:
        """根據偏好構建 RAG 查詢"""
        query_parts = []
        
        # 卡路里相關
        if preferences.get("target_calories"):
            query_parts.append(str(preferences["target_calories"]))
        
        calorie_range = preferences.get("calorie_range")
        if calorie_range and calorie_range.get("max"):
            query_parts.append(str(calorie_range["max"]))
        
        # 基礎詞彙
        if language == Language.CHINESE:
            query_parts.extend(["大卡", "推薦", "餐點", "食物"])
        else:
            query_parts.extend(["calories", "recommend", "meal", "food"])
        
        # 餐別
        meal_type = preferences.get("meal_type")
        if meal_type and meal_type != "null":
            if language == Language.CHINESE:
                meal_mapping = {
                    "早餐": "早餐", "午餐": "午餐", "晚餐": "晚餐", 
                    "點心": "點心", "全日三餐": "三餐"
                }
                mapped_meal = meal_mapping.get(meal_type, meal_type)
                query_parts.append(mapped_meal)
            else:
                query_parts.append(meal_type)
        
        # 飲食目標
        dietary_goal = preferences.get("dietary_goal")
        if dietary_goal and dietary_goal != "null":
            if language == Language.CHINESE:
                goal_mapping = {
                    "減重": "減重 低卡", "增重": "增重 高卡", 
                    "維持": "健康", "健康": "健康"
                }
                mapped_goal = goal_mapping.get(dietary_goal, dietary_goal)
                query_parts.append(mapped_goal)
            else:
                goal_mapping = {
                    "weight_loss": "weight loss low calorie",
                    "weight_gain": "weight gain high calorie",
                    "maintain": "healthy", "healthy": "healthy"
                }
                mapped_goal = goal_mapping.get(dietary_goal, dietary_goal)
                query_parts.append(mapped_goal)
        
        # 偏好關鍵字
        preference_keywords = preferences.get("preference_keywords", [])
        query_parts.extend(preference_keywords)
        
        return " ".join(query_parts)
    
    def _get_advanced_recommendation_prompt(self, preferences: Dict[str, Any], language: Language) -> str:
        """生成進階推薦的系統提示"""
        if language == Language.CHINESE:
            return f"""你是專業的個性化營養推薦助理。請根據用戶的具體偏好，從提供的食譜資料中推薦最合適的餐點。

重要：必須使用繁體中文回答，不可使用簡體中文。

翻譯指引：
- 如果食譜資料是其他語言，請將餐點名稱翻譯成常見的中文名稱
- 食材和烹飪方法請使用台灣常見的說法
- 保持卡路里數值和營養資訊的準確性
- 確保翻譯後的餐點名稱容易理解和記憶

回應格式要求：
🎯 **個性化餐點推薦**

**推薦理由**：簡要說明為什麼這些推薦符合用戶需求

**選項 1：** [餐點中文名稱] (~X大卡)
- 詳細描述為何適合用戶偏好
- 營養特點說明
- 主要食材：[翻譯成中文的食材列表]

**選項 2：** [餐點中文名稱] (~X大卡)  
- 詳細描述為何適合用戶偏好
- 營養特點說明
- 主要食材：[翻譯成中文的食材列表]

**選項 3：** [餐點中文名稱] (~X大卡)
- 詳細描述為何適合用戶偏好  
- 營養特點說明
- 主要食材：[翻譯成中文的食材列表]

**額外建議**：
- 如何調整以更符合目標
- 搭配建議或注意事項

保持推薦個性化且實用，重點突出為什麼這些選擇適合用戶的具體需求。"""
        else:
            return f"""You are a professional personalized nutrition recommendation assistant. Please recommend the most suitable meals from the provided recipe data based on user's specific preferences.

IMPORTANT: You MUST respond in English only.

Translation Guidelines:
- If the recipe data is in other languages (especially Chinese), translate ALL dish names to English
- Translate ingredients and cooking methods to common English terms
- Maintain accuracy of calorie values and nutritional information
- Ensure translated dish names are easy to understand and recognizable
- Use standard English culinary terminology

Response format requirements:
🎯 **Personalized Meal Recommendations**

**Why these recommendations**: Brief explanation of how they match user needs

**Option 1:** [English Dish Name] (~X calories)
- Detailed explanation of why it fits user preferences
- Nutritional highlights
- Main ingredients: [translated ingredient list in English]

**Option 2:** [English Dish Name] (~X calories)
- Detailed explanation of why it fits user preferences  
- Nutritional highlights
- Main ingredients: [translated ingredient list in English]

**Option 3:** [English Dish Name] (~X calories)
- Detailed explanation of why it fits user preferences
- Nutritional highlights
- Main ingredients: [translated ingredient list in English]

**Additional suggestions**:
- How to adjust to better meet goals
- Pairing recommendations or tips

Keep recommendations personalized and practical, focusing on why these choices suit the user's specific needs."""
    
    def _format_preferences_for_display(self, preferences: Dict[str, Any], language: Language) -> str:
        """格式化偏好資訊用於顯示"""
        if not preferences or preferences.get("raw_text"):
            return preferences.get("raw_text", "無法解析偏好")
        
        import json
    def generate_calorie_recommendation(self, user_id: str, wanted_cal: int, 
                                      original_text: str, language: Language) -> str:
        """生成基礎卡路里推薦（備用方法）"""
        logger.info(f"生成基礎卡路里推薦: {wanted_cal} 大卡")
        
        self.rag_system.initialize()
        
        # RAG 檢索
        if language == Language.CHINESE:
            rag_query = f"{wanted_cal} 大卡 推薦 餐點 食物"
        else:
            rag_query = f"{wanted_cal} calories recommend meal food"
            
        segments = self.rag_system.query(rag_query, top_k=5)
        
        if not segments:
            if language == Language.CHINESE:
                return f"抱歉，我找不到符合 {wanted_cal} 大卡的餐點推薦。請試試其他熱量範圍。"
            else:
                return f"Sorry, I couldn't find meal recommendations for {wanted_cal} calories. Please try a different range."
        
        # 生成回應
        context = "\n\n".join(segments)
        system_content = self._get_basic_recommendation_prompt(language, wanted_cal)
        
        if language == Language.CHINESE:
            user_content = f"用戶需求：{original_text}\n\n可選食譜：\n{context}"
        else:
            user_content = f"User request: {original_text}\n\nAvailable recipes:\n{context}"
        
        messages = self.intent_detector._compose_prompts(user_id, system_content, user_content)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"基礎推薦錯誤: {e}")
            if language == Language.CHINESE:
                return f"抱歉，無法生成 {wanted_cal} 大卡的推薦。請稍後再試。"
            else:
                return f"Sorry, I couldn't generate recommendations for {wanted_cal} calories. Please try again later."
    
    def _get_basic_recommendation_prompt(self, language: Language, wanted_cal: int) -> str:
        """獲取基礎推薦系統提示"""
        if language == Language.CHINESE:
            return f"""你是專業營養推薦助理。用戶想要 {wanted_cal} 大卡左右的餐點推薦。

重要：必須使用繁體中文回答，不可使用簡體中文。

翻譯指引：
- 如果食譜資料是其他語言，請將餐點名稱翻譯成常見的中文名稱
- 食材名稱請使用台灣常見的說法
- 保持卡路里數值的準確性

任務：根據提供的食譜資料，推薦符合用戶需求的餐點組合。

回應格式：
🍽️ **餐點推薦**

**推薦 1：** [餐點中文名稱] (~XXX大卡)
- 簡短描述和營養特點
- 主要食材：[翻譯後的食材]

**推薦 2：** [餐點中文名稱] (~XXX大卡)
- 簡短描述和營養特點
- 主要食材：[翻譯後的食材]

**推薦 3：** [餐點中文名稱] (~XXX大卡)
- 簡短描述和營養特點
- 主要食材：[翻譯後的食材]

保持推薦實用且營養均衡。"""
        else:
            return f"""You are a professional nutrition recommendation assistant. The user wants meal recommendations around {wanted_cal} calories.

IMPORTANT: You MUST respond in English only.

Translation Guidelines:
- If the recipe data is in other languages (especially Chinese), translate ALL dish names to English
- Translate ingredient names to common English terms
- Maintain accuracy of calorie values
- Use recognizable English dish names

Task: Based on the provided recipe data, recommend meal combinations that meet the user's needs.

Response format:
🍽️ **Meal Recommendations**

**Option 1:** [English Dish Name] (~XXX calories)
- Brief description and nutritional highlights
- Main ingredients: [translated ingredients in English]

**Option 2:** [English Dish Name] (~XXX calories)
- Brief description and nutritional highlights
- Main ingredients: [translated ingredients in English]

**Option 3:** [English Dish Name] (~XXX calories)
- Brief description and nutritional highlights
- Main ingredients: [translated ingredients in English]

Keep recommendations practical and nutritionally balanced."""
    
    def _extract_food_name(self, user_id: str, text: str) -> Dict[str, Any]:
        """提取食物名稱（簡化版）"""
        # 這裡可以實現更複雜的食物名稱提取邏輯
        return {"primary_food": text, "confidence": 0.8}
    
    def _get_calorie_system_prompt(self, language: Language, food_name: str) -> str:
        """獲取卡路里查詢系統提示"""
        if language == Language.CHINESE:
            return f"""你是營養資訊助理。請根據提供的食譜上下文，回答關於「{food_name}」的卡路里問題。
保持簡潔，2-3句話即可。必須使用繁體中文回答。"""
        else:
            return f"""You are a nutrition assistant. Answer calorie questions about '{food_name}' based on the recipe context.
Keep it concise, 2-3 sentences. You MUST respond in English."""
    
    def _get_recipe_system_prompt(self, language: Language) -> str:
        """獲取食譜查詢系統提示"""
        if language == Language.CHINESE:
            return """你是專業食譜助理。請根據上下文提供詳細食譜，包含：
1. 材料（含份量）
2. 步驟（編號說明）  
3. 小技巧（2-3行）
必須使用繁體中文回答。"""
        else:
            return """You are a professional recipe assistant. Provide detailed recipes with:
1. Ingredients (with quantities)
2. Steps (numbered instructions)
3. Tips (2-3 brief suggestions)
You MUST respond in English."""
    
    def _get_recommendation_system_prompt(self, language: Language, wanted_cal: int) -> str:
        """獲取推薦系統提示"""
        if language == Language.CHINESE:
            return f"""你是營養助理，推薦 {wanted_cal} 大卡左右的餐點。
格式：
- 餐點名稱：XXX 大卡
- 餐點名稱：XXX 大卡
- 餐點名稱：XXX 大卡
必須使用繁體中文。"""
        else:
            return f"""You are a nutrition assistant recommending meals around {wanted_cal} calories.
Format:
- [Meal name]: [X] calories
- [Meal name]: [X] calories  
- [Meal name]: [X] calories
You MUST respond in English."""


# ==================== TDEE 處理服務 ====================

class TDEEHandler:
    def __init__(self, openai_client: OpenAI, firebase_service: FirebaseService, 
                 response_generator: ResponseGenerator):
        self.openai_client = openai_client
        self.firebase_service = firebase_service
        self.response_generator = response_generator
        self.user_states: Dict[str, TDEEState] = {}
    
    def handle_tdee_request(self, user_id: str, text: str, language: Language) -> Tuple[str, bool]:
        """處理 TDEE 請求，返回 (回應文本, 是否完成)"""
        user_profile = self.firebase_service.get_user_profile(user_id)
        
        if TDEECalculator.has_complete_profile(user_profile):
            # 有基本資料，直接詢問活動量
            self.user_states[user_id] = TDEEState(
                stage="ask_activity",
                language=language,
                data={
                    "sex": user_profile.sex,
                    "age": user_profile.age, 
                    "height": user_profile.height,
                    "weight": user_profile.weight
                }
            )
            
            if language == Language.CHINESE:
                return "我已有你的基本資料。請告訴我你的活動量：靜態、輕度、中度、高度、劇烈", False
            else:
                return "I have your basic information. Please select activity level: sedentary, light, moderate, active, very active", False
        else:
            # 沒有基本資料，開始收集
            self.user_states[user_id] = TDEEState(
                stage="ask_sex",
                language=language,
                data={}
            )
            
            if language == Language.CHINESE:
                return "我來幫你計算TDEE！首先，請告訴我你的性別：男 或 女", False
            else:
                return "I'll help you calculate TDEE! First, please tell me your gender: male or female", False
    
    def handle_tdee_input(self, user_id: str, text: str) -> Tuple[Optional[str], bool]:
        """處理 TDEE 輸入，返回 (回應文本, 是否完成)"""
        if user_id not in self.user_states:
            return None, False
        
        state = self.user_states[user_id]
        language = state.language
        
        try:
            if state.stage == "ask_sex":
                return self._handle_sex_input(user_id, text, state, language)
            elif state.stage == "ask_age":
                return self._handle_age_input(user_id, text, state, language)
            elif state.stage == "ask_height":
                return self._handle_height_input(user_id, text, state, language)
            elif state.stage == "ask_weight":
                return self._handle_weight_input(user_id, text, state, language)
            elif state.stage == "ask_activity":
                return self._handle_activity_input(user_id, text, state, language)
            elif state.stage == "ask_goal":
                return self._handle_goal_input(user_id, text, state, language)
        except Exception as e:
            logger.error(f"TDEE handling error: {e}")
            self.user_states.pop(user_id, None)
            
            if language == Language.CHINESE:
                return "處理過程中發生錯誤，請重新開始。", True
            else:
                return "An error occurred during processing. Please start over.", True
        
        return None, False
    
    def _handle_sex_input(self, user_id: str, text: str, state: TDEEState, language: Language) -> Tuple[str, bool]:
        valid_inputs = {
            Language.CHINESE: ["男", "女"],
            Language.ENGLISH: ["male", "m", "female", "f"]
        }
        
        if text.lower() if language == Language.ENGLISH else text in valid_inputs[language]:
            if language == Language.ENGLISH:
                state.data["sex"] = "male" if text.lower() in ["male", "m"] else "female"
            else:
                state.data["sex"] = "male" if text == "男" else "female"
            
            state.stage = "ask_age"
            
            if language == Language.CHINESE:
                return "請告訴我年齡", False
            else:
                return "Please tell me your age", False
        
        return None, False
    
    def _handle_age_input(self, user_id: str, text: str, state: TDEEState, language: Language) -> Tuple[str, bool]:
        try:
            age = int(text)
            if 10 <= age <= 120:
                state.data["age"] = age
                state.stage = "ask_height"
                
                if language == Language.CHINESE:
                    return "請告訴我身高（公分）", False
                else:
                    return "Please tell me your height (cm)", False
        except ValueError:
            pass
        
        return None, False
    
    def _handle_height_input(self, user_id: str, text: str, state: TDEEState, language: Language) -> Tuple[str, bool]:
        try:
            height = float(text)
            if 50 <= height <= 250:
                state.data["height"] = height
                state.stage = "ask_weight"
                
                if language == Language.CHINESE:
                    return "請告訴我體重（公斤）", False
                else:
                    return "Please tell me your weight (kg)", False
        except ValueError:
            pass
        
        return None, False
    
    def _handle_weight_input(self, user_id: str, text: str, state: TDEEState, language: Language) -> Tuple[str, bool]:
        try:
            weight = float(text)
            if 20 <= weight <= 200:
                state.data["weight"] = weight
                
                # 儲存基本資料
                self.firebase_service.save_user_profile(user_id, {
                    "sex": state.data["sex"],
                    "age": state.data["age"],
                    "height": state.data["height"],
                    "weight": state.data["weight"]
                })
                
                state.stage = "ask_activity"
                
                if language == Language.CHINESE:
                    return "基本資料已儲存！請選擇活動量：靜態、輕度、中度、高度、劇烈", False
                else:
                    return "Basic information saved! Please select activity level: sedentary, light, moderate, active, very active", False
        except ValueError:
            pass
        
        return None, False
    
    def _handle_activity_input(self, user_id: str, text: str, state: TDEEState, language: Language) -> Tuple[str, bool]:
        activity_key = text.lower() if language == Language.ENGLISH else text
        if activity_key in TDEECalculator.ACTIVITY_MAPPING:
            state.data["activity"] = TDEECalculator.ACTIVITY_MAPPING[activity_key]
            state.stage = "ask_goal"
            
            if language == Language.CHINESE:
                return "請選擇目標：維持、減重、增重", False
            else:
                return "Please select your goal: maintain, lose weight, gain weight", False
        
        return None, False
    
    def _handle_goal_input(self, user_id: str, text: str, state: TDEEState, language: Language) -> Tuple[str, bool]:
        goal_key = text.lower() if language == Language.ENGLISH else text
        if goal_key in TDEECalculator.GOAL_MAPPING:
            state.data["goal"] = TDEECalculator.GOAL_MAPPING[goal_key]
            
            # 計算 TDEE
            tdee = TDEECalculator.calculate(
                state.data["sex"], state.data["age"], state.data["height"],
                state.data["weight"], state.data["activity"], state.data["goal"]
            )
            
            # 儲存完整資料
            self.firebase_service.save_user_profile(user_id, {
                "activity": state.data["activity"],
                "goal": state.data["goal"],
                "tdee": tdee
            })
            
            # 生成推薦
            if language == Language.CHINESE:
                msg1 = f"你的 TDEE 約為 {tdee} 大卡"
                prompt = f"請根據我每日需求 {tdee} 大卡，推薦一天三餐的食物組合，並註明每餐熱量，總和不超過 {tdee} 大卡。"
            else:
                msg1 = f"Your TDEE is approximately {tdee} calories"
                prompt = f"Based on my daily requirement of {tdee} calories, please recommend a daily three-meal food combination with calorie information for each meal, totaling no more than {tdee} calories."
            
            recommendation = self.response_generator.generate_calorie_recommendation(
                user_id, tdee, prompt, language
            )
            
            # 清除狀態
            self.user_states.pop(user_id, None)
            
            return f"{msg1}\n\n{recommendation}", True
        
        return None, False
    
    def is_in_tdee_flow(self, user_id: str) -> bool:
        """檢查用戶是否在 TDEE 流程中"""
        return user_id in self.user_states


# ==================== 主要的 Bot 處理器 ====================

class LineBotHandler:
    def __init__(self, config: Config):
        self.config = config
        self.firebase_service = FirebaseService(config.firebase_credentials_path)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.rag_system = RAGSystem()
        self.response_generator = ResponseGenerator(
            self.openai_client, self.firebase_service, self.rag_system
        )
        self.intent_detector = IntentDetector(self.openai_client, self.firebase_service)
        self.tdee_handler = TDEEHandler(
            self.openai_client, self.firebase_service, self.response_generator
        )
        
        # LINE Bot 設定
        self.line_bot_api = LineBotApi(config.line_access_token)
        self.handler = WebhookHandler(config.line_secret)
        self.setup_handlers()
    
    def setup_handlers(self):
        """設定 LINE Bot 事件處理器"""
        @self.handler.add(MessageEvent, message=TextMessage)
        def handle_text_message(event):
            self.handle_text(event)
    
    def handle_text(self, event):
        """處理文字訊息的主要邏輯"""
        user_id = event.source.user_id
        text = event.message.text.strip()
        
        logger.info(f"=== 處理訊息開始 ===")
        logger.info(f"User ID: {user_id}, 訊息: '{text}'")
        
        # 檢測語言
        language = LanguageDetector.detect(text)
        logger.info(f"檢測語言: {language.value}")
        
        # 記錄用戶訊息
        try:
            self.firebase_service.log_message(user_id, "user", text)
        except Exception as e:
            logger.error(f"記錄訊息失敗: {e}")
        
        try:
            # 檢查是否在 TDEE 流程中
            if self.tdee_handler.is_in_tdee_flow(user_id):
                logger.info("用戶在 TDEE 流程中")
                response, is_complete = self.tdee_handler.handle_tdee_input(user_id, text)
                
                if response:
                    self._send_response(event, response, user_id)
                    return
                else:
                    # TDEE 輸入無效，清除狀態並繼續正常處理
                    logger.info("TDEE 輸入無效，清除狀態")
            
            # 純 GPT 意圖檢測
            intent = self.intent_detector.detect(user_id, text)
            logger.info(f"🎯 GPT 檢測意圖: {intent.value}")
            
            # 根據意圖分發處理
            response = self._dispatch_intent_handler(intent, user_id, text, language)
            self._send_response(event, response, user_id)
            
        except Exception as e:
            logger.error(f"處理訊息時發生錯誤: {e}")
            
            # 生成錯誤回應
            if language == Language.CHINESE:
                error_response = "抱歉，處理您的請求時發生錯誤。請稍後再試。"
            else:
                error_response = "Sorry, an error occurred while processing your request. Please try again later."
            
            self._send_response(event, error_response, user_id)
        
        logger.info("=== 訊息處理完成 ===")
    
    def _dispatch_intent_handler(self, intent: Intent, user_id: str, text: str, language: Language) -> str:
        """根據意圖分發到對應的處理器"""
        logger.info(f"📤 分發意圖處理: {intent.value}")
        
        if intent == Intent.TDEE:
            response, _ = self.tdee_handler.handle_tdee_request(user_id, text, language)
            return response
            
        elif intent == Intent.CALORIE_QUERY:
            return self.response_generator.generate_calorie_query_response(
                user_id, text, language
            )
            
        elif intent == Intent.CALORIE_RECOMMENDATION:
            return self._handle_calorie_recommendation_gpt(user_id, text, language)
            
        elif intent == Intent.CHAT:
            return self.response_generator.generate_chat_response(
                user_id, text, language
            )
            
        elif intent == Intent.RECIPE:
            return self.response_generator.generate_recipe_response(
                user_id, text, language
            )
        
        elif intent == Intent.HELP:  # 新增這個分支
            return self.response_generator.generate_help_response(
                user_id, text, language
            )
            
        else:
            logger.warning(f"未知意圖: {intent}, 使用食譜處理邏輯")
            return self.response_generator.generate_recipe_response(
                user_id, text, language
        )
    
    def _handle_calorie_recommendation_gpt(self, user_id: str, text: str, language: Language) -> str:
        """使用 GPT 處理卡路里推薦"""
        logger.info(f"🔢 使用 GPT 處理卡路里推薦: {text}")
        
        # 使用 GPT 提取卡路里需求和偏好
        extraction_result = self._extract_calorie_preferences_gpt(user_id, text, language)
        
        # 生成推薦
        return self.response_generator.generate_calorie_recommendation_advanced(
            user_id, text, extraction_result, language
        )
    
    def _extract_calorie_preferences_gpt(self, user_id: str, text: str, language: Language) -> Dict[str, Any]:
        """使用 GPT 提取卡路里偏好和需求"""
        if language == Language.CHINESE:
            system_content = """
你是營養需求分析助理。請分析用戶的卡路里推薦請求，提取以下資訊。

請用 JSON 格式回答：
{
    "target_calories": 數字或null,
    "calorie_range": {"min": 數字, "max": 數字} 或 null,
    "meal_type": "早餐|午餐|晚餐|點心|全日三餐|null",
    "dietary_goal": "減重|增重|維持|健康|null", 
    "dietary_restrictions": ["素食", "低碳", "高蛋白"] 或 [],
    "preference_keywords": ["低卡", "健康", "簡單", "快手"]
}

範例：
「請給我2000大卡以內的食物推薦」→ {"calorie_range": {"min": 0, "max": 2000}, "preference_keywords": ["推薦"]}
「500大卡的早餐推薦」→ {"target_calories": 500, "meal_type": "早餐", "preference_keywords": ["推薦"]}
「減重期間的健康餐點」→ {"dietary_goal": "減重", "preference_keywords": ["健康"]}
"""
            user_content = f"分析這個請求：{text}"
        else:
            system_content = """
You are a nutrition needs analysis assistant. Please analyze the user's calorie recommendation request and extract the following information.

Respond in JSON format:
{
    "target_calories": number or null,
    "calorie_range": {"min": number, "max": number} or null,
    "meal_type": "breakfast|lunch|dinner|snack|daily_meals|null",
    "dietary_goal": "weight_loss|weight_gain|maintain|healthy|null",
    "dietary_restrictions": ["vegetarian", "low_carb", "high_protein"] or [],
    "preference_keywords": ["low_calorie", "healthy", "simple", "quick"]
}

Examples:
"Recommend meals under 2000 calories" → {"calorie_range": {"min": 0, "max": 2000}, "preference_keywords": ["recommend"]}
"500 calorie breakfast suggestions" → {"target_calories": 500, "meal_type": "breakfast", "preference_keywords": ["suggestions"]}
"Healthy meals for weight loss" → {"dietary_goal": "weight_loss", "preference_keywords": ["healthy"]}
"""
            user_content = f"Analyze this request: {text}"
        
        try:
            messages = self.intent_detector._compose_prompts(user_id, system_content, user_content)
            
            resp = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=300
            )
            
            response_text = resp.choices[0].message.content.strip()
            logger.info(f"GPT 偏好提取回應: {response_text}")
            
            # 解析 JSON
            try:
                preferences = json.loads(response_text)
                logger.info(f"✅ 成功解析用戶偏好: {preferences}")
                return preferences
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON 解析失敗: {e}")
                return {"raw_text": text}
                
        except Exception as e:
            logger.error(f"❌ GPT 偏好提取錯誤: {e}")
            return {"raw_text": text}
    
    def _send_response(self, event, response: str, user_id: str):
        """發送回應並記錄"""
        try:
            self.line_bot_api.reply_message(
                event.reply_token, 
                TextSendMessage(text=response)
            )
            self.firebase_service.log_message(user_id, "assistant", response)
            logger.info("回應發送成功")
        except Exception as e:
            logger.error(f"發送回應失敗: {e}")


# ==================== Flask 應用程式 ====================

"""創建 Flask 應用程式和 Bot 處理器"""
config = Config()
bot_handler = LineBotHandler(config)

app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    
    try:
        bot_handler.handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature.")
        abort(400)
    
    return "OK"

@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "healthy"}, 200