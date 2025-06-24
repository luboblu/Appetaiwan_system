# 🤖 營養諮詢 LINE Bot

一個智能的營養諮詢 LINE Bot，結合 GPT-4、RAG 技術和個人化 TDEE 計算，提供專業的營養建議、食譜查詢和餐點推薦服務。 

## ✨ 主要功能

### 🔥 TDEE 計算
- **個人化基礎代謝率計算**：根據性別、年齡、身高、體重計算
- **活動量評估**：支援靜態到劇烈運動五個等級
- **目標導向調整**：針對維持、減重、增重提供不同建議
- **數據持久化**：用戶資料儲存在 Firebase，無需重複輸入

### 🍳 智能食譜查詢
- **RAG 增強搜索**：結合向量搜索和 GPT-4 生成詳細食譜
- **豐富食譜庫**：涵蓋中式、西式、日式等多種料理
- **詳細製作指導**：包含材料清單、步驟說明和烹飪小技巧
- **營養資訊**：每道菜都標註熱量、蛋白質、脂肪等營養成分

### 📊 食物熱量查詢
- **精確營養分析**：查詢特定食物的卡路里和營養成分
- **智能食物識別**：GPT-4 自動提取用戶提及的食物名稱
- **比較建議**：提供同類食物的營養比較

### 🥗 個人化餐點推薦
- **智能偏好分析**：GPT-4 分析用戶需求和飲食偏好
- **多維度推薦**：考慮熱量、餐別、飲食目標、限制條件
- **營養均衡**：確保推薦餐點的營養搭配合理
- **客製化建議**：根據減重/增重/維持等目標調整推薦

### 💬 多語言對話支援
- **雙語智能**：支援繁體中文和英文
- **自動語言檢測**：根據用戶輸入自動切換語言
- **自然對話**：友善的聊天介面，支援多輪對話

### 🎯 智能意圖識別
- **GPT-4 意圖分類**：精確識別用戶需求（TDEE、食譜、熱量查詢等）
- **上下文理解**：結合對話歷史提供更準確的回應
- **多重意圖處理**：同時處理複雜的複合需求

## 🏗️ 技術架構

### 核心技術棧
- **Web 框架**：Flask
- **聊天機器人**：LINE Bot SDK
- **AI 模型**：OpenAI GPT-4.1-mini
- **資料庫**：Firebase Firestore
- **向量搜索**：FAISS + SentenceTransformers
- **文本處理**：LangChain

### 系統架構圖

![Software Architecture](docs\software_architecture.png)

**架構流程說明：**
1. **用戶互動層**：用戶透過 LINE App 發送訊息
2. **平台層**：LINE Platform 接收訊息並透過 Webhook 轉發
3. **雲端基礎設施**：
   - **Google Cloud Run**：無伺服器容器部署
   - **Flask + LINE Bot SDK**：主要應用框架
4. **AI 處理層**：
   - **Intent Detector**：使用 OpenAI SDK 進行意圖識別
   - **RAG Engine**：檢索增強生成，結合 Recipe data
   - **TDEE Module**：專業 TDEE 計算模組
5. **外部服務**：
   - **OpenAI Chat API**：GPT-4 智能對話
   - **Firebase**：用戶資料和對話歷史存儲

**關鍵技術特點：**
- 🔄 **異步處理**：Webhook 機制確保即時回應
- 🎯 **智能路由**：根據意圖自動分配到對應處理模組
- 📊 **數據持久化**：完整的用戶檔案和對話記錄
- 🚀 **雲原生**：完全基於 Google Cloud 的可擴展架構

### 模組設計

#### 🔧 核心模組
- **`LanguageDetector`**：多語言檢測
- **`IntentDetector`**：GPT-4 驅動的意圖識別
- **`TDEECalculator`**：專業 TDEE 計算引擎
- **`RAGSystem`**：向量檢索增強生成
- **`ResponseGenerator`**：智能回應生成器
- **`FirebaseService`**：數據持久化服務

#### 🎛️ 處理器
- **`TDEEHandler`**：多步驟 TDEE 收集流程
- **`LineBotHandler`**：主要訊息處理邏輯

## 🚀 安裝指南

### 前置需求
- Python 3.8+
- Firebase 專案
- LINE Developer 帳號
- OpenAI API 金鑰

### 1. 專案設置
```bash
git clone <repository-url>
cd nutrition-linebot
pip install -r requirements.txt
```

### 2. 依賴套件
```bash
pip install flask
pip install line-bot-sdk
pip install openai
pip install firebase-admin
pip install langchain
pip install sentence-transformers
pip install faiss-cpu  # 或 faiss-gpu（如有 GPU）
pip install numpy
pip install python-dotenv
```

### 3. 環境變數設置
創建 `.env` 檔案：
```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# LINE Bot
LINE_CHANNEL_ACCESS_TOKEN=your_line_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret

# Firebase
FIREBASE_CREDENTIALS_JSON=path/to/firebase-credentials.json
```

### 4. Firebase 設置
1. 在 [Firebase Console](https://console.firebase.google.com/) 創建新專案
2. 啟用 Firestore Database
3. 下載服務帳號金鑰 JSON 檔案
4. 將路徑設置到環境變數中

### 5. LINE Bot 設置
1. 在 [LINE Developers](https://developers.line.biz/) 創建 Channel
2. 取得 Channel Access Token 和 Channel Secret
3. 設置 Webhook URL：`https://your-domain.com/callback`

## ⚙️ 配置說明

### 食譜資料庫
- 檔案：`recipes.txt`
- 格式：結構化文本，包含食譜名稱、材料、步驟、營養資訊
- 支援自動分割和向量化索引

### TDEE 計算參數
```python
ACTIVITY_FACTORS = {
    "靜態": 1.20,    # 辦公室工作，很少運動
    "輕度": 1.375,   # 輕度運動，每週1-3天
    "中度": 1.55,    # 中度運動，每週3-5天
    "高度": 1.725,   # 高強度運動，每週6-7天
    "劇烈": 1.90     # 極高強度運動，體力工作
}
```

### RAG 系統配置
```python
# 文本分割參數
chunk_size = 300        # 每個文本塊大小
chunk_overlap = 50      # 重疊字數
top_k = 5              # 檢索結果數量
similarity_threshold = 0.4  # 相似度閾值
```

## 🎮 使用方法

### 基本對話
```
用戶：你好！
Bot：你好！我是專業營養助理，可以幫你計算TDEE、查詢食譜、推薦餐點。有什麼需要幫助的嗎？
```

### TDEE 計算流程
```
用戶：計算我的TDEE
Bot：我來幫你計算TDEE！首先，請告訴我你的性別：男 或 女
用戶：男
Bot：請告訴我年齡
用戶：25
Bot：請告訴我身高（公分）
用戶：175
Bot：請告訴我體重（公斤）
用戶：70
Bot：請選擇活動量：靜態、輕度、中度、高度、劇烈
用戶：中度
Bot：請選擇目標：維持、減重、增重
用戶：維持
Bot：你的 TDEE 約為 2380 大卡

🍽️ **餐點推薦**
[根據 TDEE 提供三餐建議...]
```

### 食譜查詢
```
用戶：蛋炒飯怎麼做？
Bot：🍳 **蛋炒飯食譜**

**材料：**
- 白飯 2碗（約400克）
- 雞蛋 2顆
- 蝦仁 100克
[詳細食譜內容...]
```

### 熱量查詢
```
用戶：蘋果有多少卡路里？
Bot：🍎 **蘋果營養資訊**
一顆中等大小的蘋果（約150g）含有約80大卡，富含膳食纤維和维生素C...
```

### 餐點推薦
```
用戶：推薦500大卡的早餐
Bot：🥗 **個人化餐點推薦**

**選項 1：** 蔬菜蛋餅 (~250大卡) + 豆漿 (~250大卡)
- 適合提供優質蛋白質和膳食纖維
[詳細推薦內容...]
```

## 📊 資料庫結構

### Firebase Firestore 集合

#### users 集合
```json
{
  "user_id": {
    "sex": "male/female",
    "age": 25,
    "height": 175.0,
    "weight": 70.0,
    "activity": "中度",
    "goal": "維持",
    "tdee": 2380,
    "updatedAt": "timestamp"
  }
}
```

#### users/{user_id}/messages 子集合
```json
{
  "message_id": {
    "role": "user/assistant",
    "content": "訊息內容",
    "timestamp": "timestamp"
  }
}
```

## 🔧 開發指南

### 新增意圖類型
1. 在 `Intent` 枚舉中新增意圖
2. 更新 `IntentDetector` 的系統提示
3. 在 `LineBotHandler._dispatch_intent_handler` 中新增處理邏輯

### 擴展食譜資料庫
1. 在 `recipes.txt` 中新增食譜（遵循現有格式）
2. 重啟服務讓 RAG 系統重新索引

### 自定義 TDEE 計算
修改 `TDEECalculator.ACTIVITY_FACTORS` 或計算公式

## 🚀 部署

### 本地開發
```bash
python tdee_gcp_refactored.py
```

### 生產部署（Google Cloud Platform）
```bash
# 部署到 Google Cloud Run
gcloud run deploy nutrition-bot \
  --source . \
  --platform managed \
  --region asia-east1 \
  --allow-unauthenticated
```

### 使用 Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "tdee_gcp_refactored:app"]
```

## 🧪 測試

### 健康檢查端點
```bash
curl https://your-domain.com/health
# 回應：{"status": "healthy"}
```

### LINE Bot Webhook 測試
使用 LINE 官方的 [Webhook Test Tool](https://developers.line.biz/console/)

## 🔍 監控與除錯

### 日誌級別
- `INFO`：一般操作記錄
- `WARNING`：非致命性問題
- `ERROR`：錯誤和異常

### 常見問題除錯
1. **RAG 系統初始化失敗**：檢查 `recipes.txt` 格式和路徑
2. **Firebase 連接錯誤**：驗證憑證檔案路徑和權限
3. **OpenAI API 錯誤**：檢查 API 金鑰和配額
4. **LINE Webhook 失敗**：確認 Channel Secret 和 Webhook URL

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

---

🌟 **Star 這個專案來支持我們！** ⭐