"""中文情感分析 API — FastAPI 后端
启动: uvicorn app:app --reload
文档: http://localhost:8000/docs
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import time

# ---------- 1. 加载模型 ----------
MODEL_PATH = "./sentiment-model/final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("✅ 模型加载完成")

# ---------- 2. FastAPI 应用 ----------
app = FastAPI(title="中文情感分析 API", version="1.0")

# ---------- 3. 请求/响应格式 ----------
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time_ms: float

# ---------- 4. 接口 ----------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    # 分词 + 推理
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = outputs.logits.argmax(dim=-1).item()
        conf = probs[0, pred].item()

    elapsed = (time.time() - start) * 1000

    return PredictResponse(
        text=req.text,
        sentiment="正面" if pred == 1 else "负面",
        confidence=round(conf, 4),
        processing_time_ms=round(elapsed, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
