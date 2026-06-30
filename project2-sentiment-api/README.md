# 💬 中文情感分析 API

基于 bert-base-chinese 微调的中文情感分析服务，FastAPI + Docker 部署。

## 技术栈

`PyTorch` · `Transformers` · `FastAPI` · `Docker` · `BERT`

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（生成 sentiment-model/）
python train.py

# 3. 启动服务
uvicorn app:app --reload

# 4. 打开 API 文档
# http://localhost:8000/docs
```

## Docker 部署

```bash
# 构建镜像
docker build -t sentiment-api .

# 运行容器
docker run -p 8000:8000 sentiment-api
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 预测情感（正面/负面）|
| GET | `/health` | 健康检查 |

请求示例：
```json
{"text": "这家店服务真好"}
```
响应：
```json
{
  "text": "这家店服务真好",
  "sentiment": "正面",
  "confidence": 0.95,
  "processing_time_ms": 12.3
}
```

## 数据集

ChnSentiCorp 中文酒店评论数据集（正面/负面二分类）

## 模型效果（500条训练）

- Accuracy: 85.0%
- F1: 84.9%
