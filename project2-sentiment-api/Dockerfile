# 中文情感分析 API — Docker 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 装依赖（先 copy requirements，利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型和应用代码
COPY sentiment-model/ ./sentiment-model/
COPY app.py .

# 启动 FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
