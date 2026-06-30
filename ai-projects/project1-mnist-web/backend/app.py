"""
MNIST 手写数字识别 — Flask 后端
运行时: python app.py
提供 POST /predict 接口
"""
import io
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------- 1. 模型结构（和 train.py 一模一样）----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ---------- 2. 加载训练好的权重 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
model.eval()
print(f"✅ 模型加载完成，设备: {DEVICE}")


# ---------- 3. 图片预处理 ----------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """把前端来的图片转成 MNIST 格式：居中裁剪 + 缩放"""
    image = image.convert("L")           # 转灰度

    # --- 自动裁剪到数字的边界框 ---
    arr = np.array(image)
    # 找到白色墨迹的区域（画布是黑底白字，墨迹 > 128）
    rows = np.any(arr > 128, axis=1)
    cols = np.any(arr > 128, axis=0)
    if rows.any() and cols.any():
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        # 稍微扩大一点边距
        pad = 4
        y1, y2 = max(0, y1-pad), min(280, y2+pad)
        x1, x2 = max(0, x1-pad), min(280, x2+pad)
        arr = arr[y1:y2+1, x1:x2+1]

    # --- 居中放到 20×20 的方块里 ---
    h, w = arr.shape
    size = max(h, w)
    canvas = np.full((size, size), 0, dtype=np.uint8)  # 黑底
    dy, dx = (size - h) // 2, (size - w) // 2
    canvas[dy:dy+h, dx:dx+w] = arr
    image = Image.fromarray(canvas)
    image = image.resize((20, 20), Image.Resampling.LANCZOS)

    # --- 放到 28×28 中央 ---
    final = np.full((28, 28), 0, dtype=np.uint8)
    final[4:24, 4:24] = np.array(image)
    image = Image.fromarray(final)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    tensor = transforms.ToTensor()(image)
    # 不需要 1-tensor，画布已经是黑底白字，和 MNIST 训练数据一致
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
    tensor = tensor.unsqueeze(0)
    return tensor


# ---------- 4. 接口 ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "请上传 image 文件"}), 400

    try:
        file = request.files["image"]
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        tensor = preprocess_image(img).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        return jsonify({"prediction": pred, "confidence": round(confidence, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
