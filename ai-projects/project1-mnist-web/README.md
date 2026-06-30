# ✍️ 手写数字识别 Web 应用

在浏览器画布上手写数字，实时识别 0-9。

## 技术栈

`PyTorch` · `Flask` · `HTML Canvas`

## 项目结构

```
backend/
├── train.py       # CNN 训练脚本
├── app.py         # Flask 推理服务
└── mnist_cnn.pth  # 训练好的模型
frontend/
└── index.html     # 前端画布页面
```

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchvision pillow flask numpy

# 2. 训练模型（可选，仓库已包含训练好的模型）
cd backend
python train.py

# 3. 启动后端
python app.py

# 4. 双击打开 frontend/index.html
```

## 效果

- 在黑色画布上写一个数字，点击"识别"
- 返回预测结果和置信度
- 测试准确率: 99.14%

## 实现要点

- CNN 模型：2 层卷积 + 2 层全连接
- 图片预处理：自动裁剪 + 居中 + 标准化，模拟 MNIST 官方预处理
- 前后端分离，HTTP 通信
