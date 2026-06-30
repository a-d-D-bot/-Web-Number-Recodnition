"""
MNIST 手写数字识别 — CNN 训练脚本
运行: python train.py
输出: mnist_cnn.pth（训练好的模型权重）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 1. 超参数 ----------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. 加载 MNIST 数据 ----------
transform = transforms.Compose([
    transforms.ToTensor(),           # 把图片转成 tensor，像素值 0~1
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_dataset)} 张, 测试集: {len(test_dataset)} 张")
print(f"使用设备: {DEVICE}")

# ---------- 3. 定义 CNN 网络 ----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：提取图像特征
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1×28×28 → 32×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # 32×14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 64×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # 64×7×7
        )
        # 全连接层：分类
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),                 # 防止过拟合
            nn.Linear(128, 10),              # 输出 10 个类别的分数
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ---------- 4. 训练 ----------
model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n开始训练...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {train_acc:.4f}")

# ---------- 5. 测试 ----------
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()

test_acc = correct / len(test_dataset)
print(f"\n✅ 测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ---------- 6. 保存模型 ----------
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存为 mnist_cnn.pth")
