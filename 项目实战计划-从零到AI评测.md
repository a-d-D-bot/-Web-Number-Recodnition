# 项目实战计划 — 从零到 AI 评测

> 背景：无项目经验 → 目标：能独立做出 AI 评测相关项目
> 策略：4 个项目，由浅入深，每个都能写到简历上

---

## 项目路线图

```
项目1                项目2                项目3                项目4
手写数字识别         情感分析 API         微调中文大模型        大模型评测框架
(2-3天)             (5-7天)             (7-10天)            (10-14天)
    ↓                   ↓                   ↓                   ↓
 入门项目            前后端打通            模型训推              岗位核心
```

---

## 🔰 项目1：手写数字识别 Web 应用（2-3 天）

**目的**：跑通"训练模型 → 部署 → 前端展示"的完整流程，建立信心

### 你会学到什么
- PyTorch 训练一个真实模型（CNN）
- 保存/加载模型权重
- 用 Flask 写一个最简单的后端
- HTML 画布 + JavaScript 让用户在网页上手写数字

### 步骤

```
第1天
├── 1. 用 PyTorch 训练 MNIST CNN（30分钟，网上代码很多）
│     - 网络结构：Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → FC
│     - 训练 5 个 epoch，准确率应该 > 98%
│     - 保存模型为 mnist_cnn.pth
│
├── 2. 本地测试模型（30分钟）
│     - 加载模型，随便喂一张测试图，确认能预测正确
│
└── 3. 搭 Flask 后端（1小时）
      - POST /predict 接口，接收图片，返回预测数字
      - 用 PIL 处理图片（缩放 28x28、灰度化）
```

```
第2天
├── 4. 写前端页面（2小时）
│     - 一个 280x280 的画布（Canvas），鼠标可以写字
│     - 一个"识别"按钮，点击后把画布内容发给后端
│     - 显示预测结果
│
└── 5. 联调 & 美化（1小时）
      - 确保前端→后端→模型整条链路跑通
      - 加一点简单的 CSS，让它看起来像样
```

### 最终效果
浏览器打开 → 鼠标写一个"3" → 点识别 → 显示"预测结果：3（置信度 98.2%）"

### 技术栈
`PyTorch` `Flask` `HTML Canvas` `PIL`

### 简历怎么写
> 独立开发手写数字识别 Web 应用，使用 PyTorch 训练 CNN 模型（准确率 98%+），基于 Flask 搭建 REST API，实现浏览器端实时手写识别。

---

## 🟡 项目2：中文情感分析 API 服务（5-7 天）

**目的**：学会用预训练模型做 NLP 任务，写出规范的 API，体验"模型即服务"

### 你会学到什么
- 用 HuggingFace Transformers 加载中文预训练模型
- 模型微调（Bert-base-chinese 做情感分类）
- FastAPI 写规范的 RESTful API（比 Flask 更现代）
- 用 Pydantic 做请求/响应模型校验
- 自动生成 Swagger 接口文档

### 步骤

```
第1-2天：找数据 + 训练模型
├── 1. 找中文情感分析数据集
│     - 推荐：ChnSentiCorp（酒店评论）、weibo_senti_100k（微博评论）
│     - 标注：正面 / 负面（可以先做二分类）
│
├── 2. 微调 Bert-base-chinese
│     - 使用 transformers Trainer API
│     - 训练 2-3 个 epoch，F1 应该 > 90%
│     - 保存模型到本地
│
└── 3. 测试模型，确认能正常推理
```

```
第3-4天：搭 API 服务
├── 4. FastAPI 项目结构
│     api/
│     ├── main.py          # FastAPI 入口
│     ├── model.py          # 模型加载 & 推理
│     ├── schemas.py        # Pydantic 请求/响应定义
│     └── config.py         # 配置管理
│
├── 5. 实现接口
│     - POST /predict      单条预测（输入文本，返回情感+置信度）
│     - POST /batch_predict 批量预测
│     - GET  /health       健康检查
│
└── 6. 添加中间件
      - 请求日志记录
      - 异常处理（模型加载失败、输入过长等）
```

```
第5-6天：容器化 + 测试
├── 7. 写 Dockerfile
│     - 基于 python:3.10-slim
│     - 安装依赖，复制模型和代码
│     - docker build & docker run 验证
│
├── 8. 编写测试
│     - 用 pytest + httpx 测试 API 接口
│     - 测试正常输入、空输入、超长输入
│
└── 9. 写 README（后面面试要用）
      - 项目介绍、技术架构图（ASCII 即可）
      - 如何运行、API 文档截图
```

### 最终效果
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "这家酒店的服务态度非常好，房间也很干净"}'

# 返回：
{
  "text": "这家酒店的服务态度非常好，房间也很干净",
  "sentiment": "positive",
  "confidence": 0.967,
  "processing_time_ms": 12.3
}
```
浏览器打开 `http://localhost:8000/docs` 有自动生成的 Swagger 文档。

### 技术栈
`PyTorch` `HuggingFace Transformers` `FastAPI` `Pydantic` `Docker` `pytest`

### 简历怎么写
> 基于 Bert-base-chinese 微调中文情感分析模型（F1 90%+），使用 FastAPI 搭建生产级推理 API，支持单条/批量预测，集成 Swagger 自动文档、Docker 容器化部署和 pytest 自动化测试。

---

## 🟠 项目3：微调中文大模型对话能力（7-10 天）

**目的**：真正接触 LLM，体验 SFT（监督微调），这是面试聊天的重点话题

### 你会学到什么
- LoRA/QLoRA 微调原理和实践
- 使用 LLaMA-Factory 快速上手（不用自己写训练代码）
- 构造指令数据集（Instruction Data）
- 理解 SFT 的效果和局限
- 用 vLLM 部署微调后的模型

### 为什么选 LLaMA-Factory
- 国产开源框架文档友好，社区活跃
- 一行命令启动微调、推理、评估
- 面试官知道这个框架，聊得起来

### 步骤

```
第1-2天：环境搭建 + 选模型 + 理解流程
├── 1. 安装 LLaMA-Factory
│     git clone https://github.com/hiyouga/LLaMA-Factory.git
│     pip install -e ".[torch,metrics]"
│
├── 2. 选一个 0.5B-1.5B 的小模型（训练快、省显卡）
│     - 推荐：Qwen2.5-0.5B / Qwen2.5-1.5B
│     - 你的 GPU 只要能跑 4bit 量化就行（6GB 显存足够 0.5B）
│     - 没有 GPU？用 Google Colab T4（免费）
│
└── 3. 边读边理解 LLaMA-Factory 的配置文件
      - dataset_info.json 是怎么定义数据集的
      - LoRA 的 r / alpha / dropout 是什么意思
```

```
第3-5天：构造数据集 + 微调
├── 4. 自己构造 200-500 条中文指令数据
│     - 主题：选一个你熟悉的领域（比如计算机基础知识问答）
│     - 格式：{ "instruction": "...", "input": "", "output": "..." }
│     - 一半自己写 + 一半用 GPT/Claude 生成（标注来源）
│     - 关键经验：数据质量 >> 数据数量
│
├── 5. 用 QLoRA 微调
│     - 4bit 量化加载模型
│     - LoRA 配置：rank=8, alpha=16（小模型用小的就行）
│     - 训练 1-2 个 epoch（会很快，几十分钟到几小时）
│     - 观察 loss 曲线，确保下降
│
└── 6. 对比微调前后的效果
      - 准备 20 个相同的 prompt
      - 分别用原始模型和微调模型生成
      - 对比输出质量差异，写个简单的对比表格
```

```
第6-8天：部署 + API
├── 7. 用 vLLM 部署微调后的模型（或直接用 LLaMA-Factory 的推理）
│     - 提供 OpenAI 兼容 API（/v1/chat/completions）
│     - 这样你的 API 就可以被任何 ChatGPT 客户端调用了
│
├── 8. 包装成 Docker 服务
│     - 类似项目2 的 FastAPI 结构
│     - 加上流式输出（SSE）
│
└── 9. 写对比分析报告（这是面试时的核心素材）
      - 微调前后效果对比（量化 + 定性）
      - 数据集设计思路
      - 遇到的问题和解决方案
      - Training loss 曲线
```

### 最终效果
一个微调后的中文问答模型，部署为 API，能回答你那个领域的问题，明显比原始模型更准确。

### 可能遇到的问题（提前知道更好）
- **显存不够**：降 batch_size = 1，用 4bit 量化，用 gradient_checkpointing
- **过拟合**：loss 很低但生成质量差 → 减少 epoch、加正则化
- **生成内容重复**：调整 temperature、top_p

### 技术栈
`LLaMA-Factory` `Qwen2.5` `LoRA/QLoRA` `bitsandbytes` `vLLM` `Docker`

### 简历怎么写
> 基于 Qwen2.5-1.5B 构建中文指令微调数据集（500条），使用 LLaMA-Factory + QLoRA 完成 SFT 微调。微调后模型在目标领域准确率提升 30%+，使用 vLLM 部署并提供 OpenAI 兼容 API。

---

## 🔴 项目4：大模型评测框架实战（10-14 天）

**目的**：岗位核心能力项目，直接对标华为 JD

### 你会学到什么
- 评测框架的设计思路和架构
- 如何设计评测集、评分标准
- 用 LLM 当裁判做自动打分（LLM-as-Judge）
- 评测结果的可视化和对比分析
- 这个项目是面试时最有分量的

### 整体思路
不要从零写轮子，**基于 OpenCompass 二次开发**——看懂它的源码结构，然后加自己的评测集和能力维度。

### 步骤

```
第1-3天：深入 OpenCompass
├── 1. 安装 + 跑通官方示例
│     pip install opencompass
│     用官方配置跑一个最简单的评测（如 ceval）
│
├── 2. 读核心源码，理解架构
│     - Config: 评测配置是怎么组织的
│     - Dataset: 数据集怎么加载、预处理
│     - Model: 模型是怎么封装的
│     - Evaluator: 评分是怎么计算的
│     - Partition: 推理和评分怎么并行
│
└── 3. 画一张架构图（帮助理解）
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │  Config  │ →  │Inferencer│ →  │Evaluator │ → 结果
      └──────────┘    └──────────┘    └──────────┘
           │               │                │
           ▼               ▼                ▼
       数据集配置       模型推理         指标计算
```

```
第4-6天：自建评测集
├── 4. 设计一个评测集（100-200 条，领域你定）
│     - 推荐：中文编程基础问答 / 计算机通识 / 数学推理
│     - 每道题设计好：问题、参考答案、评分标准
│     - 题型混合：单选、简答、代码填空
│
├── 5. 转换成 OpenCompass 格式
│     - 写一个 dataset 类
│     - 注册到 OpenCompass 的 dataset registry
│
└── 6. 评测 3 个开源模型
      - 如：Qwen2.5-7B、ChatGLM3-6B、Llama3-8B
      - 跑通完整流程，得到评测结果
```

```
第7-9天：实现 LLM-as-Judge 打分
├── 7. 设计 Judge Prompt（最关键的一步）
│     - 可以参考 MT-Bench 的 prompt 设计
│     - 包含：角色设定、评分维度、打分标准（1-5分）、输出格式要求
│     - 示例：
│       "你是一个专业的大模型评测裁判。请根据以下标准对模型回答打分：
│        1. 准确性（40%）：回答内容是否正确
│        2. 完整性（30%）：是否完整回答了问题
│        3. 流畅性（20%）：语言是否通顺自然
│        4. 安全性（10%）：是否存在有害或偏见内容
│        请先给出分析，最后给出 JSON 格式的分数。"
│
├── 8. 实现自动打分模块
│     - 调用 GPT-4 / Claude API 作为裁判
│     - 处理裁判输出格式不稳定的情况（retry 机制）
│     - 多个裁判打分取平均（增加可靠性）
│
└── 9. 验证裁判可靠性
      - 挑 30 条数据，人工打分
      - 对比裁判分数和人工分数的一致性（Spearman 相关系数）
      - 这是面试时能讲的很有深度的点
```

```
第10-12天：结果分析与可视化
├── 10. 做一个对比分析 Dashboard
│      - 用 plotly 画雷达图（多维度能力对比）
│      - 用柱状图展示各模型在不同评测集上的得分
│      - 用表格展示 Badcase 列表
│
├── 11. 分析 Badcase
│      - 用 Sentence-BERT 对错误回答做向量化
│      - 用 UMAP 降维到 2D，散点图展示聚类
│      - 观察：哪些类型的错误扎堆出现？
│
├── 12. 写一份评测分析报告
│      - 各模型优劣势分析
│      - 典型 Badcase 分析
│      - 评测集质量问题反思
│      - 后续优化建议
│
└── 13. 整理成项目文档
      - README: 项目背景、架构、使用方法
      - 报告: PDF/Notion 格式的评测报告
      - 代码: GitHub 仓库
```

### 最终效果
一个完整的评测系统：
- 自己的评测集（100-200题）
- 3 个模型的对比评测结果
- LLM-as-Judge 自动打分模块
- 可视化分析 Dashboard
- 一份完整的评测分析报告

### 技术栈
`OpenCompass` `GPT-4 API` `FastAPI` `plotly` `Sentence-BERT` `UMAP` `Docker`

### 简历怎么写（重点）
> 基于 OpenCompass 二次开发大模型评测框架，自建中文编程评测集（200题），实现对 Qwen2.5、ChatGLM3、Llama3 等模型的自动化评测；设计 LLM-as-Judge 多维度打分系统（准确性/完整性/流畅性/安全性），通过多裁判投票机制提升评分可靠性（Spearman r=0.85）；使用 Sentence-BERT + UMAP 对 Badcase 进行聚类分析，输出详细评测报告。

---

## 五个项目总结

| 项目 | 难度 | 时间 | 核心价值 |
|---|---|---|---|
| 手写数字识别 | ⭐ | 2-3天 | 跑通全流程，建立信心 |
| 情感分析 API | ⭐⭐ | 5-7天 | 工程化能力，API 设计 |
| 微调大模型 | ⭐⭐⭐ | 7-10天 | LLM 训推经验，面试重点 |
| 评测框架 | ⭐⭐⭐⭐ | 10-14天 | **岗位核心**，简历亮点 |

---

## 开始前的准备工作

### 1. 现在立刻做
- [ ] 注册 GitHub 账号，新建一个仓库叫 `ai-projects`
- [ ] 下载 VS Code / PyCharm，配好 Python 环境
- [ ] 安装 Docker Desktop（Windows）

### 2. 每个项目做完后
- [ ] 代码推到 GitHub（commit 写得认真点，中文英文都行）
- [ ] 写 README.md（至少包含：项目介绍、技术栈、如何运行、效果截图）
- [ ] 写一篇博客笔记（CSDN / 掘金 / 知乎 / 个人博客都行）

### 3. 遇到问题怎么办
1. 先看官方文档
2. Google / StackOverflow
3. GitHub Issues 区搜索
4. ChatGPT / Claude 直接问

### 4. 显卡问题
- 项目1、2 不需要 GPU，笔记本 CPU 就能跑
- 项目3 微调：用 Google Colab 免费 T4（每周有免费额度）
- 项目4 评测：用 API 调模型，不需要本地 GPU
- **没有显卡完全可以做完所有项目**

---

## 时间线

```
Week 1 ─ 项目1（手写数字识别）
Week 2 ─ 项目2（情感分析 API）
Week 3-4 ─ 项目3（微调大模型）
Week 5-6 ─ 项目4（评测框架）
```

> 不一定连续 6 周，根据学校课程灵活调整。关键是每个项目做到"能完整展示"的状态，不要留半拉子工程。
