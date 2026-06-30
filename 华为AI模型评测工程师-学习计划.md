# 华为 AI 模型评测工程师 — 暑期实习学习计划

> 目标岗位：华为 AI 模型工程师 - AI 模型评测岗位
> 当前背景：华中科技大学大二，已学习 Python + PyTorch
> 计划周期：约 14 周（3-4 个月）

---

## 一、差距分析

| 岗位要求 | 你的现状 | 差距 |
|---|---|---|
| 精通 Python | ✅ 已学 | 需从"会用"到"精通"（工程化能力） |
| 熟悉深度学习框架 | ✅ 已学 PyTorch | 需深入理解原理，不只是调包 |
| NLP / 多模态模型训推 | ❌ 未涉及 | **核心差距，需重点突破** |
| 模型评测体系设计 | ❌ 未涉及 | **岗位核心，需系统学习** |
| 自动化分析（聚类/打标） | ❌ 未涉及 | 需补充数据科学技能 |
| 顶会论文 / 竞赛经历 | ❌ 暂无 | 加分项，建议参与 |

---

## 二、学习路线图（3 阶段，约 14 周）

### 🔴 第一阶段：基础夯实（第 1-4 周）

**目标：从"会用 PyTorch"到"理解深度学习本质"**

#### 1.1 Python 工程化能力（持续）
- [ ] 掌握 `unittest` / `pytest` 编写单元测试
- [ ] 学习 `logging` 模块，结构化日志
- [ ] 熟练使用 `argparse` / `click` / `hydra` 做配置管理
- [ ] 学习 `git` 进阶：rebase、cherry-pick、submodule
- [ ] 了解 Docker 基础，能写 Dockerfile

#### 1.2 深度学习原理深化（第 1-2 周）
- [ ] 手写反向传播：用 NumPy 实现一个 MLP 的 forward + backward
- [ ] 深入理解 PyTorch 的 `autograd` 机制和计算图
- [ ] 学习 `torch.nn.Module` 的 `register_buffer`、`register_parameter` 等高级用法
- [ ] 掌握 `Dataset` / `DataLoader` 的 custom 实现，理解 `collate_fn`
- [ ] **必读书目**：《动手学深度学习》(d2l.ai) 重点章节

#### 1.3 Transformer 彻底搞懂（第 3-4 周）
- [ ] 从零实现一个 Transformer（参考 "The Annotated Transformer"）
- [ ] 逐行理解 Self-Attention、Multi-Head Attention 的计算过程
- [ ] 理解 Positional Encoding（Sinusoidal / RoPE / ALiBi）
- [ ] 掌握 Transformer 的推理优化：KV Cache、Flash Attention 原理
- [ ] **必看**：Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 从头实现

---

### 🟡 第二阶段：核心技能（第 5-9 周）

**目标：掌握 NLP + 多模态模型的训练、推理和评测**

#### 2.1 NLP 模型深入（第 5-6 周）
- [ ] 学习 HuggingFace `transformers` 库的完整工作流：
  - 模型加载、tokenizer 使用、`pipeline` 抽象
  - `Trainer` API 与自定义训练循环
  - 模型保存/加载/导出 ONNX
- [ ] 实践至少一项：微调 LLaMA / Qwen 等开源模型（LoRA / QLoRA）
  - 使用 `peft` + `bitsandbytes` 做量化微调
- [ ] 学习主流评测榜单和指标：
  - **通用能力**：MMLU、C-Eval、CMMLU
  - **推理能力**：GSM8K、MATH
  - **代码能力**：HumanEval、MBPP
  - **中文能力**：SuperCLUE、FlagEval

#### 2.2 多模态模型（第 7-8 周）
- [ ] 理解 CLIP 的对比学习原理和训练流程
- [ ] 学习 BLIP-2 / LLaVA 等多模态对话模型的架构
- [ ] 了解扩散模型（Stable Diffusion）基本原理
- [ ] 多模态评测指标学习：
  - 图像描述：BLEU、CIDEr、SPICE
  - VQA：准确率、ANLS
  - 多模态排序：MMBench、MME、SEED-Bench

#### 2.3 模型评测体系设计 —— **岗位核心能力**（第 8-9 周）
- [ ] 学习评测体系设计的完整方法论：
  - **评测集设计**：覆盖率、难度分布、数据污染检测
  - **评测维度**：正确性、鲁棒性、安全性、偏见、幻觉
  - **打分策略**：规则打分 vs 模型打分（LLM-as-Judge） vs 人工打分
- [ ] 研究业界评测框架源码：
  - [OpenCompass](https://github.com/open-compass/opencompass) —— 最推荐，国产主流
  - [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) —— 国际通用
  - [OpenAI Evals](https://github.com/openai/evals)
- [ ] **动手项目**：用 OpenCompass 完整跑一轮评测，理解整个 pipeline
  - 选 3 个开源模型 + 5 个数据集
  - 理解 prompt 模板、推理配置、指标计算、结果汇总

---

### 🟢 第三阶段：工程实践 & 简历强化（第 10-14 周）

**目标：做出能写在简历上的项目，准备面试**

#### 3.1 评测工具开发实践（第 10-11 周）
- [ ] **项目 1：搭建一个模型评测沙箱环境**
  - 用 Docker 封装评测环境，支持 GPU
  - 实现模型服务与评测逻辑的解耦（API 化）
  - 支持批量评测 + 结果自动收集
- [ ] **项目 2：实现 LLM-as-Judge 自动打分系统**
  - 设计 Judge Prompt 模板
  - 对比 GPT-4 / Claude 作为裁判的一致性
  - 分析位置偏差（position bias）并设计消偏策略

#### 3.2 数据分析 & 自动聚类（第 11-12 周）
- [ ] 学习文本聚类算法：
  - BERTopic：用 Sentence Embedding + UMAP + HDBSCAN
  - K-Means + TF-IDF 作为 baseline
- [ ] 实践自动打标（Auto-Labeling）：
  - 用大模型对评测 Badcase 做自动分类
  - 设计分类体系（如：事实错误、逻辑错误、格式错误…）
- [ ] 学习可视化：`matplotlib`、`seaborn`、`plotly`

#### 3.3 竞赛 / 开源贡献（第 12-14 周）
- [ ] **优先级最高**：参加一次相关竞赛
  - Kaggle：LLM Science Exam、LMSYS Chatbot Arena Human Preference
  - 天池 / 讯飞：大模型相关赛道
  - 华为云相关竞赛（与目标公司直接相关！）
- [ ] 给 OpenCompass 或 lm-evaluation-harness 提 2-3 个 PR
  - 可以是新增评测集、修 bug、改进文档
  - 这是面试时极强的加分点
- [ ] 写 2-3 篇技术博客，记录你的学习实践过程

#### 3.4 简历与面试准备（贯穿）
- [ ] 准备一个 **个人项目展示页**（GitHub README + 技术文档）
- [ ] 准备 3 个能深入讲的项目故事（STAR 法则）：
  1. 从零实现 Transformer 的过程中遇到了什么困难？
  2. 搭建评测体系时如何确保评测的公平性和可复现性？
  3. LLM-as-Judge 的消偏设计你做了什么尝试？
- [ ] 刷 LeetCode：主攻中等难度（数组、字符串、哈希、树、DP）
  - 华为笔试通常 2 道中等 + 1 道困难，每天保持 1-2 题
- [ ] 准备开放性问题：
  - "你如何设计一个评测集来评估大模型的安全能力？"
  - "评测集的数据污染问题如何检测和解决？"
  - "LLM-as-Judge 的可靠性如何验证？"

---

## 三、推荐资源清单

### 书籍

| 书名 | 用途 |
|---|---|
| 《动手学深度学习》(d2l.ai) | PyTorch + 深度学习基础 |
| 《大规模语言模型：从理论到实践》(张奇等) | LLM 入门必读 |
| 《Build a Large Language Model (From Scratch)》(Sebastian Raschka) | 从零理解 LLM |
| 《设计机器学习系统》(Chip Huyen) | 工程思维 |

### 论文（由浅入深）

1. **Attention Is All You Need** — Transformer 原始论文
2. **BERT: Pre-training of Deep Bidirectional Transformers**
3. **LoRA: Low-Rank Adaptation of Large Language Models**
4. **Training Language Models to Follow Instructions with Human Feedback (InstructGPT)**
5. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**
6. **OpenCompass: A Universal Evaluation Platform**

### 代码仓库

- [nanoGPT](https://github.com/karpathy/nanoGPT) — 最干净的 GPT 实现
- [OpenCompass](https://github.com/open-compass/opencompass) — 评测框架
- [litgpt](https://github.com/Lightning-AI/litgpt) — 轻量 LLM 训练/微调
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 中文生态最佳微调框架

---

## 四、每周时间分配建议

```
周一～周五（每天 2-3h）：
  - 1h：理论/论文阅读
  - 1-2h：代码实践
  - 通勤/碎片时间：刷 LeetCode 1 题

周末（每天 5-6h）：
  - 3-4h：项目开发
  - 1-2h：写博客/整理笔记
  - 1h：回顾 + 规划下周
```

---

## 五、可以写在简历上的项目方向

1. **多维度大模型评测框架**：基于 OpenCompass + 自研 LLM-as-Judge 的评测系统
2. **评测 Badcase 自动聚类与归因分析工具**：BERTopic + LLM 自动打标
3. **多模态模型幻觉检测与评测基准**：针对图生文场景的专项评测
4. **Docker 化的沙箱评测环境**：支持一键部署的评测基础设施

---

## 六、关键时间节点

| 时间 | 事项 |
|---|---|
| 第 4 周末 | 完成 Transformer 手写实现，发布到 GitHub |
| 第 8 周末 | 用 OpenCompass 完成一轮完整评测，输出分析报告 |
| 第 10 周末 | 完成 LLM-as-Judge 项目 |
| 第 12 周末 | 完成自动聚类分析项目 |
| 第 14 周末 | 整合所有项目，完善简历，开始投递 |

---

## 七、核心建议

> 华为这个岗位本质上是 **"懂模型的评测工程师"**，不是纯算法岗。面试时最能打动面试官的是——你对评测方法论的系统理解 + 你实际动手做过的评测项目。千万不要停留在"我会调用 API"的层面，要深入到"我理解评测集为什么这样设计、打分标准如何保证一致性、评测结果如何分析归因"这一层。
