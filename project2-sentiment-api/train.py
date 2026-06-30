"""中文情感分析 — BERT 微调脚本"""
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ---------- 1. 加载数据集 ----------
print("正在加载 ChnSentiCorp 数据集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")
print(dataset)
print(f"\n训练集: {len(dataset['train'])} 条")
print(f"验证集: {len(dataset['validation'])} 条")
print(f"测试集: {len(dataset['test'])} 条")
print(f"\n前3条样本:")
for i in range(3):
    d = dataset["train"][i]
    print(f"  [{i}] label={d['label']} | {d['text'][:60]}...")

# ---------- 2. Tokenizer ----------
print("\n加载 tokenizer...")
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)


print("分词处理...")
tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized.set_format("torch")
# 为了加快 CPU 训练，先各取少量数据跑通流程
tokenized["train"] = tokenized["train"].select(range(500))
tokenized["validation"] = tokenized["validation"].select(range(100))
tokenized["test"] = tokenized["test"].select(range(100))
print(f"训练集: {len(tokenized['train'])} | 验证集: {len(tokenized['validation'])} | 测试集: {len(tokenized['test'])}")
print(f"✅ tokenizer 就绪")

# ---------- 3. 加载模型 ----------
print("加载模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2  # 二分类：正面/负面
)

# ---------- 4. 评价指标 ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# ---------- 5. 训练配置 ----------
training_args = TrainingArguments(
    output_dir="./sentiment-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# ---------- 6. 训练 ----------
print("\n开始训练...\n")
trainer.train()

# ---------- 7. 测试集评估 ----------
print("\n测试集评估:")
metrics = trainer.evaluate(tokenized["test"])
print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"  F1: {metrics['eval_f1']:.4f}")

# ---------- 8. 保存模型 ----------
model.save_pretrained("./sentiment-model/final")
tokenizer.save_pretrained("./sentiment-model/final")
print("\n✅ 模型已保存到 ./sentiment-model/final/")
