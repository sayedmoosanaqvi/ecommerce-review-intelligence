import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


cells = [
    md_cell("""
# Cross-Platform E-Commerce Review Intelligence
## Tier 2 (Custom PyTorch BiLSTM + Attention From Scratch) & Tier 3 (Fine-Tuned XLM-RoBERTa)

**Before running:**
1. Go to **Runtime -> Change runtime type -> T4 GPU** -> Save.
2. Upload your 3 leakage-free split files (`amazon_train.csv`, `amazon_val.csv`, `amazon_test.csv`) when prompted in Cell 2.
    """),

    code_cell("""
# Cell 1: Verify GPU & install packages
!pip install -q transformers sentencepiece scikit-learn seaborn matplotlib pandas

import os, re, json, math, time, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU detected! Switch Runtime -> Change runtime type -> T4 GPU.")

OUT_DIR = Path("colab_outputs")
OUT_DIR.mkdir(exist_ok=True)
    """),

    code_cell("""
# Cell 2: Upload amazon_train.csv, amazon_val.csv, amazon_test.csv from your PC
from google.colab import files

required = ["amazon_train.csv", "amazon_val.csv", "amazon_test.csv"]
missing = [f for f in required if not Path(f).exists()]
if missing:
    print("Please select and upload:", missing)
    files.upload()

df_train = pd.read_csv("amazon_train.csv")
df_val = pd.read_csv("amazon_val.csv")
df_test = pd.read_csv("amazon_test.csv")

# Hard leakage check
assert len(set(df_train["parent_asin"]) & set(df_val["parent_asin"])) == 0
assert len(set(df_train["parent_asin"]) & set(df_test["parent_asin"])) == 0
assert len(set(df_val["parent_asin"]) & set(df_test["parent_asin"])) == 0

LABELS = ["Negative", "Mixed_Neutral", "Positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}

y_train = df_train["sentiment_3class"].map(LABEL2ID).values
y_val = df_val["sentiment_3class"].map(LABEL2ID).values
y_test = df_test["sentiment_3class"].map(LABEL2ID).values

class_weights_np = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
CLASS_WEIGHTS = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

print(f"Splits verified (ZERO product overlap): Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")
print("Balanced class weights (Neg, Mix, Pos):", [round(float(w), 3) for w in class_weights_np])
    """),

    code_cell("""
# Cell 3: Shared evaluation & plotting utilities
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Macro_Precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Macro_Recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Macro_F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Weighted_F1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }

def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
    axes[0].set_title(f"{title}\\nRaw Counts (Test n={len(y_true):,})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
    axes[1].set_title(f"{title}\\nRow-Normalized Recall")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.show()

def plot_curves(history, title, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title(f"{title} - Loss Curve"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], marker="o", label="Train Macro-F1")
    axes[1].plot(epochs, history["val_f1"], marker="s", label="Val Macro-F1")
    axes[1].set_title(f"{title} - Macro-F1 Curve"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.show()
    """),

    md_cell("""
---
## Tier 2: Custom PyTorch BiLSTM + Masked Self-Attention (100% Trained From Scratch)
* Zero pretrained weights, zero APIs.
* Vocabulary built strictly on `amazon_train.csv` to prevent data leakage.
* Padding tokens are explicitly masked out inside the attention mechanism.
    """),

    code_cell("""
# Cell 4: Tokenizer, Vocabulary (fit ONLY on Train), and PyTorch Dataset for Tier 2
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.]")

def simple_tokenize(text):
    return TOKEN_RE.findall(str(text).lower())

MAX_VOCAB = 25000
MAX_LEN_LSTM = 160

counter = Counter()
for txt in df_train["full_text"].astype(str):
    counter.update(simple_tokenize(txt))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, freq in counter.most_common(MAX_VOCAB - 2):
    if freq >= 2:
        vocab[word] = len(vocab)

print(f"Custom Scratch Vocabulary Size (fit on train only): {len(vocab):,}")

def encode_text(text, max_len=MAX_LEN_LSTM):
    tokens = simple_tokenize(text)[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) == 0:
        ids = [1]
    pad_len = max_len - len(ids)
    return ids + [0] * pad_len

class ScratchReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encoded = [encode_text(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

train_ds_lstm = ScratchReviewDataset(df_train["full_text"].astype(str).tolist(), y_train)
val_ds_lstm   = ScratchReviewDataset(df_val["full_text"].astype(str).tolist(), y_val)
test_ds_lstm  = ScratchReviewDataset(df_test["full_text"].astype(str).tolist(), y_test)

train_loader_lstm = DataLoader(train_ds_lstm, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader_lstm   = DataLoader(val_ds_lstm, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader_lstm  = DataLoader(test_ds_lstm, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    """),

    code_cell("""
# Cell 5: Define Custom BiLSTM + Masked Additive Attention Architecture & Train From Scratch
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        mask = (input_ids != 0)  # True for real tokens, False for <PAD>
        x = self.emb_drop(self.embedding(input_ids))
        h, _ = self.lstm(x)      # [B, T, 2H]
        scores = self.attn(h).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = torch.sum(h * weights, dim=1)  # [B, 2H]
        return self.head(self.norm(context))

model_lstm = BiLSTMAttentionClassifier(vocab_size=len(vocab)).to(DEVICE)
total_params = sum(p.numel() for p in model_lstm.parameters() if p.requires_grad)
print(f"Tier-2 BiLSTM+Attention trainable parameters (random init): {total_params:,}")

criterion_lstm = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
optimizer_lstm = torch.optim.AdamW(model_lstm.parameters(), lr=1.5e-3, weight_decay=1e-2)
scheduler_lstm = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode="max", factor=0.5, patience=1)

EPOCHS_LSTM = 10
PATIENCE = 3
best_val_f1_lstm = -1.0
wait = 0
history_lstm = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

def eval_lstm(loader):
    model_lstm.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model_lstm(x)
            loss = criterion_lstm(logits, y)
            total_loss += loss.item() * len(y)
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    avg_loss = total_loss / len(trues)
    f1 = f1_score(trues, preds, average="macro")
    return avg_loss, f1, np.array(trues), np.array(preds)

t0 = time.perf_counter()
for epoch in range(1, EPOCHS_LSTM + 1):
    model_lstm.train()
    tr_loss, tr_preds, tr_trues = 0.0, [], []
    for x, y in train_loader_lstm:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer_lstm.zero_grad()
        logits = model_lstm(x)
        loss = criterion_lstm(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
        optimizer_lstm.step()

        tr_loss += loss.item() * len(y)
        tr_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        tr_trues.extend(y.cpu().numpy())

    tr_loss /= len(tr_trues)
    tr_f1 = f1_score(tr_trues, tr_preds, average="macro")
    val_loss, val_f1, _, _ = eval_lstm(val_loader_lstm)
    scheduler_lstm.step(val_f1)

    history_lstm["train_loss"].append(tr_loss)
    history_lstm["val_loss"].append(val_loss)
    history_lstm["train_f1"].append(tr_f1)
    history_lstm["val_f1"].append(val_f1)

    print(f"Epoch {epoch:02d}/{EPOCHS_LSTM} | Train Loss: {tr_loss:.4f} F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

    if val_f1 > best_val_f1_lstm:
        best_val_f1_lstm = val_f1
        wait = 0
        torch.save({"model_state": model_lstm.state_dict(), "vocab": vocab, "labels": LABELS}, OUT_DIR / "tier2_bilstm_scratch.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

lstm_train_sec = round(time.perf_counter() - t0, 1)

# Load best checkpoint & evaluate strictly once on held-out Test
ckpt = torch.load(OUT_DIR / "tier2_bilstm_scratch.pt", map_location=DEVICE)
model_lstm.load_state_dict(ckpt["model_state"])
_, _, y_test_true_lstm, y_test_pred_lstm = eval_lstm(test_loader_lstm)
tier2_metrics = compute_metrics(y_test_true_lstm, y_test_pred_lstm)
tier2_metrics["Val_Macro_F1"] = round(float(best_val_f1_lstm), 4)
tier2_metrics["Train_Sec"] = lstm_train_sec

print("\\n================ TIER 2 (BiLSTM + Attention Scratch) HELD-OUT TEST ================")
print(classification_report(y_test_true_lstm, y_test_pred_lstm, target_names=LABELS, digits=4))
plot_curves(history_lstm, "Tier 2: BiLSTM + Attention (Scratch)", OUT_DIR / "tier2_bilstm_curves.png")
plot_confusion(y_test_true_lstm, y_test_pred_lstm, "Tier 2: BiLSTM + Attention (Scratch)", OUT_DIR / "tier2_bilstm_confusion_matrix.png")
    """),

    md_cell("""
---
## Tier 3: Fine-Tuned Multilingual Transformer (`xlm-roberta-base`)
* Contextual self-attention handles contrastive phrasing (*"great design, but broke in a week"*).
* Multilingual backbone natively supports both **English storefront reviews** and **Roman Urdu / multilingual YouTube comments**.
    """),

    code_cell("""
# Cell 6: Fine-tune XLM-RoBERTa-base with PyTorch AMP (FP16) on GPU
MODEL_NAME = "xlm-roberta-base"
MAX_LEN_TR = 160
BATCH_TR = 32
EPOCHS_TR = 3
LR_TR = 2e-5

tokenizer_tr = AutoTokenizer.from_pretrained(MODEL_NAME)

class TransformerDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.labels[idx])

def collate_tr(batch):
    texts, labels = zip(*batch)
    enc = tokenizer_tr(
        list(texts), padding=True, truncation=True, max_length=MAX_LEN_TR, return_tensors="pt"
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

train_loader_tr = DataLoader(TransformerDataset(df_train["full_text"].astype(str).tolist(), y_train),
                             batch_size=BATCH_TR, shuffle=True, collate_fn=collate_tr, num_workers=2, pin_memory=True)
val_loader_tr   = DataLoader(TransformerDataset(df_val["full_text"].astype(str).tolist(), y_val),
                             batch_size=64, shuffle=False, collate_fn=collate_tr, num_workers=2, pin_memory=True)
test_loader_tr  = DataLoader(TransformerDataset(df_test["full_text"].astype(str).tolist(), y_test),
                             batch_size=64, shuffle=False, collate_fn=collate_tr, num_workers=2, pin_memory=True)

model_tr = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
).to(DEVICE)

criterion_tr = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
optimizer_tr = torch.optim.AdamW(model_tr.parameters(), lr=LR_TR, weight_decay=0.01)
total_steps = len(train_loader_tr) * EPOCHS_TR
scheduler_tr = get_linear_schedule_with_warmup(
    optimizer_tr, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)
use_amp = (DEVICE.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def eval_tr(loader):
    model_tr.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model_tr(**batch).logits
                loss = criterion_tr(logits, labels)
            total_loss += loss.item() * len(labels)
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return total_loss / len(trues), f1_score(trues, preds, average="macro"), np.array(trues), np.array(preds)

best_val_f1_tr = -1.0
history_tr = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
best_tr_dir = OUT_DIR / "tier3_xlm_roberta_best"

t0 = time.perf_counter()
for epoch in range(1, EPOCHS_TR + 1):
    model_tr.train()
    tr_loss, tr_preds, tr_trues = 0.0, [], []
    for step, batch in enumerate(train_loader_tr, start=1):
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer_tr.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model_tr(**batch).logits
            loss = criterion_tr(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_tr)
        nn.utils.clip_grad_norm_(model_tr.parameters(), 1.0)
        scaler.step(optimizer_tr)
        scaler.update()
        scheduler_tr.step()

        tr_loss += loss.item() * len(labels)
        tr_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        tr_trues.extend(labels.cpu().numpy())

        if step % 200 == 0:
            print(f"  Epoch {epoch} | Step {step}/{len(train_loader_tr)} | Running Loss: {tr_loss/len(tr_trues):.4f}")

    tr_loss /= len(tr_trues)
    tr_f1 = f1_score(tr_trues, tr_preds, average="macro")
    val_loss, val_f1, _, _ = eval_tr(val_loader_tr)

    history_tr["train_loss"].append(tr_loss)
    history_tr["val_loss"].append(val_loss)
    history_tr["train_f1"].append(tr_f1)
    history_tr["val_f1"].append(val_f1)
    print(f"Epoch {epoch:02d}/{EPOCHS_TR} | Train Loss: {tr_loss:.4f} F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

    if val_f1 > best_val_f1_tr:
        best_val_f1_tr = val_f1
        model_tr.save_pretrained(best_tr_dir)
        tokenizer_tr.save_pretrained(best_tr_dir)

tr_train_sec = round(time.perf_counter() - t0, 1)

# Reload best checkpoint & evaluate on held-out Test
model_tr = AutoModelForSequenceClassification.from_pretrained(best_tr_dir).to(DEVICE)
_, _, y_test_true_tr, y_test_pred_tr = eval_tr(test_loader_tr)
tier3_metrics = compute_metrics(y_test_true_tr, y_test_pred_tr)
tier3_metrics["Val_Macro_F1"] = round(float(best_val_f1_tr), 4)
tier3_metrics["Train_Sec"] = tr_train_sec

print("\\n================ TIER 3 (Fine-Tuned XLM-RoBERTa) HELD-OUT TEST ================")
print(classification_report(y_test_true_tr, y_test_pred_tr, target_names=LABELS, digits=4))
plot_curves(history_tr, "Tier 3: Fine-Tuned XLM-RoBERTa-base", OUT_DIR / "tier3_xlm_roberta_curves.png")
plot_confusion(y_test_true_tr, y_test_pred_tr, "Tier 3: Fine-Tuned XLM-RoBERTa-base", OUT_DIR / "tier3_xlm_roberta_confusion_matrix.png")
    """),

    code_cell("""
# Cell 7: Master 3-Tier Benchmark Table & Downloadable Results Package
summary_rows = [
    {
        "Tier": "Baseline",
        "Model": "Majority Class",
        "Val_Macro_F1": 0.1887,
        "Accuracy": 0.4023,
        "Macro_Precision": 0.1341,
        "Macro_Recall": 0.3333,
        "Macro_F1": 0.1912,
        "Weighted_F1": 0.2308,
        "Train_Sec": 0.0,
    },
    {
        "Tier": "Tier 1 (Classical ML)",
        "Model": "Logistic Regression (TF-IDF, de-leaked)",
        "Val_Macro_F1": 0.6957,
        "Accuracy": 0.7355,
        "Macro_Precision": 0.6951,
        "Macro_Recall": 0.6968,
        "Macro_F1": 0.6948,
        "Weighted_F1": 0.7412,
        "Train_Sec": 6.3,
    },
    {"Tier": "Tier 2 (Deep Learning Scratch)", "Model": "BiLSTM + Masked Attention (PyTorch)", **tier2_metrics},
    {"Tier": "Tier 3 (Fine-Tuned Transformer)", "Model": "XLM-RoBERTa-base (PyTorch)", **tier3_metrics},
]

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_DIR / "all_tiers_benchmark.csv", index=False)
print("\\n================ MASTER 3-TIER COMPARISON (TEST n=4,522) ================")
print(df_summary.to_string(index=False))

# Save held-out test predictions for downstream error analysis
pd.DataFrame({
    "parent_asin": df_test["parent_asin"],
    "category": df_test["category"],
    "true_label": [ID2LABEL[i] for i in y_test],
    "tier2_bilstm_pred": [ID2LABEL[i] for i in y_test_pred_lstm],
    "tier3_xlmr_pred": [ID2LABEL[i] for i in y_test_pred_tr],
    "full_text": df_test["full_text"],
}).to_csv(OUT_DIR / "test_predictions_tier2_tier3.csv", index=False)

# Zip plots, metrics, predictions & Tier-2 scratch weights for quick download
!zip -r colab_results_light.zip colab_outputs/*.png colab_outputs/*.csv colab_outputs/tier2_bilstm_scratch.pt
print("\\nReady! Download colab_results_light.zip below (and optionally mount Google Drive to keep tier3_xlm_roberta_best/ weights).")
files.download("colab_results_light.zip")
    """),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out_nb = NOTEBOOKS_DIR / "05_tier2_tier3_pytorch_gpu.ipynb"
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created Colab GPU Notebook: {out_nb}")