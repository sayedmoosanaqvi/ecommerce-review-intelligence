import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/cross_platform")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Mixed_Neutral", "Positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.]")
MAX_LEN_LSTM = 160
BATCH_SIZE = 256


# ---------------------------------------------------------
# EXACT TIER-2 PYTORCH ARCHITECTURE (MATCHES COLAB CHECKPOINT)
# ---------------------------------------------------------
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

    def forward(self, input_ids, return_attention=False):
        mask = (input_ids != 0)
        x = self.emb_drop(self.embedding(input_ids))
        h, _ = self.lstm(x)
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        logits = self.head(self.norm(context))
        if return_attention:
            return logits, weights
        return logits


def simple_tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def encode_texts(texts, vocab, max_len=MAX_LEN_LSTM):
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = simple_tokenize(text)[:max_len]
        ids = [vocab.get(t, 1) for t in tokens] or [1]
        encoded[i, : len(ids)] = ids
    return encoded


def shannon_entropy(probs):
    """Normalized Shannon entropy in [0, 1] across 3 classes (higher = more model uncertainty)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(probs.shape[1])


def predict_tier1(pipe, texts):
    t0 = time.perf_counter()
    probs = pipe.predict_proba(texts)
    elapsed = time.perf_counter() - t0
    # Align probability columns with LABELS order
    clf_classes = list(pipe.named_steps["clf"].classes_)
    col_order = [clf_classes.index(lbl) for lbl in LABELS]
    probs = probs[:, col_order]
    pred_ids = np.argmax(probs, axis=1)
    return pred_ids, probs, elapsed


def predict_tier2(model, vocab, texts):
    t0 = time.perf_counter()
    X_ids = encode_texts(texts, vocab)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_ids)), batch_size=BATCH_SIZE, shuffle=False)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (bx,) in loader:
            logits = model(bx)
            probs = torch.softmax(logits, dim=-1).numpy()
            all_probs.append(probs)
    probs = np.vstack(all_probs)
    elapsed = time.perf_counter() - t0
    pred_ids = np.argmax(probs, axis=1)
    return pred_ids, probs, elapsed


# Expressive issue/intent triage patterns (secondary analytical signals, not verified transactions)
TRIAGE_PATTERNS = {
    "signal_refund_intent": re.compile(r"\b(refund|money back|get my money|chargeback)\b", re.I),
    "signal_return_replace": re.compile(r"\b(return(ed|ing)?|send(ing)? it back|replace(ment)?|exchange)\b", re.I),
    "signal_product_defect": re.compile(r"\b(broke|broken|defect(ive)?|stopped working|dead on arrival|malfunction|glitch|overheat)\b", re.I),
    "signal_misleading_fake": re.compile(r"\b(scam|fake|counterfeit|not as described|misleading|paid review|sponsored lie|waste of money)\b", re.I),
    "signal_delivery_service": re.compile(r"\b(late delivery|never arrived|damaged box|customer service|no reply|support ignored|terrible shipping)\b", re.I),
}


def compute_triage_rates(texts):
    s = pd.Series(texts, dtype="string").fillna("")
    return {name: round(float(s.str.contains(pat).mean() * 100), 2) for name, pat in TRIAGE_PATTERNS.items()}


def explain_with_attention(model, vocab, text, top_k=5):
    tokens = simple_tokenize(text)[:MAX_LEN_LSTM]
    ids = [vocab.get(t, 1) for t in tokens] or [1]
    padded = ids + [0] * (MAX_LEN_LSTM - len(ids))
    inp = torch.tensor([padded], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, weights = model(inp, return_attention=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    attn = weights.squeeze(0).numpy()[: len(tokens)]
    top_idx = np.argsort(attn)[-top_k:][::-1]
    top_tokens = [(tokens[i], round(float(attn[i]), 3)) for i in top_idx if i < len(tokens)]
    pred_label = LABELS[int(np.argmax(probs))]
    return pred_label, float(np.max(probs)), top_tokens


def plot_domain_shift(amz_p2, yt_p2, amz_pred2, yt_pred2, yt_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Predicted Class Distribution Shift (Amazon vs YouTube)
    dist_df = pd.DataFrame({
        "Class": LABELS * 2,
        "Share (%)": (
            [np.mean(amz_pred2 == i) * 100 for i in range(3)]
            + [np.mean(yt_pred2 == i) * 100 for i in range(3)]
        ),
        "Platform": ["Amazon Storefront (n=4,522)"] * 3 + ["YouTube Social Comments (n=27,933)"] * 3,
    })
    sns.barplot(data=dist_df, x="Class", y="Share (%)", hue="Platform", ax=axes[0], palette="Set2")
    axes[0].set_title("Tier-2 BiLSTM Predicted Distribution\nStorefront vs. Social Commerce")
    axes[0].set_xlabel("")
    axes[0].legend(fontsize=8)

    # Panel 2: Prediction Entropy Shift (Uncertainty Diagnostic)
    ent_amz = shannon_entropy(amz_p2)
    ent_yt = shannon_entropy(yt_p2)
    sns.kdeplot(ent_amz, label=f"Amazon In-Domain (mean={ent_amz.mean():.3f})", fill=True, ax=axes[1])
    sns.kdeplot(ent_yt, label=f"YouTube Out-of-Domain (mean={ent_yt.mean():.3f})", fill=True, ax=axes[1])
    axes[1].set_title("Predictive Uncertainty (Normalized Entropy)\nHigher = Greater Domain Shift")
    axes[1].set_xlabel("Shannon Entropy [0 = Certain, 1 = Uniform]")
    axes[1].legend(fontsize=8)

    # Panel 3: YouTube Predicted Sentiment by E-Commerce Vertical
    ct = pd.crosstab(yt_df["domain"], yt_df["tier2_pred"], normalize="index")[LABELS] * 100
    ct.plot(kind="barh", stacked=True, ax=axes[2], color=["#d9534f", "#f0ad4e", "#5cb85c"])
    axes[2].set_title("YouTube Comments by Retail Vertical\n(Tier-2 BiLSTM Predictions)")
    axes[2].set_xlabel("Percentage of Comments (%)")
    axes[2].set_ylabel("")
    axes[2].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main():
    print("Loading locally trained zero-API models (Tier 1 Classical + Tier 2 PyTorch Scratch)...")
    tier1_pipe = joblib.load(MODELS_DIR / "tier1_best_classical.joblib")
    ckpt = torch.load(MODELS_DIR / "tier2_bilstm_scratch.pt", map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    tier2_model = BiLSTMAttentionClassifier(vocab_size=len(vocab))
    tier2_model.load_state_dict(ckpt["model_state"])
    tier2_model.eval()

    df_amz = pd.read_csv(PROCESSED_DIR / "amazon_test.csv")
    df_yt = pd.read_csv(PROCESSED_DIR / "youtube_clean.csv")
    amz_texts = df_amz["full_text"].astype(str).tolist()
    yt_texts = df_yt["clean_text"].astype(str).tolist()
    y_amz_true = df_amz["sentiment_3class"].map(LABEL2ID).values

    # 1. In-Domain Storefront Inference & Verification
    print(f"\n1/3: Running offline inference on Amazon Held-Out Test (n={len(amz_texts):,})...")
    amz_pred1, amz_p1, t1_amz = predict_tier1(tier1_pipe, amz_texts)
    amz_pred2, amz_p2, t2_amz = predict_tier2(tier2_model, vocab, amz_texts)

    print(
        f"   Tier 1 Verified -> Acc: {accuracy_score(y_amz_true, amz_pred1):.4f} | "
        f"Macro-F1: {f1_score(y_amz_true, amz_pred1, average='macro'):.4f} | "
        f"Latency: {t1_amz / len(amz_texts) * 1000:.3f} ms/review"
    )
    print(
        f"   Tier 2 Verified -> Acc: {accuracy_score(y_amz_true, amz_pred2):.4f} | "
        f"Macro-F1: {f1_score(y_amz_true, amz_pred2, average='macro'):.4f} | "
        f"Latency: {t2_amz / len(amz_texts) * 1000:.3f} ms/review"
    )

    # 2. Out-of-Domain YouTube Inference
    print(f"\n2/3: Running offline inference on YouTube Social-Commerce Corpus (n={len(yt_texts):,})...")
    yt_pred1, yt_p1, t1_yt = predict_tier1(tier1_pipe, yt_texts)
    yt_pred2, yt_p2, t2_yt = predict_tier2(tier2_model, vocab, yt_texts)

    print(f"   Tier 1 completed in {t1_yt:.2f}s ({t1_yt / len(yt_texts) * 1000:.3f} ms/comment)")
    print(f"   Tier 2 completed in {t2_yt:.2f}s ({t2_yt / len(yt_texts) * 1000:.3f} ms/comment)")

    # 3. Domain-Shift Comparison Metrics
    amz_disagree = float(np.mean(amz_pred1 != amz_pred2) * 100)
    yt_disagree = float(np.mean(yt_pred1 != yt_pred2) * 100)

    summary = pd.DataFrame([
        {
            "Platform": "Amazon_Storefront_Test",
            "Samples": len(amz_texts),
            "Tier1_Mean_Conf": round(float(amz_p1.max(axis=1).mean()), 4),
            "Tier2_Mean_Conf": round(float(amz_p2.max(axis=1).mean()), 4),
            "Tier1_Mean_Entropy": round(float(shannon_entropy(amz_p1).mean()), 4),
            "Tier2_Mean_Entropy": round(float(shannon_entropy(amz_p2).mean()), 4),
            "Tier1_vs_Tier2_Disagreement_%": round(amz_disagree, 2),
            "Tier2_Neg_%": round(float(np.mean(amz_pred2 == 0) * 100), 2),
            "Tier2_Mix_%": round(float(np.mean(amz_pred2 == 1) * 100), 2),
            "Tier2_Pos_%": round(float(np.mean(amz_pred2 == 2) * 100), 2),
        },
        {
            "Platform": "YouTube_Social_Comments",
            "Samples": len(yt_texts),
            "Tier1_Mean_Conf": round(float(yt_p1.max(axis=1).mean()), 4),
            "Tier2_Mean_Conf": round(float(yt_p2.max(axis=1).mean()), 4),
            "Tier1_Mean_Entropy": round(float(shannon_entropy(yt_p1).mean()), 4),
            "Tier2_Mean_Entropy": round(float(shannon_entropy(yt_p2).mean()), 4),
            "Tier1_vs_Tier2_Disagreement_%": round(yt_disagree, 2),
            "Tier2_Neg_%": round(float(np.mean(yt_pred2 == 0) * 100), 2),
            "Tier2_Mix_%": round(float(np.mean(yt_pred2 == 1) * 100), 2),
            "Tier2_Pos_%": round(float(np.mean(yt_pred2 == 2) * 100), 2),
        },
    ])

    triage_amz = compute_triage_rates(amz_texts)
    triage_yt = compute_triage_rates(yt_texts)
    df_triage = pd.DataFrame([
        {"Platform": "Amazon_Storefront_Test", **triage_amz},
        {"Platform": "YouTube_Social_Comments", **triage_yt},
    ])

    # Save enriched YouTube predictions
    df_yt["tier1_pred"] = [LABELS[i] for i in yt_pred1]
    df_yt["tier2_pred"] = [LABELS[i] for i in yt_pred2]
    df_yt["tier2_confidence"] = np.round(yt_p2.max(axis=1), 4)
    df_yt["tier2_entropy"] = np.round(shannon_entropy(yt_p2), 4)
    df_yt["models_disagree"] = yt_pred1 != yt_pred2

    summary.to_csv(REPORTS_DIR / "domain_shift_summary.csv", index=False)
    df_triage.to_csv(REPORTS_DIR / "issue_signal_comparison.csv", index=False)
    df_yt.to_csv(REPORTS_DIR / "youtube_28k_predictions.csv", index=False, encoding="utf-8-sig")

    plot_path = REPORTS_DIR / "domain_shift_comparison.png"
    plot_domain_shift(amz_p2, yt_p2, amz_pred2, yt_pred2, df_yt, plot_path)

    print("\n================ CROSS-PLATFORM DOMAIN SHIFT SUMMARY ================")
    print(summary.to_string(index=False))

    print("\n================ EXPRESSIVE ISSUE & INTENT SIGNALS (%) ================")
    print(df_triage.to_string(index=False))

    # Show 3 real BiLSTM Attention Explanations on YouTube Comments
    print("\n================ SAMPLE NEURAL ATTENTION EXPLANATIONS (YOUTUBE) ================")
    sample_comments = df_yt[df_yt["word_count_clean"].between(10, 35)].sample(3, random_state=42)
    for row in sample_comments.itertuples():
        lbl, conf, top_toks = explain_with_attention(tier2_model, vocab, row.clean_text)
        tok_str = ", ".join(f"{w} ({wt:.2f})" for w, wt in top_toks)
        print(f"  [{row.domain}] Pred={lbl} ({conf:.2%}) | Top Attention: {tok_str}")
        print(f"     Text: {row.clean_text[:130]}...")

    print(f"\nSaved Domain-Shift Plot : {plot_path}")
    print(f"Saved Summary CSV       : {REPORTS_DIR / 'domain_shift_summary.csv'}")
    print(f"Saved YouTube Preds CSV : {REPORTS_DIR / 'youtube_28k_predictions.csv'}")


if __name__ == "__main__":
    main()