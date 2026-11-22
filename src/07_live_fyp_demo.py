import re
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# MODEL LOADING (100% OFFLINE LOCAL WEIGHTS - ZERO LLM API)
# ---------------------------------------------------------
MODELS_DIR = Path("models")
LABELS = ["Negative", "Mixed_Neutral", "Positive"]
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.]")
MAX_LEN_LSTM = 160


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


tier1_pipe = joblib.load(MODELS_DIR / "tier1_best_classical.joblib")
tier1_classes = list(tier1_pipe.named_steps["clf"].classes_)
tier1_order = [tier1_classes.index(lbl) for lbl in LABELS]

ckpt = torch.load(MODELS_DIR / "tier2_bilstm_scratch.pt", map_location="cpu", weights_only=False)
vocab = ckpt["vocab"]
tier2_model = BiLSTMAttentionClassifier(vocab_size=len(vocab))
tier2_model.load_state_dict(ckpt["model_state"])
tier2_model.eval()

TRIAGE_PATTERNS = {
    "Refund Intent (IIO10/16C)": re.compile(r"\b(?:refund|money back|get my money|chargeback)\b", re.I),
    "Return / Replacement Intent": re.compile(r"\b(?:return(?:ed|ing)?|send(?:ing)? it back|replace(?:ment)?|exchange)\b", re.I),
    "Product Defect (IIO04)": re.compile(r"\b(?:broke|broken|defect(?:ive)?|stopped working|dead on arrival|malfunction|glitch|overheat)\b", re.I),
    "Misleading / Fake Suspicion (IIO01/05)": re.compile(r"\b(?:scam|fake|counterfeit|not as described|misleading|paid review|sponsored lie|waste of money)\b", re.I),
    "Delivery / Service Failure (IIO06/07)": re.compile(r"\b(?:late delivery|never arrived|damaged box|customer service|no reply|support ignored|terrible shipping)\b", re.I),
}

app = FastAPI(
    title="E-Commerce Review Intelligence API (Zero-API Local Models)",
    description="Offline inference microservice powered by custom-trained Scikit-Learn & PyTorch BiLSTM+Attention models.",
    version="1.0.0",
)


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=5000)
    star_rating: Optional[int] = Field(None, ge=1, le=5)


def shannon_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(probs)))


@app.post("/api/analyze-review")
def analyze_review(req: ReviewRequest):
    text = req.text.strip()

    # Tier 1: TF-IDF + Logistic Regression
    t0 = time.perf_counter()
    p1_raw = tier1_pipe.predict_proba([text])[0]
    p1 = p1_raw[tier1_order]
    t1_ms = round((time.perf_counter() - t0) * 1000, 2)
    pred1 = LABELS[int(np.argmax(p1))]

    # Tier 2: Scratch-Trained PyTorch BiLSTM + Masked Attention
    t0 = time.perf_counter()
    tokens = TOKEN_RE.findall(text.lower())[:MAX_LEN_LSTM]
    ids = [vocab.get(tok, 1) for tok in tokens] or [1]
    oov_count = sum(1 for tok in tokens if tok not in vocab)
    oov_rate = round((oov_count / max(len(tokens), 1)) * 100, 1)

    padded = ids + [0] * (MAX_LEN_LSTM - len(ids))
    inp = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        logits, attn = tier2_model(inp, return_attention=True)
        p2 = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    t2_ms = round((time.perf_counter() - t0) * 1000, 2)
    pred2 = LABELS[int(np.argmax(p2))]
    ent2 = round(shannon_entropy(p2), 4)

    attn_scores = attn.squeeze(0).numpy()[: len(tokens)]
    max_attn = float(attn_scores.max()) if len(attn_scores) > 0 else 1.0
    token_heatmap = [
        {
            "token": tok,
            "weight": round(float(w), 4),
            "norm": round(float(w / max(max_attn, 1e-9)), 3),
            "in_vocab": tok in vocab,
        }
        for tok, w in zip(tokens, attn_scores)
    ]

    # Issue & Refund/Return Triage Signals
    detected_signals = [name for name, pat in TRIAGE_PATTERNS.items() if pat.search(text)]

    # Trustworthiness, Domain-Shift & Credibility Flags
    alerts = []
    if req.star_rating is not None:
        if req.star_rating >= 4 and pred2 == "Negative" and float(p2[0]) >= 0.55:
            alerts.append("Rating-Text Contradiction: High star rating (4-5★) contradicts strongly negative review text — flag for human audit.")
        elif req.star_rating <= 2 and pred2 == "Positive" and float(p2[2]) >= 0.55:
            alerts.append("Rating-Text Contradiction: Low star rating (1-2★) contradicts strongly positive review text — flag for human audit.")

    if len(tokens) <= 4:
        alerts.append("Low Diagnosticity (IIO02): Very short review lacks actionable product-feature detail.")
    if oov_rate >= 30.0:
        alerts.append(f"Out-of-Domain / Multilingual Shift Warning: {oov_rate}% out-of-vocabulary tokens (e.g., Roman Urdu/slang) — route to multilingual Tier-3 XLM-RoBERTa.")
    if pred1 != pred2:
        alerts.append(f"Cross-Model Uncertainty: Tier 1 ({pred1}) and Tier 2 ({pred2}) disagree — moderate confidence.")
    if ent2 >= 0.72:
        alerts.append(f"High Predictive Entropy ({ent2}): Model confidence is low; recommend human verification.")

    return {
        "tier1_classical": {
            "model": "TF-IDF + Logistic Regression (Test Macro-F1: 0.6948)",
            "prediction": pred1,
            "confidence": round(float(np.max(p1)), 4),
            "probabilities": {lbl: round(float(val), 4) for lbl, val in zip(LABELS, p1)},
            "latency_ms": t1_ms,
        },
        "tier2_pytorch_scratch": {
            "model": "PyTorch BiLSTM + Masked Attention (Random Init | Test Macro-F1: 0.7013)",
            "prediction": pred2,
            "confidence": round(float(np.max(p2)), 4),
            "entropy": ent2,
            "oov_rate_pct": oov_rate,
            "probabilities": {lbl: round(float(val), 4) for lbl, val in zip(LABELS, p2)},
            "latency_ms": t2_ms,
            "token_heatmap": token_heatmap,
        },
        "detected_issue_signals": detected_signals,
        "credibility_and_governance_alerts": alerts,
        "api_cost_usd": 0.0,
        "external_llm_used": False,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Review Intelligence Engine — Zero-API Local PyTorch Demo</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#f1f5f9; --muted:#94a3b8; --accent:#38bdf8; --neg:#ef4444; --mix:#f59e0b; --pos:#22c55e; }
  * { box-sizing:border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  body { background:var(--bg); color:var(--text); margin:0; padding:24px; }
  .wrap { max-width:1050px; margin:0 auto; }
  .header { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:20px; }
  .pill { background:#064e3b; color:#6ee7b7; border:1px solid #10b981; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:600; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px; margin-bottom:18px; }
  textarea { width:100%; min-height:110px; background:#0b1220; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:12px; font-size:15px; resize:vertical; }
  .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:12px; }
  select, button { padding:10px 16px; border-radius:8px; border:1px solid var(--border); font-size:14px; }
  select { background:#0b1220; color:var(--text); }
  button.primary { background:var(--accent); color:#0f172a; font-weight:700; border:none; cursor:pointer; }
  button.preset { background:#0b1220; color:#cbd5e1; cursor:pointer; font-size:12px; padding:7px 11px; }
  button.preset:hover { border-color:var(--accent); color:white; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }
  .badge { display:inline-block; padding:4px 10px; border-radius:6px; font-weight:700; font-size:13px; }
  .Negative { background:rgba(239,68,68,.18); color:#fca5a5; border:1px solid var(--neg); }
  .Mixed_Neutral { background:rgba(245,158,11,.18); color:#fcd34d; border:1px solid var(--mix); }
  .Positive { background:rgba(34,197,94,.18); color:#86efac; border:1px solid var(--pos); }
  .bar-row { display:flex; align-items:center; gap:8px; margin:6px 0; font-size:13px; }
  .bar-bg { flex:1; height:10px; background:#0b1220; border-radius:999px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:999px; }
  .token { display:inline-block; padding:3px 6px; margin:3px 2px; border-radius:5px; font-size:14px; }
  .oov { border-bottom:2px dashed #f59e0b; }
  .tag { display:inline-block; background:#312e81; color:#c7d2fe; border:1px solid #6366f1; padding:5px 10px; border-radius:8px; margin:4px 6px 4px 0; font-size:13px; }
  .alert { background:rgba(245,158,11,.12); border-left:4px solid var(--mix); padding:10px 12px; border-radius:6px; margin:8px 0; font-size:13px; color:#fde68a; }
  .muted { color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <h2 style="margin:0;">Cross-Platform E-Commerce Review Intelligence</h2>
      <div class="muted">Custom-Trained Scikit-Learn &amp; PyTorch BiLSTM+Attention Engine &bull; <a href="/docs" style="color:#38bdf8;">Open REST API (/docs)</a></div>
    </div>
    <span class="pill">100% OFFLINE LOCAL WEIGHTS &bull; $0.00 API COST &bull; NO GROQ/LLM</span>
  </div>

  <div class="card">
    <div style="margin-bottom:8px; font-size:13px; color:#94a3b8;">Quick Demo Presets:</div>
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
      <button class="preset" onclick="loadPreset('mixed')">Mixed Contrastive Review (IIO03)</button>
      <button class="preset" onclick="loadPreset('defect')">Defect + Refund Request (IIO04/16C)</button>
      <button class="preset" onclick="loadPreset('mismatch')">Suspicious 5★ Mismatch (IIO01)</button>
      <button class="preset" onclick="loadPreset('urdu')">Roman Urdu YouTube Shift Test</button>
    </div>
    <textarea id="reviewText" placeholder="Paste any product review or YouTube shopping comment here..."></textarea>
    <div class="controls">
      <label class="muted">Optional Storefront Star Rating:</label>
      <select id="starRating">
        <option value="">None (YouTube / Unrated Comment)</option>
        <option value="5">5 Stars</option>
        <option value="4">4 Stars</option>
        <option value="3">3 Stars</option>
        <option value="2">2 Stars</option>
        <option value="1">1 Star</option>
      </select>
      <button class="primary" onclick="analyze()">Run Offline Models</button>
    </div>
  </div>

  <div id="results" style="display:none;">
    <div class="grid">
      <div class="card" id="t1Card"></div>
      <div class="card" id="t2Card"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0;">PyTorch BiLSTM Learned Attention Heatmap</h3>
      <div class="muted" style="margin-bottom:10px;">Brighter blue = higher neural attention weight. Dashed amber underline = out-of-vocabulary (OOV) token.</div>
      <div id="heatmap"></div>
    </div>
    <div class="grid">
      <div class="card">
        <h3 style="margin-top:0;">Expressive Issue &amp; Intent Triage Signals</h3>
        <div id="signals"></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">Credibility &amp; Domain-Shift Governance Alerts</h3>
        <div id="alerts"></div>
      </div>
    </div>
  </div>
</div>

<script>
const PRESETS = {
  mixed: { stars: "3", text: "The build quality and packaging are genuinely great, and the sound is crisp. However, the battery drains way too fast and Bluetooth disconnects every 20 minutes." },
  defect: { stars: "1", text: "Completely broken after three days! The charger stopped working and the listing was misleading — not as described. I want a full refund and am returning this item." },
  mismatch: { stars: "5", text: "Terrible cheap plastic, total waste of money and customer service ignored my messages." },
  urdu: { stars: "", text: "Bhai ab Soundpeats pearl clip pro vs qcy crossky C30s ka comparison karyn please" }
};
function loadPreset(k) {
  document.getElementById("reviewText").value = PRESETS[k].text;
  document.getElementById("starRating").value = PRESETS[k].stars;
  analyze();
}
function renderModelCard(title, data, extraHtml="") {
  const colors = { Negative:"#ef4444", Mixed_Neutral:"#f59e0b", Positive:"#22c55e" };
  let bars = "";
  for (const [cls, prob] of Object.entries(data.probabilities)) {
    const pct = (prob * 100).toFixed(1);
    bars += `<div class="bar-row"><span style="width:110px;">${cls}</span><div class="bar-bg"><div class="bar-fill" style="width:${pct}%; background:${colors[cls]};"></div></div><span style="width:48px; text-align:right;">${pct}%</span></div>`;
  }
  return `<h3 style="margin-top:0;">${title}</h3>
    <div class="muted" style="margin-bottom:10px;">${data.model}</div>
    <div style="margin-bottom:12px;"><span class="badge ${data.prediction}">${data.prediction}</span> &nbsp; <span class="muted">Confidence: <b>${(data.confidence*100).toFixed(1)}%</b> &bull; Latency: <b>${data.latency_ms} ms</b></span></div>
    ${bars}${extraHtml}`;
}
async function analyze() {
  const text = document.getElementById("reviewText").value.trim();
  if (text.length < 2) return;
  const sr = document.getElementById("starRating").value;
  const payload = { text, star_rating: sr ? parseInt(sr) : null };
  const res = await fetch("/api/analyze-review", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  const d = await res.json();
  document.getElementById("results").style.display = "block";
  document.getElementById("t1Card").innerHTML = renderModelCard("Tier 1: Classical ML", d.tier1_classical);
  document.getElementById("t2Card").innerHTML = renderModelCard(
    "Tier 2: PyTorch BiLSTM (Scratch)",
    d.tier2_pytorch_scratch,
    `<div class="muted" style="margin-top:10px;">Entropy: <b>${d.tier2_pytorch_scratch.entropy}</b> &bull; OOV Tokens: <b>${d.tier2_pytorch_scratch.oov_rate_pct}%</b></div>`
  );
  document.getElementById("heatmap").innerHTML = d.tier2_pytorch_scratch.token_heatmap.map(t => {
    const alpha = (0.08 + t.norm * 0.72).toFixed(2);
    return `<span class="token ${t.in_vocab ? '' : 'oov'}" style="background:rgba(56,189,248,${alpha});" title="Attention: ${t.weight} | In Vocab: ${t.in_vocab}">${t.token}</span>`;
  }).join(" ");
  document.getElementById("signals").innerHTML = d.detected_issue_signals.length
    ? d.detected_issue_signals.map(s => `<span class="tag">${s}</span>`).join("")
    : `<span class="muted">No defect, refund, or scam triage flags triggered.</span>`;
  document.getElementById("alerts").innerHTML = d.credibility_and_governance_alerts.length
    ? d.credibility_and_governance_alerts.map(a => `<div class="alert">${a}</div>`).join("")
    : `<span class="muted">No credibility contradiction or domain-shift alerts triggered.</span>`;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return DASHBOARD_HTML