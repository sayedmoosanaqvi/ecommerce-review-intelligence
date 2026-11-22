# Cross-Platform E-Commerce Review Intelligence

**Zero-API Supervised Machine Learning & PyTorch Deep Learning Across Structured Storefront Reviews and Unstructured YouTube Social-Commerce Comments**

This repository implements an end-to-end Applied NLP and Supervised Machine Learning pipeline designed to classify consumer feedback, audit dataset target leakage, quantify cross-platform domain shift, and serve offline review-intelligence predictions via a local REST API—without relying on commercial hosted LLM APIs.

---

## 1. Research Motivation & Engineering Problem

Modern online shoppers split their trust between two fundamentally different sources of consumer feedback:
1. **Structured Storefront Reviews** (e.g., Amazon, online retail stores) — SKU-linked, star-rated, longer, and rich in post-purchase defect/return signals, yet vulnerable to rating manipulation and hidden mixed opinions.
2. **Unstructured Social-Commerce Comments** (e.g., YouTube product/service review videos) — conversational, ~4× shorter, emoji/slang-heavy, and mixed with non-diagnostic creator chatter or spam, yet rich in candid pre-purchase skepticism.

Instead of wrapping an external LLM API, this project trains and benchmarks **three independent model tiers** locally and on GPU—from classical bag-of-words baselines and a **100% from-scratch PyTorch BiLSTM + Masked Attention network** to a **fine-tuned multilingual Transformer (`XLM-RoBERTa-base`)**—evaluated strictly on product-grouped, leakage-free splits.

---

## 2. Dual-Corpus Dataset Audit (60,500 Raw Consumer Texts)

| Corpus | Source (100% ToS-Compliant) | Raw Rows | Clean Unique Rows | Grouping Unit | Median / Mean Words | Notable Characteristics |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **Storefront Reviews** | McAuley Lab Amazon Reviews 2023 (5 matched verticals: Electronics, Fashion, Home & Kitchen, Beauty, Software; 1,200/star) | 30,000 | **29,934** | `parent_asin` (24,753 unique products) | 37.0 / 68.6 | 74.60% verified purchase; **2,034 auto-generated star titles stripped** to eliminate target leakage |
| **Social-Commerce Comments** | Official YouTube Data API v3 (`01_youtube_crawler.py`, 6 e-commerce verticals, max 600 top-level comments/video) | 30,500 | **28,354** | `video_id` (71 review videos) | 10.0 / 19.4 | 32.06% contain emojis/non-ASCII; includes English & Roman Urdu code-mixed buyer comments |

### Leakage-Free Product-Grouped Splits (`parent_asin` Overlap = 0)
* **Train:** `21,001` reviews (`70%`)
* **Validation:** `4,411` reviews (`15%`) — used strictly for model selection and early stopping
* **Held-Out Test:** `4,522` reviews (`15%`) — evaluated once for final reporting

---

## 3. Target-Leakage Ablation Study

Post-hoc feature-weight auditing of initial classical models uncovered hidden shortcut learning: Amazon auto-populates default titles (`"Five Stars"`, `"Three Stars"`, `"One Star"`) when buyers omit a headline, leaking numeric star labels into the text. Stripping **2,034 auto-generated star titles** (`6.78%` of the corpus) eliminated shortcut weights and produced honest semantic baselines:

| `TF-IDF + Logistic Regression` | Test Accuracy | Test Macro-$F_1$ | 5-Fold GroupKFold Macro-$F_1$ | Top Learned Features (Negative / Positive) |
| :--- | :---: | :---: | :---: | :--- |
| **Before De-Leaking** | 75.10% | 0.7083 | 0.7090 ± 0.0052 | `two stars (+6.38)`, `one star (+3.37)` / `four stars (+4.27)` |
| **After De-Leaking (Final)** | **73.55%** | **0.6948** | **0.6873 ± 0.0072** | `not (+5.22)`, `poor (+3.45)`, `waste (+3.39)` / `great (+5.79)`, `love (+5.53)` |

---

## 4. Three-Tier Model Training Benchmark (Held-Out Test, `n = 4,522`)

All tiers were evaluated on identical, de-leaked, product-grouped test splits using class-balanced training objectives:

| Tier | Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro $F_1$ | Weighted $F_1$ | `Mixed_Neutral` $F_1$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Majority Class (`DummyClassifier`) | 40.23% | 0.1341 | 0.3333 | **0.1912** | 0.2308 | 0.0000 |
| **Tier 1 (Classical ML)** | Best of 10 pipelines (`TF-IDF + Logistic Regression`) | 73.55% | 0.6951 | 0.6968 | **0.6948** | 0.7412 | 0.4693 |
| **Tier 2 (Deep Learning Scratch)** | **Custom PyTorch BiLSTM + Masked Attention** *(random init, train-only vocab)* | **74.35%** | 0.7009 | 0.7028 | **0.7013** | 0.7474 | **0.4776** |
| **Tier 3 (Fine-Tuned Transformer)** | **PyTorch `xlm-roberta-base`** *(FP16 AMP, GPU)* | **79.23%** | 0.7478 | 0.7486 | **0.7482** | 0.7931 | **0.5301** |

**Key Finding:** The custom scratch-trained PyTorch BiLSTM outperformed all 10 Scikit-Learn classical pipelines without pretrained weights. Fine-tuning `XLM-RoBERTa-base` delivered the largest improvement on the hardest contrastive class—**`Mixed_Neutral` $F_1$ jumped from `0.4693` to `0.5301` (+6.08 percentage points)**.

### Visual Results
* **Tier 1 Confusion Matrix:** `reports/tier1_classical/tier1_best_confusion_matrix.png`
* **Tier 2 Training Curves & Confusion Matrix:** `reports/tier2_tier3_neural/colab_outputs/tier2_bilstm_curves.png` | `tier2_bilstm_confusion_matrix.png`
* **Tier 3 Training Curves & Confusion Matrix:** `reports/tier2_tier3_neural/colab_outputs/tier3_xlm_roberta_curves.png` | `tier3_xlm_roberta_confusion_matrix.png`

---

## 5. Cross-Platform Domain Shift & Offline CPU Efficiency

Running the locally trained Tier-1 and Tier-2 models completely offline on CPU across all `28,354` cleaned YouTube comments revealed sharp cross-platform domain shift alongside sub-2ms batch latency:

| Platform | Samples | Tier-2 Mean Confidence | Tier-2 Normalized Entropy | Tier-1 vs. Tier-2 Disagreement | Batch CPU Latency (Tier 1 / Tier 2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Amazon Storefront (In-Domain)** | 4,522 | 0.7986 | 0.4406 | 18.95% | 0.093 ms / 1.978 ms per review |
| **YouTube Social Comments (Out-of-Domain)** | 28,354 | **0.6877** (-0.111) | **0.6432** (+0.205) | **30.68%** (+11.54 pp) | 0.024 ms / 1.750 ms per comment |

* **Behavioral Asymmetry:** Amazon reviews exhibited **4.6× higher return/replacement intent** (`7.45%` vs. `1.58%`) and **6.8× higher product-defect mentions** (`3.34%` vs. `0.52%`), whereas YouTube comments exhibited **nearly 2× higher fake/scam/misleading accusations** (`2.86%` vs. `1.64%`).
* **Domain-Shift Plot:** `reports/cross_platform/domain_shift_comparison.png`

---

## 6. Live Offline FastAPI Microservice & Attention Demo

`src/07_live_fyp_demo.py` serves both trained local models at `http://127.0.0.1:8000` (with interactive OpenAPI docs at `/docs`) at **$0.00 API cost** (~1–7 ms single-review latency on CPU):
* Side-by-side Tier-1 (`TF-IDF + LR`) and Tier-2 (`PyTorch BiLSTM + Masked Attention`) probabilities,
* Live **token-level Neural Attention Heatmap** with out-of-vocabulary (OOV) domain-shift detection,
* **Rating-vs.-Text Contradiction Alerts** (flagging suspicious star/text mismatches for human audit),
* Expressive issue triage (`Refund Intent`, `Return/Replacement`, `Product Defect`, `Misleading/Fake Suspicion`, `Delivery/Service Failure`).

---

## 7. Repository Structure & Reproduction

```text
ecommerce-review-intelligence/
├── src/
│   ├── 01_youtube_crawler.py         # Official YouTube Data API v3 crawler + 3-sheet Excel exporter
│   ├── 02_amazon_streamer.py         # HTTPS line-streaming loader for Amazon Reviews 2023
│   ├── 03_data_audit_and_splits.py   # De-leaking, product-grouped splits & 800-row human annotation workbook
│   ├── 04_train_classical_ml.py      # 10-model Scikit-Learn grid, 5-fold GroupKFold CV & feature audit
│   ├── make_colab_notebook.py        # Generates GPU training notebook for Tier 2 & Tier 3
│   ├── 06_cross_platform_analysis.py # Offline CPU benchmark, entropy shift & neural attention analysis
│   └── 07_live_fyp_demo.py           # Zero-API FastAPI microservice + interactive browser dashboard
├── notebooks/
│   └── 05_tier2_tier3_pytorch_gpu.ipynb
├── data/annotated/
│   └── Human_Annotation_Workbook_800.xlsx
├── reports/                          # Reproducible benchmark CSVs, confusion matrices & domain-shift plots
├── requirements.txt
└── README.md