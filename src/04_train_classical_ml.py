import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/tier1_classical")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Mixed_Neutral", "Positive"]
RANDOM_SEED = 42


def load_splits():
    df_train = pd.read_csv(PROCESSED_DIR / "amazon_train.csv")
    df_val = pd.read_csv(PROCESSED_DIR / "amazon_val.csv")
    df_test = pd.read_csv(PROCESSED_DIR / "amazon_test.csv")

    # Safety assertion: ensure zero product leakage before training
    assert len(set(df_train["parent_asin"]) & set(df_test["parent_asin"])) == 0
    return df_train, df_val, df_test


def build_models():
    """Exact 2 vectorizers x 5 classifiers benchmark modeled on Moller et al. (2024)."""
    vectorizers = {
        "Count": lambda: CountVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            min_df=3,
            lowercase=True,
        ),
        "TF-IDF": lambda: TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            min_df=3,
            sublinear_tf=True,
            lowercase=True,
        ),
    }

    classifiers = {
        "Multinomial_NB": lambda: MultinomialNB(alpha=0.5),
        "Logistic_Regression": lambda: LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Linear_SVM": lambda: LinearSVC(
            C=0.5,
            class_weight="balanced",
            max_iter=20000,
            tol=1e-3,
            random_state=RANDOM_SEED,
        ),
        "Decision_Tree": lambda: DecisionTreeClassifier(
            max_depth=35,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "KNN_Cosine": lambda: KNeighborsClassifier(
            n_neighbors=11,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ),
    }
    return vectorizers, classifiers


def evaluate_predictions(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro_Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Macro_Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Weighted_F1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def save_confusion_matrix_plot(y_true, y_pred, model_title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=axes[0]
    )
    axes[0].set_title(f"{model_title}\nRaw Counts (Test n={len(y_true):,})")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Greens",
        xticklabels=LABELS, yticklabels=LABELS, ax=axes[1]
    )
    axes[1].set_title(f"{model_title}\nRow-Normalized Recall")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def extract_top_features(pipe, save_csv_path, top_k=15):
    """Extract top positive feature weights per class (interpretable alternative to eli5)."""
    vec = pipe.named_steps["vec"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return

    feature_names = np.array(vec.get_feature_names_out())
    rows = []
    for idx, cls_name in enumerate(clf.classes_):
        weights = clf.coef_[idx]
        top_idx = np.argsort(weights)[-top_k:][::-1]
        for rank, fi in enumerate(top_idx, start=1):
            rows.append({
                "class": cls_name,
                "rank": rank,
                "feature_ngram": feature_names[fi],
                "weight": round(float(weights[fi]), 4),
            })

    df_feat = pd.DataFrame(rows)
    df_feat.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTop 8 Learned Features per Class ({pipe.named_steps['clf'].__class__.__name__}):")
    for cls_name in LABELS:
        sub = df_feat[df_feat["class"] == cls_name].head(8)
        terms = ", ".join(f"{r.feature_ngram} ({r.weight:+.2f})" for r in sub.itertuples())
        print(f"  {cls_name:14s}: {terms}")


def run_benchmark():
    df_train, df_val, df_test = load_splits()
    X_train, y_train = df_train["full_text"].astype(str), df_train["sentiment_3class"]
    X_val, y_val = df_val["full_text"].astype(str), df_val["sentiment_3class"]
    X_test, y_test = df_test["full_text"].astype(str), df_test["sentiment_3class"]

    results = []
    best_val_f1 = -1.0
    best_name = None
    best_pipe = None

    print(f"Loaded leakage-free splits: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")
    print("\nTraining Baseline + 10 Classical ML Pipelines (2 Vectorizers x 5 Classifiers)...")

    # 0. Naive Majority-Class Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_val = evaluate_predictions(y_val, dummy.predict(X_val))
    dummy_test = evaluate_predictions(y_test, dummy.predict(X_test))
    results.append({
        "Model": "Majority_Baseline",
        "Vectorizer": "None",
        "Val_Macro_F1": dummy_val["Macro_F1"],
        "Test_Accuracy": dummy_test["Accuracy"],
        "Test_Macro_Precision": dummy_test["Macro_Precision"],
        "Test_Macro_Recall": dummy_test["Macro_Recall"],
        "Test_Macro_F1": dummy_test["Macro_F1"],
        "Test_Weighted_F1": dummy_test["Weighted_F1"],
        "Train_Sec": 0.0,
    })
    print(f"  [Baseline] Majority Class          -> Val Macro-F1: {dummy_val['Macro_F1']:.4f} | Test Macro-F1: {dummy_test['Macro_F1']:.4f}")

    # 1. Full 2 x 5 Grid
    vectorizers, classifiers = build_models()
    for vec_name, vec_fn in vectorizers.items():
        for clf_name, clf_fn in classifiers.items():
            t0 = time.perf_counter()
            pipe = Pipeline([
                ("vec", vec_fn()),
                ("clf", clf_fn()),
            ])
            pipe.fit(X_train, y_train)
            elapsed = round(time.perf_counter() - t0, 2)

            val_metrics = evaluate_predictions(y_val, pipe.predict(X_val))
            test_metrics = evaluate_predictions(y_test, pipe.predict(X_test))

            row_name = f"{clf_name} ({vec_name})"
            results.append({
                "Model": clf_name,
                "Vectorizer": vec_name,
                "Val_Macro_F1": val_metrics["Macro_F1"],
                "Test_Accuracy": test_metrics["Accuracy"],
                "Test_Macro_Precision": test_metrics["Macro_Precision"],
                "Test_Macro_Recall": test_metrics["Macro_Recall"],
                "Test_Macro_F1": test_metrics["Macro_F1"],
                "Test_Weighted_F1": test_metrics["Weighted_F1"],
                "Train_Sec": elapsed,
            })

            print(
                f"  {row_name:34s} -> Val Macro-F1: {val_metrics['Macro_F1']:.4f} | "
                f"Test Acc: {test_metrics['Accuracy']:.4f} | Test Macro-F1: {test_metrics['Macro_F1']:.4f} ({elapsed}s)"
            )

            # Select best model strictly on Validation Macro-F1 (never peek at Test!)
            if val_metrics["Macro_F1"] > best_val_f1:
                best_val_f1 = val_metrics["Macro_F1"]
                best_name = row_name
                best_pipe = pipe

    # Save full benchmark table
    df_results = pd.DataFrame(results).sort_values("Val_Macro_F1", ascending=False)
    df_results.to_csv(REPORTS_DIR / "tier1_benchmark_results.csv", index=False)

    # 5-Fold Product-Grouped Cross-Validation on Best Model
    print(f"\nRunning 5-Fold Product-Grouped Cross-Validation on Best Model: {best_name}...")
    best_clf_key = df_results.iloc[0]["Model"]
    best_vec_key = df_results.iloc[0]["Vectorizer"]
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    groups = df_train["parent_asin"].values
    for fold, (tr_i, va_i) in enumerate(gkf.split(X_train, y_train, groups=groups), start=1):
        cv_pipe = Pipeline([("vec", vectorizers[best_vec_key]()), ("clf", classifiers[best_clf_key]())])
        cv_pipe.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
        fold_f1 = f1_score(y_train.iloc[va_i], cv_pipe.predict(X_train.iloc[va_i]), average="macro")
        cv_scores.append(fold_f1)
    print(f"  5-Fold GroupKFold Macro-F1: Mean = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final Held-Out Test Report & Confusion Matrix for Best Model
    y_test_pred = best_pipe.predict(X_test)
    print(f"\n================ BEST TIER-1 MODEL: {best_name} ================")
    print(classification_report(y_test, y_test_pred, labels=LABELS, digits=4))

    # Save artifacts
    model_path = MODELS_DIR / "tier1_best_classical.joblib"
    cm_path = REPORTS_DIR / "tier1_best_confusion_matrix.png"
    feat_path = REPORTS_DIR / "tier1_top_features.csv"

    joblib.dump(best_pipe, model_path)
    save_confusion_matrix_plot(y_test, y_test_pred, best_name, cm_path)
    extract_top_features(best_pipe, feat_path)

    # Save representative error cases for manual error analysis in paper/viva
    df_err = df_test.copy()
    df_err["predicted_label"] = y_test_pred
    df_err = df_err[df_err["sentiment_3class"] != df_err["predicted_label"]][
        ["category", "rating", "sentiment_3class", "predicted_label", "full_text"]
    ].head(100)
    df_err.to_csv(REPORTS_DIR / "tier1_error_samples.csv", index=False, encoding="utf-8-sig")

    print("\nSaved Artifacts:")
    print(f"  Benchmark CSV     : {REPORTS_DIR / 'tier1_benchmark_results.csv'}")
    print(f"  Confusion Matrix  : {cm_path}")
    print(f"  Top Features CSV  : {feat_path}")
    print(f"  Error Samples CSV : {REPORTS_DIR / 'tier1_error_samples.csv'}")
    print(f"  Trained Model     : {model_path}")


if __name__ == "__main__":
    run_benchmark()