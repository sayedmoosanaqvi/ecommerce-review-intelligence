import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ANNOT_DIR = Path("data/annotated")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

STAR_TITLE_RE = re.compile(
    r"^\s*(one|two|three|four|five|1|2|3|4|5)\s+stars?[\s!.\-]*$",
    flags=re.IGNORECASE,
)


def normalize_whitespace(text):
    """Preserve negation, punctuation, emojis, and Roman Urdu; only collapse broken whitespace."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_default_star_title(title):
    """Remove Amazon's auto-generated star-rating titles (e.g., 'Five Stars') to prevent target leakage."""
    clean = normalize_whitespace(title)
    if STAR_TITLE_RE.match(clean):
        return ""
    return clean


def map_star_to_3class(star):
    """3-class polarity benchmark: 0=Negative (1-2★), 1=Mixed/Neutral (3★), 2=Positive (4-5★)."""
    if star <= 2:
        return "Negative"
    elif star == 3:
        return "Mixed_Neutral"
    else:
        return "Positive"


def audit_and_split_amazon():
    print("1/3: Auditing Amazon Storefront Corpus & creating product-grouped splits...")
    df = pd.read_csv(RAW_DIR / "amazon_2023_stratified_30k.csv")
    initial_n = len(df)

    # Strip auto-generated star titles to prevent target leakage
    raw_titles = df["review_title"].apply(normalize_whitespace)
    df["review_title"] = raw_titles.apply(strip_default_star_title)
    n_stripped = int((raw_titles != df["review_title"]).sum())
    print(f"   Leakage audit: stripped {n_stripped:,} auto-generated star titles (e.g., 'Five Stars').")

    df["review_text"] = df["review_text"].apply(normalize_whitespace)
    df["full_text"] = np.where(
        df["review_title"] != "",
        df["review_title"] + ". " + df["review_text"],
        df["review_text"],
    )

    # Drop empty texts and exact (parent_asin, full_text) duplicates
    df = df[df["full_text"].str.len() >= 3].copy()
    df = df.drop_duplicates(subset=["parent_asin", "full_text"]).reset_index(drop=True)
    df["sentiment_3class"] = df["rating"].apply(map_star_to_3class)
    df["word_count_clean"] = df["full_text"].str.split().str.len()

    # Leakage-free 70 / 15 / 15 split grouped strictly by parent_asin (product ID)
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss_outer.split(df, df["sentiment_3class"], groups=df["parent_asin"]))

    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_rel_idx, test_rel_idx = next(
        gss_inner.split(df_temp, df_temp["sentiment_3class"], groups=df_temp["parent_asin"])
    )
    df_val = df_temp.iloc[val_rel_idx].copy()
    df_test = df_temp.iloc[test_rel_idx].copy()

    # Hard assertion: ZERO product leakage across splits
    train_prods = set(df_train["parent_asin"])
    val_prods = set(df_val["parent_asin"])
    test_prods = set(df_test["parent_asin"])
    assert len(train_prods & val_prods) == 0, "Leakage detected between train and val!"
    assert len(train_prods & test_prods) == 0, "Leakage detected between train and test!"
    assert len(val_prods & test_prods) == 0, "Leakage detected between val and test!"

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_train.to_csv(PROCESSED_DIR / "amazon_train.csv", index=False, encoding="utf-8-sig")
    df_val.to_csv(PROCESSED_DIR / "amazon_val.csv", index=False, encoding="utf-8-sig")
    df_test.to_csv(PROCESSED_DIR / "amazon_test.csv", index=False, encoding="utf-8-sig")

    print(f"   Raw rows: {initial_n:,} -> Clean unique rows: {len(df):,}")
    print(f"   Leakage-free splits (by parent_asin): Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")
    print("   Zero product overlap verified: PASS")
    print(f"   Amazon median/mean words: {df['word_count_clean'].median():.1f} / {df['word_count_clean'].mean():.1f}")
    return df, df_test


def audit_youtube():
    print("\n2/3: Auditing YouTube Social-Commerce Corpus...")
    df = pd.read_csv(RAW_DIR / "youtube_ecommerce_comments_raw.csv")
    initial_n = len(df)

    df["clean_text"] = df["comment_text"].apply(normalize_whitespace)
    df = df[df["clean_text"].str.len() >= 2].copy()
    df = df.drop_duplicates(subset=["video_id", "clean_text"]).reset_index(drop=True)
    df["word_count_clean"] = df["clean_text"].str.split().str.len()
    df["has_url"] = df["clean_text"].str.contains(r"http[s]?://|www\.", case=False, regex=True)
    df["has_non_ascii"] = df["clean_text"].apply(lambda s: any(ord(c) > 127 for c in s))

    df.to_csv(PROCESSED_DIR / "youtube_clean.csv", index=False, encoding="utf-8-sig")

    print(f"   Raw rows: {initial_n:,} -> Clean unique rows: {len(df):,}")
    print(f"   Unique videos: {df['video_id'].nunique()} | Domains: {df['domain'].nunique()}")
    print(f"   YouTube median/mean words: {df['word_count_clean'].median():.1f} / {df['word_count_clean'].mean():.1f}")
    print(f"   Comments with URLs: {df['has_url'].sum():,} ({df['has_url'].mean()*100:.2f}%)")
    print(f"   Comments with emojis/non-ASCII: {df['has_non_ascii'].sum():,} ({df['has_non_ascii'].mean()*100:.2f}%)")
    return df


def build_human_annotation_workbook(df_yt, df_amz_test):
    """Create stratified 800-row human coding workbook (600 YouTube + 200 Amazon held-out test)."""
    print("\n3/3: Generating Human Annotation Workbook (with 120-row Krippendorff's Alpha overlap)...")

    yt_samples = []
    for dom, grp in df_yt.groupby("domain"):
        n_take = min(100, len(grp))
        yt_samples.append(grp.sample(n=n_take, random_state=RANDOM_SEED))
    df_yt_sample = pd.concat(yt_samples, ignore_index=True)

    amz_samples = []
    for cat, grp in df_amz_test.groupby("category"):
        n_take = min(40, len(grp))
        amz_samples.append(grp.sample(n=n_take, random_state=RANDOM_SEED))
    df_amz_sample = pd.concat(amz_samples, ignore_index=True)

    yt_std = pd.DataFrame({
        "sample_id": [f"YT_{i+1:04d}" for i in range(len(df_yt_sample))],
        "platform": "YouTube",
        "domain_or_category": df_yt_sample["domain"].values,
        "group_id": df_yt_sample["video_id"].values,
        "context_title": df_yt_sample["video_title"].values,
        "text_to_code": df_yt_sample["clean_text"].values,
    })

    amz_std = pd.DataFrame({
        "sample_id": [f"AMZ_{i+1:04d}" for i in range(len(df_amz_sample))],
        "platform": "Amazon",
        "domain_or_category": df_amz_sample["category"].values,
        "group_id": df_amz_sample["parent_asin"].values,
        "context_title": "Storefront Product Review",
        "text_to_code": df_amz_sample["full_text"].values,
    })

    df_annot = pd.concat([yt_std, amz_std], ignore_index=True)
    df_annot = df_annot.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    df_annot["is_reliability_overlap"] = False
    df_annot.loc[:119, "is_reliability_overlap"] = True

    for col in [
        "language (EN/RU/OTHER)",
        "is_relevant (1/0)",
        "sentiment (NEG/MIX/POS/NA)",
        "issue_defect_IIO04 (1/0)",
        "issue_misleading_IIO05 (1/0)",
        "issue_service_delivery_IIO06_07 (1/0)",
        "intent_refund_return (1/0)",
        "suspicion_fake_spam_IIO01 (1/0)",
        "coder_notes",
    ]:
        df_annot[col] = ""

    codebook_rows = [
        ("language", "EN = English; RU = Roman Urdu / code-mixed Urdu; OTHER = other language"),
        ("is_relevant", "1 = Reflects real product/service opinion, usage experience, question, or shopping issue; 0 = Spam, channel praise only ('great video!'), timestamp joke, or off-topic"),
        ("sentiment", "NEG = Clearly negative; MIX = Both pros and cons OR neutral/question; POS = Clearly positive; NA = Irrelevant"),
        ("issue_defect_IIO04", "1 = Reports broken part, malfunction, poor build quality, stopped working; 0 = Otherwise"),
        ("issue_misleading_IIO05", "1 = Product differs from listing/ad/hype ('not as described', wrong size/spec); 0 = Otherwise"),
        ("issue_service_delivery_IIO06_07", "1 = Late parcel, damaged shipping, poor seller support, ignored complaint, chatbot failure; 0 = Otherwise"),
        ("intent_refund_return", "1 = Explicit customer intent to return, replace, cancel, or request a refund; 0 = Otherwise"),
        ("suspicion_fake_spam_IIO01", "1 = Comment alleges fake/paid reviews OR is itself promotional spam; 0 = Otherwise"),
    ]
    df_codebook = pd.DataFrame(codebook_rows, columns=["Column", "Coding_Rule"])

    out_xlsx = ANNOT_DIR / "Human_Annotation_Workbook_800.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_codebook.to_excel(writer, sheet_name="Codebook_Rules", index=False)
        df_annot.to_excel(writer, sheet_name="Annotation_Sheet_800", index=False)
        df_annot[df_annot["is_reliability_overlap"]].to_excel(
            writer, sheet_name="Dual_Coder_Overlap_120", index=False
        )

    key_df = pd.DataFrame({
        "sample_id": amz_std["sample_id"],
        "true_star_rating": df_amz_sample["rating"].values,
        "true_sentiment_3class": df_amz_sample["sentiment_3class"].values,
        "verified_purchase": df_amz_sample["verified_purchase"].values,
    })
    key_df.to_csv(ANNOT_DIR / "amazon_200_blind_key.csv", index=False)

    print(f"   Saved Human Annotation Workbook: {out_xlsx}")
    print(f"   Total blind items: {len(df_annot)} (YouTube={len(yt_std)}, Amazon Held-Out={len(amz_std)}) | Dual-coder overlap: 120")


if __name__ == "__main__":
    df_amz, df_amz_test = audit_and_split_amazon()
    df_yt = audit_youtube()
    build_human_annotation_workbook(df_yt, df_amz_test)
    print("\n================ STEP 6 COMPLETE ================")