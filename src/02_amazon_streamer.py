import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CATEGORIES = [
    "Electronics",
    "Amazon_Fashion",
    "Home_and_Kitchen",
    "All_Beauty",
    "Software",
]
PER_STAR_LIMIT = 1200          # 1,200 x 5 star levels = 6,000/category -> 30,000 total
MAX_SCAN_PER_CATEGORY = 150000 # Safety ceiling per category

BASE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/{category}.jsonl"
)

RAW_DIR = Path("data/raw")
CHECKPOINT_DIR = RAW_DIR / "amazon_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def stream_category(category):
    """Stream JSONL line-by-line over HTTPS and stop immediately once balanced quota is met."""
    ckpt_path = CHECKPOINT_DIR / f"{category}.csv"
    if ckpt_path.exists():
        print(f"\n[Checkpoint found] Loading cached {category} from {ckpt_path}")
        return pd.read_csv(ckpt_path)

    url = BASE_URL.format(category=category)
    star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    target_total = PER_STAR_LIMIT * 5
    rows = []

    print(f"\nStreaming {category} (target: {target_total:,} rating-balanced reviews)...")
    pbar = tqdm(total=target_total, unit="reviews")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for idx, raw_line in enumerate(resp.iter_lines(decode_unicode=True)):
            if idx >= MAX_SCAN_PER_CATEGORY:
                break
            if all(c >= PER_STAR_LIMIT for c in star_counts.values()):
                break
            if not raw_line:
                continue

            try:
                r = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            rating = r.get("rating")
            text = (r.get("text") or "").strip()
            title = (r.get("title") or "").strip()

            if rating is None or not text:
                continue

            star_int = int(round(float(rating)))
            if star_int not in star_counts or star_counts[star_int] >= PER_STAR_LIMIT:
                continue

            rows.append({
                "category": category,
                "rating": star_int,
                "review_title": title,
                "review_text": text,
                "asin": r.get("asin", ""),
                "parent_asin": r.get("parent_asin", ""),
                "user_id": r.get("user_id", ""),
                "timestamp_ms": r.get("timestamp", None),
                "helpful_vote": int(r.get("helpful_vote", 0)),
                "verified_purchase": bool(r.get("verified_purchase", False)),
                "word_count": len(text.split()),
            })
            star_counts[star_int] += 1
            pbar.update(1)

    pbar.close()
    df_cat = pd.DataFrame(rows)
    df_cat.to_csv(ckpt_path, index=False, encoding="utf-8-sig")
    print(f"Saved {category}: {len(df_cat):,} reviews | Star counts: {star_counts}")
    return df_cat


if __name__ == "__main__":
    dfs = [stream_category(cat) for cat in CATEGORIES]
    df_amazon = pd.concat(dfs, ignore_index=True)

    out_path = RAW_DIR / "amazon_2023_stratified_30k.csv"
    df_amazon.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n================ AMAZON STREAMING COMPLETE ================")
    print(f"Total Storefront Reviews Saved : {len(df_amazon):,}")
    print(f"Unique Products (parent_asin)  : {df_amazon['parent_asin'].nunique():,}")
    print(f"Unique Reviewers (user_id)     : {df_amazon['user_id'].nunique():,}")
    print(f"Verified Purchase Rate         : {df_amazon['verified_purchase'].mean() * 100:.2f}%")
    print(f"Output File                    : {out_path}")
    print("\nBreakdown by Category & Star Rating:")
    print(pd.crosstab(df_amazon["category"], df_amazon["rating"]))