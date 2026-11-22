import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY in .env file.")

TARGET_COMMENTS = 30500       # Aim just above 30,000 for Assignment 3 & FYP
MAX_COMMENTS_PER_VIDEO = 600  # Mirrors Moller et al. (2024) 600-comment cap per video
MIN_VIDEO_COMMENTS = 250      # Ensures active review threads across all 6 domains

# Domain-general e-commerce product & service queries (10 queries = 1,000 quota units)
SEARCH_QUERIES = [
    ("Electronics", "smartphone honest review after 6 months"),
    ("Electronics", "wireless earbuds honest review don't buy"),
    ("Fashion_Apparel", "online clothing haul honest review quality"),
    ("Fashion_Apparel", "running shoes honest review pros and cons"),
    ("Home_Kitchen", "air fryer kitchen appliance honest review"),
    ("Home_Kitchen", "amazon home gadgets honest review worth it"),
    ("Beauty_Care", "skincare product honest review waste of money"),
    ("Digital_Services", "online shopping app scam delivery review"),
    ("Digital_Services", "software subscription service honest review worth it"),
    ("General_Ecommerce", "Daraz Amazon online shopping honest review"),
]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

youtube = build("youtube", "v3", developerKey=API_KEY)


def clean_for_excel(val):
    """Strip illegal XML chars, normalize newlines, and prevent Excel formula errors."""
    if pd.isna(val):
        return ""
    cleaned = ILLEGAL_CHARACTERS_RE.sub("", str(val))
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned).strip()
    if cleaned.startswith(("=", "+", "-", "@")):
        cleaned = "'" + cleaned
    return cleaned[:32000]


def discover_candidate_videos():
    """Step 1: Discover public product & service review videos via YouTube Data API v3."""
    candidates = {}
    print("Step 1/3: Discovering product & service review videos via YouTube Data API v3...")
    for domain, query in SEARCH_QUERIES:
        try:
            req = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                order="relevance",
                maxResults=25,
                relevanceLanguage="en",
            )
            res = req.execute()
            for item in res.get("items", []):
                vid = item["id"]["videoId"]
                if vid not in candidates:
                    candidates[vid] = {
                        "video_id": vid,
                        "domain": domain,
                        "search_query": query,
                    }
        except HttpError as e:
            print(f"Search warning for '{query}': {e}")
        time.sleep(0.15)

    print(f"Found {len(candidates)} unique candidate videos.")
    return candidates


def fetch_video_metadata(candidates):
    """Step 2: Fetch metadata, exclude kids/low-activity videos, and balance across domains."""
    print("Step 2/3: Fetching video metadata and balancing across e-commerce domains...")
    video_ids = list(candidates.keys())
    valid_videos = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        res = youtube.videos().list(
            part="snippet,statistics,status",
            id=",".join(batch),
        ).execute()

        for item in res.get("items", []):
            vid = item["id"]
            status = item.get("status", {})
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})

            if status.get("madeForKids", False):
                continue
            comment_count = int(stats.get("commentCount", 0))
            if comment_count < MIN_VIDEO_COMMENTS:
                continue

            valid_videos.append({
                "video_id": vid,
                "domain": candidates[vid]["domain"],
                "search_query": candidates[vid]["search_query"],
                "video_title": snippet.get("title", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "video_published_at": snippet.get("publishedAt", ""),
                "video_description": snippet.get("description", "")[:1000],
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "total_comment_count": comment_count,
            })

    df_meta = pd.DataFrame(valid_videos)
    if df_meta.empty:
        raise RuntimeError("No eligible videos found. Try lowering MIN_VIDEO_COMMENTS.")

    # Round-robin ordering across domains so no single vertical dominates the 30k sample
    df_meta = df_meta.sort_values("total_comment_count", ascending=False)
    df_meta["domain_rank"] = df_meta.groupby("domain").cumcount()
    df_meta = (
        df_meta.sort_values(["domain_rank", "total_comment_count"], ascending=[True, False])
        .drop(columns=["domain_rank"])
        .reset_index(drop=True)
    )

    print(
        f"Selected {len(df_meta)} eligible review videos across "
        f"{df_meta['domain'].nunique()} domains."
    )
    return df_meta


def crawl_comments(df_meta):
    """Step 3: Collect up to MAX_COMMENTS_PER_VIDEO top-level comments per video."""
    print(f"Step 3/3: Crawling top-level comments (target: {TARGET_COMMENTS:,})...")
    all_comments = []
    seen_comment_ids = set()
    crawled_video_ids = set()
    collected_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    pbar = tqdm(total=TARGET_COMMENTS, unit="comments")

    for _, row in df_meta.iterrows():
        if len(all_comments) >= TARGET_COMMENTS:
            break

        vid = row["video_id"]
        video_collected = 0
        next_page_token = None

        while video_collected < MAX_COMMENTS_PER_VIDEO and len(all_comments) < TARGET_COMMENTS:
            try:
                res = youtube.commentThreads().list(
                    part="snippet",
                    videoId=vid,
                    maxResults=100,
                    order="time",
                    textFormat="plainText",
                    pageToken=next_page_token,
                ).execute()
            except HttpError as e:
                if "commentsDisabled" in str(e):
                    break
                print(f"\nAPI notice on video {vid}: {e}")
                break

            items = res.get("items", [])
            if not items:
                break

            for item in items:
                cid = item["id"]
                if cid in seen_comment_ids:
                    continue
                seen_comment_ids.add(cid)

                top_snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = top_snippet.get("textDisplay", "").strip()
                if not text:
                    continue

                all_comments.append({
                    "comment_id": cid,
                    "video_id": vid,
                    "domain": row["domain"],
                    "video_title": row["video_title"],
                    "channel_title": row["channel_title"],
                    "comment_text": text,
                    "comment_published_at": top_snippet.get("publishedAt", ""),
                    "comment_like_count": int(top_snippet.get("likeCount", 0)),
                    "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
                    "char_length": len(text),
                    "word_count": len(text.split()),
                    "collected_at_utc": collected_at_utc,
                })
                video_collected += 1
                pbar.update(1)

                if video_collected >= MAX_COMMENTS_PER_VIDEO or len(all_comments) >= TARGET_COMMENTS:
                    break

            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(0.05)

        if video_collected > 0:
            crawled_video_ids.add(vid)

    pbar.close()
    df_comments = pd.DataFrame(all_comments)
    df_used_meta = df_meta[df_meta["video_id"].isin(crawled_video_ids)].copy()
    return df_used_meta, df_comments


def export_outputs(df_meta, df_comments):
    """Export raw CSVs for ML training + structured 3-sheet Excel workbook."""
    raw_csv_path = RAW_DIR / "youtube_ecommerce_comments_raw.csv"
    meta_csv_path = RAW_DIR / "youtube_video_metadata.csv"
    excel_path = PROCESSED_DIR / "YouTube_Ecommerce_Reviews_Dataset.xlsx"

    df_comments.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    df_meta.to_csv(meta_csv_path, index=False, encoding="utf-8-sig")

    domain_summary = (
        df_comments.groupby("domain")
        .agg(
            unique_videos=("video_id", "nunique"),
            total_comments=("comment_id", "count"),
            avg_word_count=("word_count", "mean"),
            median_likes=("comment_like_count", "median"),
        )
        .reset_index()
    )
    domain_summary["avg_word_count"] = domain_summary["avg_word_count"].round(2)

    df_meta_xl = df_meta.copy()
    for col in ["video_title", "channel_title", "video_description"]:
        df_meta_xl[col] = df_meta_xl[col].apply(clean_for_excel)

    df_comments_xl = df_comments.copy()
    for col in ["video_title", "channel_title", "comment_text"]:
        df_comments_xl[col] = df_comments_xl[col].apply(clean_for_excel)

    print(f"\nWriting structured Excel workbook to {excel_path} (takes ~20-40 seconds)...")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        domain_summary.to_excel(writer, sheet_name="Dataset_Summary", index=False)
        df_meta_xl.to_excel(writer, sheet_name="Video_Metadata", index=False)
        df_comments_xl.to_excel(writer, sheet_name="Collected_Comments", index=False)

    print("\n================ CRAWL COMPLETE ================")
    print(f"Total Videos Crawled   : {len(df_meta)}")
    print(f"Total Comments Saved   : {len(df_comments):,}")
    print(f"Raw CSV (for Python)   : {raw_csv_path}")
    print(f"Excel Deliverable      : {excel_path}")
    print("\nBreakdown by E-Commerce Domain:")
    print(domain_summary.to_string(index=False))


if __name__ == "__main__":
    candidates = discover_candidate_videos()
    df_meta = fetch_video_metadata(candidates)
    df_used_meta, df_comments = crawl_comments(df_meta)
    export_outputs(df_used_meta, df_comments)