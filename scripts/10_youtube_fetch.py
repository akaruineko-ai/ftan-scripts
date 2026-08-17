"""Step 10: fetch YouTube comments and regex-prefilter them into candidates.

YouTube comment sections (cat videos, cooking, gaming let's-plays, ...) are a
relaxed, casual register where people almost never swear at each other, so
they are a clean-filler source for the clean bank (bank_c) — the direct
counterweight to the flame-heavy Reddit / 4chan data.

Requires a YouTube Data API v3 key (free tier: 10k units/day):

    .venv/bin/python scripts/10_youtube_fetch.py \
        --api_key AIza... --query "cat videos" \
        --max_videos 20 --max_comments_per_video 200 --out data/youtube/raw

Because the whole point is *clean* filler, every clean comment is kept by
default (--clean_sample_frac 1.0) — lower it to sample.

Output: same candidate schema as the other fetchers (data/youtube/raw/*.parquet)
with `source="youtube"` and `subreddit=<video title> [videoId]`.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

from candidate_common import Sink, _is_latin, prefilter_keep

API = "https://www.googleapis.com/youtube/v3"
# casual / kid-friendly channels where comments stay clean
DEFAULT_QUERY = "cat videos"

REMOVED = {"[removed]", "[deleted]", "", "undefined"}


def _row_from_comment(c: dict, video_title: str, video_id: str) -> dict | None:
    snip = (c.get("snippet") or {}).get("topLevelComment", {}).get("snippet") or {}
    body = (snip.get("textDisplay") or "").strip()
    if not body or body in REMOVED:
        return None
    published = snip.get("publishedAt") or ""
    try:
        ts = int(datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        ts = 0
    return {
        "text": body,
        "source": "youtube",
        "subreddit": f"{video_title} [{video_id}]",
        "created_utc": ts,
        "score": int(snip.get("likeCount") or 0),
        "id": str((c.get("snippet") or {}).get("topLevelComment", {}).get("id") or ""),
    }


def _api_get(path: str, params: dict, api_key: str, retries: int) -> dict | None:
    params = {**params, "key": api_key}
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
            err = r.json().get("error", {}).get("message")
            if err:
                last_err += f": {err}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(3 * (attempt + 1))
    print(f"    giving up on {path}: {last_err}")
    return None


def _search_videos(args) -> list[tuple[str, str]]:
    """Return (video_id, title) list for the query, oldest-first."""
    out = []
    token = None
    with tqdm(desc="searching", unit="page", leave=False) as pbar:
        while True:
            params = {
                "part": "id,snippet",
                "q": args.query,
                "type": "video",
                "maxResults": "50",
                "relevanceLanguage": "en",
                "videoEmbeddable": "true",
            }
            if args.published_before:
                params["publishedBefore"] = args.published_before
            if token:
                params["pageToken"] = token
            data = _api_get("search", params, args.api_key, args.retries)
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                vid = (item.get("id") or {}).get("videoId")
                if not vid:
                    continue
                out.append((vid, (item.get("snippet") or {}).get("title") or vid))
                if len(out) >= args.max_videos:
                    pbar.update(1)
                    pbar.set_postfix(videos=len(out))
                    return out
            token = data.get("nextPageToken")
            if not token:
                break
            pbar.update(1)
            pbar.set_postfix(videos=len(out))
            time.sleep(args.api_delay)
    return out


def fetch_youtube(args, sink: Sink, rng) -> None:
    videos = _search_videos(args)
    print(f"found {len(videos):,} videos for {args.query!r}")
    with tqdm(videos, desc="videos", unit="video", leave=False) as pbar:
        for vid, title in pbar:
            if sink.kept >= args.max_rows:
                print("  reached --max_rows, stopping")
                return
            token = None
            per_video = 0
            with tqdm(desc="comments", unit="comment", leave=False) as cpbar:
                while True:
                    if per_video >= args.max_comments_per_video:
                        break
                    params = {
                        "part": "snippet",
                        "videoId": vid,
                        "maxResults": "100",
                        "textFormat": "plainText",
                        "order": "relevance",
                    }
                    if token:
                        params["pageToken"] = token
                    data = _api_get("commentThreads", params, args.api_key, args.retries)
                    if not data or not data.get("items"):
                        break
                    for c in data["items"]:
                        row = _row_from_comment(c, title, vid)
                        if row is None:
                            continue
                        if len(row["text"].split()) < args.min_words:
                            continue
                        if args.require_latin and not _is_latin(row["text"]):
                            continue
                        info, keep = prefilter_keep(row["text"], args.clean_sample_frac, rng)
                        if keep:
                            row.update({
                                "category": info["category"],
                                "profanity": int(info["profanity"]),
                                "insult": int(info["insult"]),
                                "attack": int(info["attack"]),
                            })
                            sink.add(row)
                            per_video += 1
                        if per_video >= args.max_comments_per_video or sink.kept >= args.max_rows:
                            break
                    token = data.get("nextPageToken")
                    if not token:
                        break
                    cpbar.update(1)
                    cpbar.set_postfix(kept=sink.kept)
                    time.sleep(args.api_delay)
            pbar.set_postfix(kept=sink.kept)
            time.sleep(args.api_delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=None, help="YouTube Data API v3 key")
    ap.add_argument("--query", default=DEFAULT_QUERY, help="search query for videos")
    ap.add_argument("--published_before", default=None,
                    help="only videos published before this RFC3339 datetime")
    ap.add_argument("--max_videos", type=int, default=20)
    ap.add_argument("--max_comments_per_video", type=int, default=200)
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--clean_sample_frac", type=float, default=1.0,
                    help="fraction of regex-clean comments kept (YouTube is a clean source)")
    ap.add_argument("--require_latin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_rows", type=int, default=1_000_000)
    ap.add_argument("--api_delay", type=float, default=0.4, help="seconds between API calls")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/youtube/raw")
    ap.add_argument("--flush_rows", type=int, default=500_000)
    args = ap.parse_args()

    if not args.api_key:
        ap.error("--api_key is required (YouTube Data API v3, free 10k units/day)")

    from pathlib import Path

    out_dir = Path(args.out)
    sink = Sink(out_dir, flush_rows=args.flush_rows)
    rng = random.Random(args.seed)

    fetch_youtube(args, sink, rng)

    sink.close()
    print(f"\nkept {sink.kept:,} rows | dropped {sink.dropped_dup:,} exact dups | shards={sink.shard}")
    if sink.kept:
        df = pd.read_parquet(out_dir)
        print(df.category.value_counts().to_string())
        print(f"  -> {len(df):,} rows in {out_dir}")


if __name__ == "__main__":
    main()