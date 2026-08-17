"""Step 9: fetch 4chan posts and regex-prefilter them into candidates.

4chan has the highest profanity density of any public forum, so it is a fast
source for the attack / emotional banks (a few boards = tens of thousands of
candidates in a minute, no API key needed).

Two modes:

    # live: snapshot every currently-open thread on the given boards via the
    # official (public, keyless) a.4cdn.org API
    .venv/bin/python scripts/09_4chan_fetch.py --live \
        --boards b,r9k,pol,v,gaming,trash --max_rows 500000 --out data/4chan/raw

    # archive: bulk historical search via the 4plebs JSON API (optional,
    # needs no key; 4plebs only archives some boards)
    .venv/bin/python scripts/09_4chan_fetch.py --archive \
        --board pol --search "fuck you" --max_rows 200000 --out data/4chan/raw

Output: same candidate schema as the Reddit fetcher (data/4chan/raw/*.parquet)
with `source="4chan"` and `subreddit=<board>`.
"""

from __future__ import annotations

import argparse
import random
import time

import pandas as pd
import requests
from tqdm import tqdm

from candidate_common import Sink, _is_latin, clean_4chan_html, prefilter_keep

LIVE_BASE = "https://a.4cdn.org"
ARCHIVE_BASE = "https://archive.4plebs.org"

# profanity / flame-war prone boards (live API keeps the ~1-week window)
DEFAULT_BOARDS = ["b", "pol", "r9k", "v", "gaming", "trash", "adv", "soc"]

# boards the 4plebs archive actually mirrors
ARCHIVE_BOARDS = ["adv", "b", "hr", "o", "pol", "s4s", "tg", "trv", "x", "y"]


def _row(post: dict, board: str, source_score: int, com_field: str = "com") -> dict | None:
    body = clean_4chan_html(post.get(com_field))
    if not body or body in {"[deleted]", "[ removed ]"}:
        return None
    return {
        "text": body,
        "source": "4chan",
        "subreddit": board,
        "created_utc": int(post.get("time") or 0),
        "score": int(source_score),
        "id": str(post.get("no") or ""),
    }


def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60) -> dict | None:
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(2 * (attempt + 1))
    print(f"    giving up on {url}: {last_err}")
    return None


# --------------------------------------------------------------------------- #
# live mode: catalog -> threads -> posts
# --------------------------------------------------------------------------- #
def fetch_live(args, sink: Sink, rng) -> None:
    boards = (
        [b.strip() for b in args.boards.split(",") if b.strip()]
        if args.boards
        else DEFAULT_BOARDS
    )
    run = 0
    while True:
        if run > 0:
            print(f"\n=== re-scan #{run} (waited {args.repeat_delay}s) ===")
        for board in boards:
            print(f"=== /{board}/ live catalog ===")
            catalog = _get(f"{LIVE_BASE}/{board}/catalog.json")
            if not catalog:
                continue
            thread_ids = [
                page["threads"][i]["no"]
                for page in catalog
                for i in range(len(page["threads"]))
            ]
            thread_ids = thread_ids[: args.threads_per_board]
            if len(thread_ids) < args.threads_per_board:
                print(f"  note: only {len(thread_ids):,} threads found (live mode "
                      f"only has currently-open threads; use --archive or --repeat "
                      f"to accumulate more)")
            print(f"  {len(thread_ids):,} threads")
            with tqdm(thread_ids, desc=f"/{board}/", unit="thread", leave=False) as pbar:
                for t in pbar:
                    if sink.kept >= args.max_rows:
                        print("  reached --max_rows, stopping")
                        return
                    data = _get(f"{LIVE_BASE}/{board}/thread/{t}.json")
                    if not data or not data.get("posts"):
                        continue
                    n_thread = 0
                    for post in data["posts"]:
                        score = post.get("replies", 0) if post.get("op") else 0
                        row = _row(post, board, score)
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
                            n_thread += 1
                    pbar.set_postfix(kept=sink.kept, this=n_thread)
                    time.sleep(args.delay)
            print(f"  /{board}/ done (total kept so far: {sink.kept:,})")
        run += 1
        if run >= args.repeat or sink.kept >= args.max_rows:
            break
        print(f"  sleeping {args.repeat_delay}s before re-scan #{run + 1} ...")
        time.sleep(args.repeat_delay)


# --------------------------------------------------------------------------- #
# archive mode: 4plebs JSON API
# --------------------------------------------------------------------------- #
def fetch_archive(args, sink: Sink, rng) -> None:
    board = args.board or "pol"
    params = {
        "board": board,
        "json": "1",
        "per_page": str(args.per_page),
        "order": "desc",
    }
    if args.search:
        params["search"] = args.search
    page = 1
    pbar = tqdm(total=args.max_pages, desc=f"/{board}/ archive", unit="page", leave=False)
    while True:
        if sink.kept >= args.max_rows:
            print("  reached --max_rows, stopping")
            break
        params["page"] = str(page)
        data = _get(f"{ARCHIVE_BASE}/_/api/cc/board/post/search/", params=params)
        if not data or not data.get("data"):
            print(f"  no more results at page {page}")
            return
        posts = data["data"]["posts"]
        if not posts:
            return
        n_kept_before = sink.kept
        for post in posts:
            row = _row(post, board, 0, com_field="comment")
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
        print(f"  page {page}: {sink.kept - n_kept_before:,} kept (total {sink.kept:,})")
        pbar.update(1)
        pbar.set_postfix(kept=sink.kept)
        page += 1
        if page > args.max_pages:
            print(f"  reached {args.max_pages} pages, stopping")
            break
        time.sleep(args.api_delay)
    pbar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="official a.4cdn.org live API")
    ap.add_argument("--archive", action="store_true", help="4plebs archive JSON API")
    ap.add_argument("--boards", default=None,
                    help=f"comma-separated boards (default: {','.join(DEFAULT_BOARDS)})")
    ap.add_argument("--board", default=None, help="single board for --archive")
    ap.add_argument("--search", default=None, help="body search term for --archive")
    ap.add_argument("--threads_per_board", type=int, default=250)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between thread fetches")
    ap.add_argument("--repeat", type=int, default=0,
                    help="re-scan the catalog N times after finishing (0 = off). "
                         "Use with --repeat_delay to keep harvesting new threads over time.")
    ap.add_argument("--repeat_delay", type=float, default=300.0,
                    help="seconds to wait between catalog re-scans when --repeat > 0")
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--clean_sample_frac", type=float, default=0.05,
                    help="fraction of regex-clean comments kept")
    ap.add_argument("--require_latin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_rows", type=int, default=2_000_000)
    ap.add_argument("--per_page", type=int, default=100, help="archive page size")
    ap.add_argument("--max_pages", type=int, default=200)
    ap.add_argument("--api_delay", type=float, default=0.4, help="seconds between archive pages")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/4chan/raw")
    ap.add_argument("--flush_rows", type=int, default=500_000)
    args = ap.parse_args()

    from pathlib import Path

    out_dir = Path(args.out)
    sink = Sink(out_dir, flush_rows=args.flush_rows)
    rng = random.Random(args.seed)

    if args.live:
        fetch_live(args, sink, rng)
    elif args.archive:
        if args.board and args.board not in ARCHIVE_BOARDS:
            print(f"note: 4plebs does not archive /{args.board}/; "
                  f"archived boards: {', '.join(ARCHIVE_BOARDS)}")
        fetch_archive(args, sink, rng)
    else:
        ap.error("specify --live or --archive")

    sink.close()
    print(f"\nkept {sink.kept:,} rows | dropped {sink.dropped_dup:,} exact dups | shards={sink.shard}")
    if sink.kept:
        df = pd.read_parquet(out_dir)
        print(df.category.value_counts().to_string())
        print(df.groupby(["subreddit", "category"]).size().unstack(fill_value=0).to_string())
        print(f"  -> {len(df):,} rows in {out_dir}")


if __name__ == "__main__":
    main()