"""Step 7: fetch Reddit comments and regex-prefilter them into candidates.

Data comes from the Arctic Shift API (the Pushshift successor) or from a
monthly dump file, either local or over HTTP(S):

    # API mode: pull recent comments from conflict-heavy subreddits
    .venv/bin/python scripts/07_reddit_fetch.py --api \
        --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
        --per_subreddit 50000 --after 2024-01-01

    # dump mode: stream a monthly .zst dump (torrent first) or a local file
    .venv/bin/python scripts/07_reddit_fetch.py --dump RC_2025-01.zst \
        --clean_sample_frac 0.05 --out data/reddit/raw

Regex pre-filtering keeps every attack / grey / emotional candidate and only
a small sample of clean comments (the clean bank fills up easily), so the
resulting parquet is a fraction of the raw dump size.

Output: data/reddit/raw/*.parquet with columns
    text, subreddit, created_utc, score, id, category, profanity, insult, attack

API reference: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
"""

from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd
import requests
from tqdm import tqdm

from candidate_common import Sink, _is_latin
from common import clean_text
from reddit_vocab import prefilter_keep

API_BASE = "https://arctic-shift.photon-reddit.com/api"

# conflict / profanity-prone subreddits used when --subreddits is not given
DEFAULT_SUBREDDITS = [
    "PublicFreakout", "FightPorn", "politics", "Conservative", "WorldNews",
    "AskReddit", "leagueoflegends", "GlobalOffensive", "VALORANT", "nba",
    "nfl", "soccer", "boxing", "MMA", "ufc", "AmItheAsshole", "wallstreetbets",
    "gaming", "Teenagers", "unpopularopinion", "RoastMe", "pcmasterrace",
    "WhitePeopleTwitter", "BlackPeopleTwitter", "PoliticalHumor", "sports",
]

REMOVED = {"[removed]", "[deleted]", ""}


# --------------------------------------------------------------------------- #
# Row-level helpers
# --------------------------------------------------------------------------- #
def _row_from_comment(c: dict) -> dict | None:
    body = clean_text(c.get("body", ""))
    if not body or body in REMOVED:
        return None
    return {
        "text": body,
        "source": "reddit",
        "subreddit": c.get("subreddit", ""),
        "created_utc": int(c.get("created_utc") or 0),
        "score": int(c.get("score") or 0),
        "id": c.get("id", ""),
    }


# --------------------------------------------------------------------------- #
# Arctic Shift API
# --------------------------------------------------------------------------- #
def _api_get(path: str, params: dict, retries: int) -> dict | None:
    url = f"{API_BASE}{path}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                reset = r.headers.get("X-RateLimit-Reset")
                wait = float(reset) if reset else 15 * (attempt + 1)
                print(f"    rate-limited; sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(5 * (attempt + 1))
    print(f"    giving up on {path}: {last_err}")
    return None


def fetch_api(args, sink: Sink, rng) -> None:
    subs = (
        [s.strip() for s in args.subreddits.split(",") if s.strip()]
        if args.subreddits
        else DEFAULT_SUBREDDITS
    )
    for sub in subs:
        print(f"=== r/{sub} ===")
        before = args.before  # iso date or None
        window = 0
        with tqdm(total=args.max_pages, desc=f"r/{sub}", unit="page", leave=False) as pbar:
            while True:
                params = {
                    "subreddit": sub,
                    "limit": "auto",
                    "sort": "desc",
                    "fields": "body,id,subreddit,created_utc,score",
                }
                if args.after:
                    params["after"] = args.after
                if before:
                    params["before"] = before

                data = _api_get("/comments/search", params, retries=args.retries)
                if not data or not data.get("data"):
                    break
                comments = data["data"]
                if not comments:
                    break

                oldest = None
                for c in comments:
                    row = _row_from_comment(c)
                    if row is None:
                        continue
                    if len(row["text"].split()) < args.min_words:
                        continue
                    if args.require_latin and not _is_latin(row["text"]):
                        continue
                    info, keep = prefilter_keep(row["text"], args.clean_sample_frac, rng)
                    if keep:
                        row["category"] = info["category"]
                        row["profanity"] = int(info["profanity"])
                        row["insult"] = int(info["insult"])
                        row["attack"] = int(info["attack"])
                        sink.add(row)
                    if oldest is None or c["created_utc"] < oldest:
                        oldest = c["created_utc"]

                if sink.kept >= args.max_rows:
                    print("  reached --max_rows, stopping")
                    break
                if oldest is None:
                    break
                before = oldest - 1  # epoch seconds, exclusive cursor
                window += 1
                pbar.update(1)
                pbar.set_postfix(kept=sink.kept)
                if window >= args.max_pages:
                    print(f"  reached {args.max_pages} pages for r/{sub}, stopping")
                    break
                time.sleep(args.api_delay)
        if sink.kept >= args.max_rows:
            break
        print(f"  r/{sub} done (total kept so far: {sink.kept:,})")


# --------------------------------------------------------------------------- #
# Dump files (.zst / .zst_blocks / .jsonl / .ndjson / .json)
# --------------------------------------------------------------------------- #
def _iter_dump_blocks(fh):
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    for raw in fh:
        if not raw.strip():
            continue
        try:
            yield dctx.decompress(raw)
        except Exception:
            continue


def _iter_lines(stream):
    buf = b""
    for chunk in stream:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                yield line
    if buf.strip():
        yield buf


def iter_comment_dicts(path_or_url: str):
    """Yield raw comment dicts from a local file or HTTP(S) dump."""
    if path_or_url.startswith(("http://", "https://")):
        with requests.get(path_or_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            yield from _iter_lines(r.iter_content(chunk_size=1 << 20))
    elif path_or_url.endswith(".zst"):
        import zstandard as zstd

        with open(path_or_url, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=True) as rd:
                yield from _iter_lines(rd)
    elif path_or_url.endswith(".zst_blocks"):
        with open(path_or_url, "rb") as fh:
            for blob in _iter_dump_blocks(fh):
                yield blob
    else:
        with open(path_or_url, "rb") as fh:
            yield from _iter_lines([fh.read()])


def fetch_dump(args, sink: Sink, rng) -> None:
    n = 0
    print(f"reading {args.dump} ...")
    with tqdm(desc="scanning", unit="comment", leave=False) as pbar:
        for raw in iter_comment_dicts(args.dump):
            try:
                c = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            row = _row_from_comment(c)
            if row is None:
                continue
            if len(row["text"].split()) < args.min_words:
                continue
            if args.require_latin and not _is_latin(row["text"]):
                continue
            info, keep = prefilter_keep(row["text"], args.clean_sample_frac, rng)
            if keep:
                row["category"] = info["category"]
                row["profanity"] = int(info["profanity"])
                row["insult"] = int(info["insult"])
                row["attack"] = int(info["attack"])
                sink.add(row)
            n += 1
            if sink.kept >= args.max_rows:
                print("  reached --max_rows, stopping")
                break
            pbar.update(1)
            pbar.set_postfix(scanned=n, kept=sink.kept)
    print(f"  scanned {n:,} comments | kept {sink.kept:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true", help="fetch via Arctic Shift API")
    ap.add_argument("--dump", default=None,
                    help="local path or HTTP(S) URL of a comment dump (.zst/.jsonl/.ndjson)")
    ap.add_argument("--subreddits", default=None,
                    help="comma-separated subreddits for API mode (default: conflict-heavy list)")
    ap.add_argument("--max_pages", type=int, default=5000,
                    help="max API pages (~1000 comments each) per subreddit")
    ap.add_argument("--after", default="2024-01-01", help="only comments after this date (API)")
    ap.add_argument("--before", default=None, help="only comments before this date (API)")
    ap.add_argument("--api_delay", type=float, default=0.3, help="seconds between API pages")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--clean_sample_frac", type=float, default=0.05,
                    help="fraction of regex-clean comments kept")
    ap.add_argument("--require_latin", action=argparse.BooleanOptionalAction, default=True,
                    help="drop comments that are mostly non-Latin (default on)")
    ap.add_argument("--max_rows", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/reddit/raw")
    ap.add_argument("--flush_rows", type=int, default=500_000)
    args = ap.parse_args()

    from pathlib import Path

    out_dir = Path(args.out)
    sink = Sink(out_dir, flush_rows=args.flush_rows)
    rng = random.Random(args.seed)

    if args.api:
        fetch_api(args, sink, rng)
    elif args.dump:
        fetch_dump(args, sink, rng)
    else:
        ap.error("specify --api or --dump")

    sink.close()
    print(f"\nkept {sink.kept:,} rows | dropped {sink.dropped_dup:,} exact dups | shards={sink.shard}")
    if sink.kept:
        df = pd.read_parquet(out_dir)
        print(df.category.value_counts().to_string())
        print(f"  -> {len(df):,} rows in {out_dir}")


if __name__ == "__main__":
    main()