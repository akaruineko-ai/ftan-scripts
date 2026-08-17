"""Shared helpers for the community-data fetchers (Reddit / 4chan / YouTube).

All fetchers write the same candidate parquet schema:
    text, source, subreddit, created_utc, score, id,
    category, profanity, insult, attack

`source` is one of "reddit" / "4chan" / "youtube" and is kept through the
curate step into the banks, so the merged dataset records real provenance.
"""

from __future__ import annotations

import hashlib
import html
import re

import pandas as pd

from pathlib import Path

from reddit_vocab import prefilter_keep  # noqa: F401  (re-exported for fetchers)


def _is_latin(text: str, min_ratio: float = 0.5) -> bool:
    """Heuristic: skip comments that are mostly non-Latin (non-English)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    latin = sum(1 for ch in letters if ord(ch) < 0x250)
    return latin / len(letters) >= min_ratio


class Sink:
    """Collects rows and flushes them to sharded parquet files."""

    def __init__(self, out_dir, flush_rows=500_000):
        self.out_dir = Path(out_dir)
        self.flush_rows = flush_rows
        self.rows: list[dict] = []
        self.seen: set[int] = set()
        self.shard = 0
        self.kept = 0
        self.dropped_dup = 0

    def add(self, row: dict) -> None:
        digest = int.from_bytes(hashlib.md5(row["text"].encode("utf-8")).digest()[:8], "big")
        if digest in self.seen:
            self.dropped_dup += 1
            return
        self.seen.add(digest)
        self.rows.append(row)
        self.kept += 1
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"candidates_{self.shard:03d}.parquet"
        df = pd.DataFrame(self.rows)
        df.to_parquet(path, index=False)
        self.rows = []
        self.shard += 1

    def close(self) -> None:
        self.flush()

    def load_existing(self) -> int:
        """Rebuild `seen`/`kept`/`shard` from parquet shards already on disk.

        Used when resuming from a checkpoint: previously kept rows are loaded
        into the dedup set so we don't re-write them, and `shard` continues so
        new flushes don't overwrite old files. Returns the number of rows loaded.
        """
        if not self.out_dir.exists():
            return 0
        files = sorted(self.out_dir.glob("candidates_*.parquet"))
        total = 0
        for p in files:
            df = pd.read_parquet(p)
            for t in df["text"]:
                digest = int.from_bytes(hashlib.md5(str(t).encode("utf-8")).digest()[:8], "big")
                self.seen.add(digest)
            total += len(df)
        self.kept = total
        self.shard = len(files)
        self.rows = []
        return total


# --------------------------------------------------------------------------- #
# 4chan post text
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_SPAN_RE = re.compile(r"class=\"quote\"|class=\"deadlink\"|class=\"quoteLink\"")


def clean_4chan_html(com: str | None) -> str:
    """Turn a 4chan post `com` field (escaped HTML) into plain text."""
    if not com:
        return ""
    s = html.unescape(com)
    s = _SPAN_RE.sub("", s)
    s = _TAG_RE.sub(" ", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = re.sub(r" *\n *", " ", s)
    return s.strip()