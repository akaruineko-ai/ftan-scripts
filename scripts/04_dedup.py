"""Step 4: Orthographic-cluster assignment (dedup + leak-safe clusters).

Exact duplicates were already removed in step 3. This step:
1. Canonicalizes each text: lowercase, homoglyph -> latin, leet digits ->
   letters, strip separators/punctuation, collapse repeated chars.
2. Clusters rows that share the same canonical form (i.e. a sentence and its
   leet / censored / homoglyph / fullwidth variants).
3. Keeps *all* rows (mutation variants are valuable training data) but tags
   them with a shared `cluster` id so train/validation/test never split a
   sentence from its own obfuscated variants.
4. Drops rows that are exact text duplicates of another kept row.

Outputs: data/processed/deduped.parquet (adds `cluster` column)
"""

from __future__ import annotations

import argparse
import re

import pandas as pd
from tqdm import tqdm

from common import PROCESSED_DIR

pd.set_option("future.no_silent_downcasting", True)

# homoglyph (Cyrillic / Greek lookalikes) -> latin
_HOMO_REV = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0441": "c", "\u0440": "p",
    "\u0443": "y", "\u0445": "x", "\u043a": "k", "\u043c": "m", "\u0442": "t",
    "\u043d": "h", "\u0456": "i", "\u044c": "b", "\u03bf": "o", "\u03b9": "i",
    "\u03b5": "e", "\u03c1": "p", "\u03c3": "c", "\u03c4": "t",
}
# leet digits -> letters (common, unambiguous mappings)
_LEET_REV = {
    "4": "a", "8": "b", "3": "e", "9": "g", "1": "i", "0": "o",
    "5": "s", "7": "t", "2": "z",
}
_FULLWIDTH_OFFSET = 0xFEE0

_TRANS = str.maketrans({**{k: v for k, v in _HOMO_REV.items()},
                        **{chr(ord(c) + _FULLWIDTH_OFFSET): c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}})


def canonicalize(text: str) -> str:
    t = text.translate(_TRANS).lower()
    for d, ch in _LEET_REV.items():
        t = t.replace(d, ch)
    t = re.sub(r"[^a-z0-9]+", "", t)
    t = re.sub(r"(.)\1+", r"\1", t)  # fuuuuuck -> fuck
    return t


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    src = PROCESSED_DIR / "mutated.parquet"
    df = pd.read_parquet(src)
    print(f"input rows: {len(df):,}")

    canon = [canonicalize(t) for t in tqdm(df["text"], desc="canonicalize")]
    df["_canon"] = canon

    # exact-text duplicates -> drop all but first
    df["_exact_ix"] = df["text"].str.lower().str.strip()
    before_exact = len(df)
    df = df.drop_duplicates(subset="_exact_ix", keep="first")
    print(f"exact-text dups removed: {before_exact - len(df):,}")

    # cluster by canonical form
    codes, uniques = pd.factorize(df["_canon"], sort=False)
    df["cluster"] = pd.Series(codes, index=df.index)
    df = df.drop(columns=["_canon", "_exact_ix"]).reset_index(drop=True)

    n_clusters = uniques.size
    print(f"clusters: {n_clusters:,} | kept rows: {len(df):,}")

    # only-cluster-singletons note
    sizes = df.groupby("cluster").size()
    multi = int((sizes > 1).sum())
    print(f"multi-row clusters: {multi:,} (variants held together for splitting)")

    print("\nby class:")
    print(df["label"].value_counts().sort_index().rename("count").to_string())
    print("\nby source x class:")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())

    df.to_parquet(PROCESSED_DIR / "deduped.parquet", index=False)
    print(f"\nsaved: {PROCESSED_DIR / 'deduped.parquet'}")


if __name__ == "__main__":
    main()
