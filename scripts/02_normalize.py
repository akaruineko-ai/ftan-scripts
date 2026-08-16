"""Step 2: Merge raw parquets into one normalized dataset.

- Concatenate all sources
- Exact-deduplicate on lowercased text (keep first occurrence)
- Filter: non-empty, 3..512 tokens
- Print per-source / per-class stats
Outputs: data/processed/normalized.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import PROCESSED_DIR, RAW_DIR, UNIFIED_COLUMNS

pd.set_option("future.no_silent_downcasting", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in sorted(RAW_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        frames.append(df)
        print(f"  {p.name}: {len(df):,} rows")
    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df["_key"] = df["text"].str.lower().str.strip()
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    after_exact = len(df)

    lens = df["text"].str.split().str.len()
    df = df[(lens >= args.min_len) & (lens <= args.max_len)]
    after_len = len(df)

    print(f"\nrows: {before:,} -> exact-dedup {after_exact:,} -> length-filter {after_len:,}")

    print("\n--- by class ---")
    print(df["label"].value_counts().sort_index().to_string())

    print("\n--- by source x class ---")
    cross = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    print(cross.to_string())

    df[UNIFIED_COLUMNS].to_parquet(PROCESSED_DIR / "normalized.parquet", index=False)
    print(f"\nsaved: {PROCESSED_DIR / 'normalized.parquet'}")


if __name__ == "__main__":
    main()
