"""Build a benchmark parquet for scoring ftan moderation classifiers.

Reads data/processed/split.parquet, takes the held-out `test` rows, appends
the obfuscated subset of them (the same rows as the `test_obfuscated` split),
and tags every row with a `benchmark_split` column:

    test              plain + obfuscated test rows      (30k)
    test_obfuscated   the mutated subset, re-tagged      (14k)

Both sets carry the full schema, so a runner can also slice by `mutated`
(plain vs obfuscated) or by `source`.

Usage:
    .venv/bin/python scripts/make_benchmark.py
    .venv/bin/python scripts/make_benchmark.py --out data/final/benchmark/benchmark.parquet
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from common import FINAL_DIR, PROCESSED_DIR

pd.set_option("future.no_silent_downcasting", True)

COLS = ["text", "label", "source", "origin_label", "split_origin",
        "mutated", "variant", "cluster", "benchmark_split"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", default=str(PROCESSED_DIR / "split.parquet"),
                    help="pipeline parquet with a `split` column (default: "
                         "data/processed/split.parquet)")
    ap.add_argument("--out", default=str(FINAL_DIR / "benchmark" / "benchmark.parquet"),
                    help="output parquet path")
    args = ap.parse_args()

    df = pd.read_parquet(args.split_file)
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise SystemExit("no `test` rows found in the split file")

    obf = test[test["mutated"] == 1].copy()
    print(f"test rows: {len(test):,} (mutated {len(obf):,})")

    test["benchmark_split"] = "test"
    obf["benchmark_split"] = "test_obfuscated"

    bench = pd.concat([test, obf], ignore_index=True)[COLS]
    bench = bench.sort_values(["benchmark_split", "cluster"]).reset_index(drop=True)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bench.to_parquet(out, index=False)

    print("\n--- benchmark_split x label ---")
    print(bench.groupby(["benchmark_split", "label"]).size().unstack(fill_value=0).to_string())
    print("\n--- benchmark_split x source ---")
    print(bench.groupby(["benchmark_split", "source"]).size().unstack(fill_value=0).to_string())
    print(f"\nsaved: {out} ({len(bench):,} rows)")


if __name__ == "__main__":
    main()
