"""Step 5: Cluster-stratified train / validation / test split.

Splitting happens at the cluster level (see step 4) so that mutated variants
of the same base sentence never straddle two splits.

Test set is built first from held-out sources (HateXplain + Jigsaw
balanced_test), then topped up to a balanced target.
Validation is a balanced sample; the remainder is the train set.

Outputs: data/processed/split.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import PROCESSED_DIR

pd.set_option("future.no_silent_downcasting", True)

HELD_OUT_SOURCES = {"hatexplain"}


def _cluster_table(df: pd.DataFrame):
    cdf = (
        df.groupby("cluster")
        .agg(
            size=("text", "size"),
            label=("label", lambda s: int(s.mode().iloc[0])),
            source=("source", "first"),
            split_origin=("split_origin", "first"),
        )
        .reset_index()
    )
    return cdf


def _pick(cdf, target_pos, target_neg, exclude, rng):
    """Greedily sample clusters to reach per-class targets. Returns picked clusters."""
    remaining = cdf[~cdf["cluster"].isin(exclude)].copy()
    remaining = remaining.iloc[rng.permutation(len(remaining))]
    picked, pos, neg = [], 0, 0
    for _, r in remaining.iterrows():
        if pos >= target_pos and neg >= target_neg:
            break
        if r["label"] == 1 and pos < target_pos:
            picked.append(r["cluster"])
            pos += r["size"]
        elif r["label"] == 0 and neg < target_neg:
            picked.append(r["cluster"])
            neg += r["size"]
    return picked, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_target", type=int, default=30_000)
    ap.add_argument("--val_target", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = PROCESSED_DIR / "deduped.parquet"
    df = pd.read_parquet(src)
    cdf = _cluster_table(df)
    print(f"rows: {len(df):,} | clusters: {len(cdf):,}")

    # ---- test: balanced sample from held-out sources ----
    held = cdf[cdf["source"].isin(HELD_OUT_SOURCES) | (cdf["split_origin"] == "test")]
    test_picked, test_pos, test_neg = _pick(
        held, args.test_target // 2, args.test_target // 2, set(), rng
    )

    # ---- validation ----
    val_picked, val_pos, val_neg = _pick(
        cdf, args.val_target // 2, args.val_target // 2, set(test_picked), rng
    )

    assigned = set(test_picked) | set(val_picked)
    train_picked = cdf[~cdf["cluster"].isin(assigned)]["cluster"].tolist()

    split_of_cluster = {}
    for c in test_picked:
        split_of_cluster[c] = "test"
    for c in val_picked:
        split_of_cluster[c] = "validation"
    for c in train_picked:
        split_of_cluster[c] = "train"

    df["split"] = df["cluster"].map(split_of_cluster)
    assert df["split"].notna().all(), "unmapped cluster!"

    print("\n--- split x class ---")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\n--- test split x source ---")
    print(df[df["split"] == "test"].groupby(["source", "label"]).size().unstack(fill_value=0).to_string())

    df.to_parquet(PROCESSED_DIR / "split.parquet", index=False)
    print(f"\nsaved: {PROCESSED_DIR / 'split.parquet'}")


if __name__ == "__main__":
    main()
