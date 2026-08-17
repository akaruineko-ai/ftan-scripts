"""Rebalance the final dataset train split via cluster-level downsampling.

The train split of data/final/dataset is positive-biased (~63/37). This script
downsamples the majority class at the cluster level: whole near-duplicate
clusters (a base sentence plus all its obfuscation variants) are kept or
dropped together, so the near-duplicate invariant from step 4 is preserved.
The resulting train split is ~50/50.

Validation / test / test_obfuscated are left untouched. The original dataset
is backed up next to it before being overwritten, and stats.json /
dataset_card.md are regenerated.

Usage:
    .venv/bin/python scripts/make_balanced.py
    .venv/bin/python scripts/make_balanced.py --seed 7 --no-backup
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk

from common import FINAL_DIR

SCRIPTS = Path(__file__).resolve().parent

pd.set_option("future.no_silent_downcasting", True)


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export06", str(SCRIPTS / "06_export.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_stats(dsd: DatasetDict) -> dict:
    """Regenerate the stats.json structure from an in-memory DatasetDict."""
    splits = {k: len(v) for k, v in dsd.items()}
    labels: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    mutated: dict[str, dict] = {}
    for name, ds in dsd.items():
        for lbl in (0, 1):
            labels.setdefault(str(lbl), {})[name] = int(
                (np.asarray(ds["label"]) == lbl).sum()
            )
        for src, n in pd.Series(ds["source"]).value_counts().items():
            sources.setdefault(str(src), {})[name] = int(n)
        for m in (0, 1):
            mutated.setdefault(str(m), {})[name] = int(
                (np.asarray(ds["mutated"]) == m).sum()
            )
    return {
        "splits": splits,
        "total_rows": sum(splits.values()),
        "label_distribution": labels,
        "source_distribution": sources,
        "mutated_share": mutated,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(FINAL_DIR / "dataset"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the unbalanced backup copy")
    args = ap.parse_args()

    export = _load_export_module()
    data_dir = Path(args.data_dir)
    if not (data_dir / "dataset_dict.json").exists():
        raise SystemExit(f"{data_dir} does not look like a DatasetDict")

    dsd = load_from_disk(str(data_dir))
    print("before:", {k: len(v) for k, v in dsd.items()})

    train = dsd["train"]
    labels = np.asarray(train["label"])
    clusters = np.asarray(train["cluster"])

    cdf = pd.DataFrame({"cluster": clusters, "label": labels})
    agg = (
        cdf.groupby("cluster")
        .agg(size=("cluster", "size"), label=("label", "first"))
        .reset_index()
    )
    pos_total = int(agg.loc[agg["label"] == 1, "size"].sum())
    neg_total = int(agg.loc[agg["label"] == 0, "size"].sum())
    print(f"  train before: pos={pos_total:,} neg={neg_total:,} "
          f"({pos_total / (pos_total + neg_total):.1%} pos)")

    if pos_total >= neg_total:
        major_label, major_total, minor_total = 1, pos_total, neg_total
    else:
        major_label, major_total, minor_total = 0, neg_total, pos_total
    excess = major_total - minor_total
    if excess <= 0:
        print("train is already balanced; nothing to do")
        sys.exit(0)

    major = agg[agg["label"] == major_label].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(major))
    dropped, dropped_ids = 0, []
    for ix in order:
        dropped += int(major.loc[ix, "size"])
        dropped_ids.append(int(major.loc[ix, "cluster"]))
        if dropped >= excess:
            break

    keep = ~pd.Series(clusters).isin(set(dropped_ids))
    new_train = train.select(np.flatnonzero(keep.values).tolist())

    new_labels = np.asarray(new_train["label"])
    new_pos = int((new_labels == 1).sum())
    new_neg = int((new_labels == 0).sum())
    print(f"  dropped {len(dropped_ids):,} majority (label {major_label}) clusters "
          f"({dropped:,} rows)")
    print(f"  train after: pos={new_pos:,} neg={new_neg:,} "
          f"({new_pos / (new_pos + new_neg):.1%} pos)")

    new_dsd = DatasetDict({"train": new_train})
    for name in ("validation", "test", "test_obfuscated"):
        if name in dsd:
            new_dsd[name] = dsd[name]

    if not args.no_backup:
        backup = data_dir.parent / f"{data_dir.name}_unbalanced"
        if not backup.exists():
            shutil.copytree(data_dir, backup)
            print(f"backup of unbalanced dataset: {backup}")

    tmp = data_dir.parent / f"{data_dir.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    new_dsd.save_to_disk(str(tmp))
    export._clean_dir(tmp)
    shutil.rmtree(data_dir)
    tmp.rename(data_dir)
    export._clean_dir(data_dir)
    print(f"saved balanced DatasetDict to {data_dir}")

    stats = _build_stats(new_dsd)
    with open(FINAL_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    export._write_card(new_dsd, stats)
    print(f"saved {FINAL_DIR / 'stats.json'} and {FINAL_DIR / 'dataset_card.md'}")

    print("\n=== dataset summary ===")
    for name, ds in new_dsd.items():
        npos = int((np.asarray(ds["label"]) == 1).sum())
        print(f"  split {name:<16} {len(ds):>10,} rows  "
              f"(pos={npos:,} neg={len(ds) - npos:,})")
    print(f"  {'TOTAL':<22} {sum(len(ds) for ds in new_dsd.values()):>10,} rows")


if __name__ == "__main__":
    main()
