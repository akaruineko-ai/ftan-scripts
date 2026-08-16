"""Step 6: Export a Hugging Face DatasetDict.

- Reads data/processed/split.parquet
- Builds a `datasets.DatasetDict` with train / validation / test
- Optionally adds `test_obfuscated` (mutated rows held out for robustness eval)
- Saves to data/final/dataset (save_to_disk) and, optionally, pushes to the Hub
- Writes data/final/stats.json and data/final/dataset_card.md

Columns in the exported dataset:
    text, label (1=offensive, 0=clean), source, origin_label,
    split_origin, mutated (0/1), variant, cluster
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

from common import FINAL_DIR, PROCESSED_DIR

pd.set_option("future.no_silent_downcasting", True)

FEATURES = Features({
    "text": Value("string"),
    "label": Value("int8"),
    "source": Value("string"),
    "origin_label": Value("string"),
    "split_origin": Value("string"),
    "mutated": Value("int8"),
    "variant": Value("int8"),
    "cluster": Value("int64"),
})


def _split_df(df, split_name):
    return df[df["split"] == split_name].drop(columns=["split"]).reset_index(drop=True)


def _clean_dir(path):
    """Remove stray cache artifacts that break Hub auto-processing."""
    if not path.exists():
        return
    for p in path.glob("cache-*.arrow"):
        p.unlink()
    for split_dir in path.iterdir():
        if split_dir.is_dir():
            for p in split_dir.glob("cache-*.arrow"):
                p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_obfuscation", action="store_true", default=True,
                    help="add a test_obfuscated split with mutated test rows")
    ap.add_argument("--hub_id", default=None,
                    help="push to the Hub under this repo id (e.g. user/name)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    src = PROCESSED_DIR / "split.parquet"
    df = pd.read_parquet(src)
    print(f"rows: {len(df):,}")

    splits = {}
    for name in ("train", "validation", "test"):
        d = _split_df(df, name)
        splits[name] = Dataset.from_pandas(d, features=FEATURES, preserve_index=False)
        print(f"  {name}: {len(d):,}")

    if args.eval_obfuscation:
        obf = _split_df(df, "test")
        obf = obf[obf["mutated"] == 1].reset_index(drop=True)
        if len(obf):
            splits["test_obfuscated"] = Dataset.from_pandas(obf, features=FEATURES, preserve_index=False)
            print(f"  test_obfuscated: {len(obf):,}")
        else:
            print("  test_obfuscated: empty, skipping")

    dsd = DatasetDict(splits)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out = FINAL_DIR / "dataset"
    _clean_dir(out)
    dsd.save_to_disk(str(out))
    _clean_dir(out)
    print(f"\nsaved DatasetDict to {out}")

    stats = {
        "splits": {k: len(v) for k, v in dsd.items()},
        "total_rows": sum(len(v) for v in dsd.values()),
        "label_distribution": (
            df.groupby(["split", "label"]).size().unstack(fill_value=0).astype(int).to_dict()
        ),
        "source_distribution": (
            df.groupby(["split", "source"]).size().unstack(fill_value=0).astype(int).to_dict()
        ),
        "mutated_share": (
            df.groupby(["split", "mutated"]).size().unstack(fill_value=0).astype(int).to_dict()
        ),
    }
    with open(FINAL_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {FINAL_DIR / 'stats.json'}")

    _write_card(dsd, stats)
    print(f"saved {FINAL_DIR / 'dataset_card.md'}")

    if args.hub_id:
        dsd.push_to_hub(args.hub_id, private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.hub_id}")


def _write_card(dsd, stats):
    lines = [
        "---",
        "license: cc0-1.0",
        "task_categories:",
        "  - text-classification",
        "language:",
        "  - en",
        "size_categories:",
        "  - 100K<n<1M",
        "---",
        "",
        "# ftan-2.0 offensive / clean dataset",
        "",
        "Binary offensive-language classification dataset combining several public sources and",
        "augmented with an **obfuscation engine** (leet-speak, separators, censoring, repeated",
        "chars, case shuffle, unicode homoglyphs, fullwidth) so classifiers learn to detect",
        "censored / mutated curse words.",
        "",
        "## Schema",
        "",
        "| column | type | meaning |",
        "|---|---|---|",
        "| `text` | string | input sentence |",
        "| `label` | int8 | 1 = offensive, 0 = clean |",
        "| `source` | string | originating dataset |",
        "| `origin_label` | string | label as given by the source |",
        "| `split_origin` | string | original train/test split of the source |",
        "| `mutated` | int8 | 1 = obfuscation-engine variant |",
        "| `variant` | int8 | variant index within the source row |",
        "| `cluster` | int64 | near-duplicate cluster id (split safely) |",
        "",
        "## Splits",
        "",
    ]
    for k, v in stats["splits"].items():
        lines.append(f"- `{k}`: {v:,} rows")
    lines += [
        "",
        "## Sources",
        "",
        "- **Jigsaw Toxic Comment** (`tcapelle/jigsaw-toxic-comment-classification-challenge`)",
        "- **Davidson et al. hate/offensive** (`contemmcm/hate-speech-and-offensive-language`)",
        "- **ToxiGen** (`toxigen/toxigen-data`, human toxicity scores)",
        "- **HateXplain** (`Hate-speech-CNERG/hatexplain`)",
        "- **Wikipedia** neutral filler (`wikimedia/wikipedia` 20231101.en)",
        "",
        "## Design notes",
        "",
        "- Both classes receive orthographic mutations so obfuscation itself is not a cue for",
        "  offense (prevents false positives on innocent leet like `ex4mp1e`).",
        "- Split is performed at the **near-duplicate cluster** level so a mutated variant never",
        "  leaks across train/validation/test.",
        "- `test_obfuscated` holds mutated rows of the test set for robustness evaluation.",
        "",
        "## Disclaimer",
        "",
        "This dataset contains raw offensive language. It is intended for research and",
        "moderation-model training only.",
    ]
    (FINAL_DIR / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
