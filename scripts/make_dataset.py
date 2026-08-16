"""One-shot dataset builder: runs the whole ftan-2.0 pipeline in order.

Core pipeline (sources -> raw parquet -> normalized -> mutated -> deduped ->
split -> HF DatasetDict in data/final/dataset):
    download  -> scripts/01_download.py
    normalize -> scripts/02_normalize.py
    mutate    -> scripts/03_mutate.py
    dedup     -> scripts/04_dedup.py
    split     -> scripts/05_split.py
    export    -> scripts/06_export.py

Reddit expansion (new 3M dataset from live comments):
    fetch     -> scripts/07_reddit_fetch.py   (candidates, regex prefiltered)
    curate    -> scripts/08_reddit_curate.py  (regex + FTAN -> banks)
    merge     -> (built in) banks -> data/raw/reddit.parquet, so the core
                 pipeline picks them up like any other source

Training (optional, after the dataset exists):
    train     -> scripts/train.py + scripts/finalize.py -> data/final/model

Usage:
    # full default run (core sources; reddit auto-skipped unless --api/--dump)
    .venv/bin/python scripts/make_dataset.py --device 0

    # core + reddit, target 1M per bank
    .venv/bin/python scripts/make_dataset.py --api \
        --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
        --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0

    # reddit from a monthly dump only
    .venv/bin/python scripts/make_dataset.py --steps reddit --dump RC_2025-01.zst

    # reuse already-fetched candidates (skip fetch, keep curate)
    .venv/bin/python scripts/make_dataset.py --skip-fetch --device 0

    # build the dataset, then train the model on it
    .venv/bin/python scripts/make_dataset.py --steps all,train --device 0

Steps run in order and stream their output; a summary of the resulting
dataset is printed at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import RAW_DIR, UNIFIED_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PROCESSED_DIR = ROOT / "data" / "processed"
FINAL_DIR = ROOT / "data" / "final"

PRESETS = {
    "core": "download,normalize,mutate,dedup,split,export",
    "reddit": "fetch,curate,merge,normalize,mutate,dedup,split,export",
    "all": "download,fetch,curate,merge,normalize,mutate,dedup,split,export",
    "train": "train,finalize",
}
STEPS_ORDER = [
    "download", "fetch", "curate", "merge",
    "normalize", "mutate", "dedup", "split", "export",
    "train", "finalize",
]


def _resolve_steps(value: str) -> list[str]:
    steps = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in PRESETS:
            for sub in _resolve_steps(PRESETS[tok]):
                if sub not in steps:
                    steps.append(sub)
        else:
            if tok not in STEPS_ORDER:
                raise SystemExit(f"unknown step {tok!r}; choose from {', '.join(STEPS_ORDER)}")
            if tok not in steps:
                steps.append(tok)
    return steps


def _run(script: str, argv: list[str]) -> None:
    cmd = [sys.executable, "-u", str(SCRIPTS / script), *argv]
    print(f"\n### {script} {' '.join(argv)}", flush=True)
    subprocess.run(cmd, check=True)


def _notice(msg: str) -> None:
    print(f"\n### [make_dataset] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# step 1: core sources
# --------------------------------------------------------------------------- #
def build_download_args(args) -> list[str]:
    a = ["--wiki_cap", str(args.wiki_cap)]
    if args.sources:
        a += ["--sources", *args.sources]
    return a


def step_download(args) -> bool:
    if not args.re_download and any(RAW_DIR.glob("*.parquet")):
        _notice("data/raw/ already has parquet sources, skipping download "
                "(pass --re-download to force)")
        return False
    _run("01_download.py", build_download_args(args))
    return True


# --------------------------------------------------------------------------- #
# reddit expansion
# --------------------------------------------------------------------------- #
def build_fetch_args(args) -> list[str]:
    a = []
    if args.api:
        a += ["--api"]
    if args.dump:
        a += ["--dump", args.dump]
    if args.subreddits:
        a += ["--subreddits", args.subreddits]
    if args.after:
        a += ["--after", args.after]
    if args.before:
        a += ["--before", args.before]
    a += [
        "--max_pages", str(args.max_pages),
        "--max_rows", str(args.max_rows),
        "--min_words", str(args.min_words),
        "--clean_sample_frac", str(args.clean_sample_frac),
        "--api_delay", str(args.api_delay),
        "--retries", str(args.retries),
        "--out", str(args.raw_out),
        "--flush_rows", str(args.flush_rows),
        "--seed", str(args.seed),
    ]
    if not args.require_latin:
        a += ["--no-require_latin"]
    return a


def step_fetch(args) -> bool:
    if not (args.api or args.dump):
        _notice("no --api or --dump given, skipping fetch")
        return False
    _run("07_reddit_fetch.py", build_fetch_args(args))
    return True


def build_curate_args(args) -> list[str]:
    a = [
        "--raw", str(args.raw),
        "--out", str(args.banks_out),
        "--model", str(args.model),
        "--batch_size", str(args.batch_size),
        "--target_a", str(args.target_a),
        "--target_b", str(args.target_b),
        "--target_c", str(args.target_c),
        "--grey_low", str(args.grey_low),
        "--grey_high", str(args.grey_high),
        "--verify_frac", str(args.verify_frac),
        "--verify_attack_conf", str(args.verify_attack_conf),
        "--verify_clean_conf", str(args.verify_clean_conf),
        "--seed", str(args.seed),
    ]
    if args.device:
        a += ["--device", str(args.device)]
    return a


def step_curate(args) -> bool:
    raw = Path(args.raw)
    if not raw.is_dir() or not list(raw.glob("*.parquet")):
        _notice(f"no candidates in {raw}, skipping curate")
        return False
    _run("08_reddit_curate.py", build_curate_args(args))
    return True


def step_merge(args) -> bool:
    """Turn curated banks into data/raw/reddit.parquet (unified schema)."""
    banks = Path(args.banks_out)
    parts = []
    for name in ("bank_a", "bank_b", "bank_c"):
        p = banks / f"{name}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        _notice(f"no banks in {banks}, skipping merge")
        return False

    df = pd.concat(parts, ignore_index=True)
    df = df[df["label"].isin((0, 1))]  # manual_check (-1) needs human review
    df = df.reindex(columns=UNIFIED_COLUMNS)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "reddit.parquet"
    df.to_parquet(out, index=False)
    print(f"\n### merge: {len(df):,} reddit rows -> {out}")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    return True


# --------------------------------------------------------------------------- #
# core processing steps
# --------------------------------------------------------------------------- #
def build_normalize_args(args) -> list[str]:
    return ["--min_len", str(args.min_len), "--max_len", str(args.max_len)]


def build_mutate_args(args) -> list[str]:
    return [
        "--offensive_variants", str(args.offensive_variants),
        "--clean_variant_frac", str(args.clean_variant_frac),
        "--max_pos_per_source", str(args.max_pos_per_source),
        "--max_neg_per_source", str(args.max_neg_per_source),
        "--seed", str(args.seed),
    ]


def build_split_args(args) -> list[str]:
    return [
        "--test_target", str(args.test_target),
        "--val_target", str(args.val_target),
        "--seed", str(args.seed),
    ]


def build_export_args(args) -> list[str]:
    a = []
    if not args.eval_obfuscation:
        a += ["--no-eval_obfuscation"]
    if args.hub_id:
        a += ["--hub_id", args.hub_id]
    if args.private:
        a += ["--private"]
    return a


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def build_train_args(args) -> list[str]:
    a = [
        "--model", args.model,
        "--data_dir", str(args.data_dir),
        "--output_dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]
    if args.max_train_rows is not None:
        a += ["--max_train_rows", str(args.max_train_rows)]
    if args.max_val_rows is not None:
        a += ["--max_val_rows", str(args.max_val_rows)]
    if args.fp16:
        a += ["--fp16"]
    if args.push_to_hub:
        a += ["--push_to_hub", args.push_to_hub]
    return a


def build_finalize_args(args) -> list[str]:
    a = ["--output_dir", str(args.output_dir)]
    if args.data_dir:
        a += ["--data_dir", str(args.data_dir)]
    if args.max_length != "auto":
        a += ["--max_length", str(args.max_length)]
    if args.max_val_rows is not None:
        a += ["--max_val_rows", str(args.max_val_rows)]
    return a


def step_train(args) -> None:
    _run("train.py", build_train_args(args))
    _run("finalize.py", build_finalize_args(args))


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def print_summary() -> None:
    print("\n=== dataset summary ===")
    if (FINAL_DIR / "dataset").is_dir():
        try:
            from datasets import load_from_disk
            dsd = load_from_disk(str(FINAL_DIR / "dataset"))
            for name, ds in dsd.items():
                pos = sum(1 for lbl in ds["label"] if lbl == 1)
                print(f"  split {name:<16} {len(ds):>10,} rows  "
                      f"(pos={pos:,} neg={len(ds) - pos:,})")
            print(f"  {'TOTAL':<22} {sum(len(ds) for ds in dsd.values()):>10,} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read final dataset: {exc}")

    banks = ROOT / "data" / "reddit" / "banks"
    if banks.is_dir():
        total = 0
        for name in ("bank_a", "bank_b", "bank_c", "manual_check"):
            p = banks / f"{name}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            total += len(df)
            print(f"  bank {name:<12} {len(df):>10,} rows")
        print(f"  {'banks TOTAL':<22} {total:>10,} rows")
    if not (FINAL_DIR / "dataset").is_dir() and not banks.is_dir():
        print("  nothing built yet")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # what to run
    ap.add_argument("--steps", default="all",
                    help="comma-separated steps, or preset: "
                         f"{', '.join(PRESETS)} "
                         f"(steps: {', '.join(STEPS_ORDER)})")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-curate", action="store_true")
    ap.add_argument("--re-download", action="store_true",
                    help="force re-running download even if data/raw has parquets")

    # step 1: download
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--wiki_cap", type=int, default=350_000)

    # fetch
    ap.add_argument("--api", action="store_true", help="fetch via Arctic Shift API")
    ap.add_argument("--dump", default=None,
                    help="local path or HTTP(S) URL of a comment dump (.zst/.jsonl)")
    ap.add_argument("--subreddits", default=None,
                    help="comma-separated subreddits for API mode")
    ap.add_argument("--after", default="2024-01-01")
    ap.add_argument("--before", default=None)
    ap.add_argument("--max_pages", type=int, default=5000)
    ap.add_argument("--max_rows", type=int, default=5_000_000)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--clean_sample_frac", type=float, default=0.05)
    ap.add_argument("--require_latin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--api_delay", type=float, default=0.3)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--raw_out", default="data/reddit/raw")
    ap.add_argument("--flush_rows", type=int, default=500_000)

    # curate
    ap.add_argument("--raw", default="data/reddit/raw",
                    help="candidate parquet dir/file for the curate step")
    ap.add_argument("--banks_out", default="data/reddit/banks")
    ap.add_argument("--model", default="data/final/model/model")
    ap.add_argument("--device", default=None, help="cuda device id, e.g. 0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--target_a", type=int, default=1_000_000)
    ap.add_argument("--target_b", type=int, default=1_000_000)
    ap.add_argument("--target_c", type=int, default=1_000_000)
    ap.add_argument("--grey_low", type=float, default=0.30)
    ap.add_argument("--grey_high", type=float, default=0.70)
    ap.add_argument("--verify_frac", type=float, default=0.0)
    ap.add_argument("--verify_attack_conf", type=float, default=0.40)
    ap.add_argument("--verify_clean_conf", type=float, default=0.90)

    # normalize / mutate / split / export
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--offensive_variants", type=int, default=3,
                    help="obfuscation variants per offensive row (0 = no mutation)")
    ap.add_argument("--clean_variant_frac", type=float, default=0.4)
    ap.add_argument("--max_pos_per_source", type=int, default=0)
    ap.add_argument("--max_neg_per_source", type=int, default=0)
    ap.add_argument("--test_target", type=int, default=30_000)
    ap.add_argument("--val_target", type=int, default=20_000)
    ap.add_argument("--eval_obfuscation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hub_id", default=None)
    ap.add_argument("--private", action="store_true")

    # train
    ap.add_argument("--data_dir", default=str(FINAL_DIR / "dataset"))
    ap.add_argument("--output_dir", default=str(FINAL_DIR / "model"))
    ap.add_argument("--max_train_rows", type=int, default=None)
    ap.add_argument("--max_val_rows", type=int, default=None)
    ap.add_argument("--max_length", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--push_to_hub", default=None)

    # global
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    steps = [s for s in _resolve_steps(args.steps) if s != "download"]
    if "download" in _resolve_steps(args.steps) and not args.skip_download:
        steps = ["download", *steps]

    if args.skip_fetch:
        steps = [s for s in steps if s != "fetch"]
    if args.skip_curate:
        steps = [s for s in steps if s != "curate"]

    print(f"plan: {' -> '.join(steps)}", flush=True)

    if "download" in steps:
        step_download(args)

    if "fetch" in steps:
        step_fetch(args)
    if "curate" in steps:
        step_curate(args)
    if "merge" in steps:
        step_merge(args)

    if "normalize" in steps:
        _run("02_normalize.py", build_normalize_args(args))
    if "mutate" in steps:
        _run("03_mutate.py", build_mutate_args(args))
    if "dedup" in steps:
        _run("04_dedup.py", [])
    if "split" in steps:
        _run("05_split.py", build_split_args(args))
    if "export" in steps:
        _run("06_export.py", build_export_args(args))

    if "train" in steps:
        step_train(args)
    elif "finalize" in steps:
        _run("finalize.py", build_finalize_args(args))

    print_summary()


if __name__ == "__main__":
    main()