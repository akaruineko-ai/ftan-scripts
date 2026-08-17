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
    "reddit": "fetch,curate,merge,subsample,normalize,mutate,dedup,split,export",
    "community": "fetch,fetch-4chan,fetch-hf,fetch-youtube,curate,merge,merge-ftan,subsample,normalize,mutate,dedup,split,export",
    "all": "download,fetch,fetch-4chan,fetch-hf,fetch-youtube,curate,merge,merge-ftan,subsample,normalize,mutate,dedup,split,export",
    "train": "train,finalize",
}
STEPS_ORDER = [
    "download", "fetch", "fetch-4chan", "fetch-hf", "fetch-youtube", "curate", "merge",
    "merge-ftan",
    "subsample",
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


def _has_parquets(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and bool(list(p.glob("*.parquet")))


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


def build_4chan_args(args) -> list[str]:
    a = ["--live" if args.chan_mode == "live" else "--archive"]
    if args.boards:
        a += ["--boards", args.boards]
    if args.chan_board:
        a += ["--board", args.chan_board]
    if args.chan_search:
        a += ["--search", args.chan_search]
    a += [
        "--max_rows", str(args.max_rows),
        "--min_words", str(args.min_words),
        "--clean_sample_frac", str(args.clean_sample_frac),
        "--delay", str(args.chan_delay),
        "--repeat", str(args.chan_repeats),
        "--repeat_delay", str(args.chan_repeat_delay),
        "--out", str(args.chan_out),
        "--seed", str(args.seed),
    ]
    if not args.require_latin:
        a += ["--no-require_latin"]
    return a


def build_youtube_args(args) -> list[str]:
    a = [
        "--api_key", args.yt_api_key,
        "--query", args.yt_query,
        "--max_videos", str(args.yt_max_videos),
        "--max_comments_per_video", str(args.yt_max_comments),
        "--min_words", str(args.min_words),
        "--clean_sample_frac", str(args.yt_clean_frac),
        "--max_rows", str(args.max_rows),
        "--out", str(args.yt_out),
        "--seed", str(args.seed),
    ]
    if not args.require_latin:
        a += ["--no-require_latin"]
    return a


def step_fetch(args) -> bool:
    if not (args.api or args.dump):
        _notice("no --api or --dump given, skipping reddit fetch")
        return False
    _run("07_reddit_fetch.py", build_fetch_args(args))
    return True


def step_fetch_4chan(args) -> bool:
    if not getattr(args, "4chan", False):
        _notice("no --4chan given, skipping 4chan fetch")
        return False
    _run("09_4chan_fetch.py", build_4chan_args(args))
    return True


def step_fetch_youtube(args) -> bool:
    if not args.youtube:
        _notice("no --youtube given, skipping youtube fetch")
        return False
    if not args.yt_api_key:
        _notice("--youtube needs --yt-api-key, skipping youtube fetch")
        return False
    _run("10_youtube_fetch.py", build_youtube_args(args))
    return True


def build_hf_args(args) -> list[str]:
    a = [
        "--dataset", args.hf_dataset,
        "--split", args.hf_split,
        "--text-col", args.hf_text_col or "text",
        "--source-col", args.hf_source_col or "board",
        "--time-col", args.hf_time_col or "timestamp",
        "--score-col", args.hf_score_col or "replies",
        "--id-col", args.hf_id_col or "no",
        "--max_rows", str(args.max_rows),
        "--min_words", str(args.min_words),
        "--clean_sample_frac", str(args.clean_sample_frac),
        "--out", str(args.hf_out),
        "--seed", str(args.seed),
    ]
    if args.hf_source_name:
        a += ["--source-name", args.hf_source_name]
    if args.hf_ftan_model:
        a += ["--ftan-model", args.hf_ftan_model,
              "--ftan-device", str(args.hf_ftan_device or ""),
              "--ftan-batch-size", str(args.hf_ftan_batch_size),
              "--ftan-threshold", str(args.hf_ftan_threshold),
              "--ftan-grey-low", str(args.hf_ftan_grey_low),
              "--ftan-grey-high", str(args.hf_ftan_grey_high),
              "--ftan-max-length", str(args.hf_ftan_max_length),
              "--checkpoint_every", str(args.hf_checkpoint_every)]
        if not args.hf_resume:
            a += ["--no-resume"]
    return a


def step_fetch_hf(args) -> bool:
    if not args.hf_dataset:
        _notice("no --hf-dataset given, skipping hf fetch")
        return False
    _run("11_hf_fetch.py", build_hf_args(args))
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
    if not any(_has_parquets(p) for p in args.raw.split(",")):
        _notice(f"no candidates in {args.raw}, skipping curate")
        return False
    _run("08_reddit_curate.py", build_curate_args(args))
    return True


def step_merge(args) -> bool:
    """Turn curated banks into data/raw/community.parquet (unified schema)."""
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
    out = RAW_DIR / "community.parquet"
    df.to_parquet(out, index=False)
    print(f"\n### merge: {len(df):,} community rows -> {out}")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    return True


def step_merge_ftan(args) -> bool:
    """Append FTAN-labeled candidates (label in (0,1)) to community.parquet.

    Reads candidates_*.parquet from hf_out (e.g. data/4chan/raw) and merges
    them into data/raw/community.parquet, preserving FTAN labels. Rows that
    still carry label -1 (unsure, not yet manually fixed) are excluded.
    """
    src = Path(args.hf_out)
    files = sorted(src.glob("candidates_*.parquet"))
    if not files:
        _notice(f"no FTAN candidates in {src}, skipping merge-ftan")
        return False
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        parts.append(df)
    cand = pd.concat(parts, ignore_index=True)
    before = len(cand)
    cand = cand[cand["label"].isin((0, 1))]
    after_label = len(cand)
    cand = cand.drop_duplicates(subset=["text"]).reset_index(drop=True)
    cand = cand.reindex(columns=UNIFIED_COLUMNS)
    if not len(cand):
        _notice(f"no confident FTAN rows (label in 0/1) in {src}; nothing to merge")
        return False

    out = RAW_DIR / "community.parquet"
    if out.exists():
        old = pd.read_parquet(out)
        merged = pd.concat([old, cand], ignore_index=True)
        merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
        rows = len(merged)
        added = rows - len(old)
    else:
        old = None
        merged = cand
        rows = len(merged)
        added = rows
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"\n### merge-ftan: {before:,} candidates -> "
          f"{after_label:,} confident (label 0/1) -> +{added:,} into {out}")
    print(merged.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    return True


def review_hf_candidates(args) -> None:
    """Stop (or regex-fallback) when FTAN labeling left label==-1 rows behind.

    Called right after the fetch-hf step. Scans hf_out for candidates with
    label -1 and either exits for manual review (default) or rewrites them
    using regex classification (--hf-fallback-regex).
    """
    if not args.hf_dataset or not args.hf_ftan_model:
        return
    src = Path(args.hf_out)
    files = sorted(src.glob("candidates_*.parquet"))
    if not files:
        return
    parts = [pd.read_parquet(p) for p in files]
    cand = pd.concat(parts, ignore_index=True)
    unsure = cand[cand["label"] == -1]
    if not len(unsure):
        return
    per_file = {p.name: int(pd.read_parquet(p)["label"].eq(-1).sum()) for p in files}
    print(f"\n### [make_dataset] {len(unsure):,} rows labeled -1 (model unsure):")
    for name, n in per_file.items():
        if n:
            print(f"  {name}: {n:,}")
    if args.hf_fallback_regex:
        from reddit_vocab import classify_regex
        txt = unsure["text"].tolist()
        cats = [classify_regex(t)["category"] for t in txt]
        fix = {t: (1 if c == "attack" else 0) for t, c in zip(txt, cats) if c != "grey"}
        drop = sum(1 for c in cats if c == "grey")
        print(f"  --hf-fallback-regex: attack->1, emotional/clean->0, "
              f"regex-grey dropped: {drop:,}")
        kept = 0
        for p in files:
            df = pd.read_parquet(p)
            m = df["label"] == -1
            mapped = df.loc[m, "text"].map(fix)
            df.loc[m, "label"] = mapped
            df = df[df["label"].isin((0, 1))].reset_index(drop=True)
            df.to_parquet(p, index=False)
            kept += len(df)
        print(f"  rewrote {len(files)} shards | kept {kept:,} rows | "
              f"dropped {drop:,} regex-grey")
        return
    csv = src / "manual_check.csv"
    if csv.exists():
        fixed = pd.read_csv(csv)
        fixed = fixed[fixed["label"].isin((0, 1))]
        if len(fixed):
            fix = dict(zip(fixed["text"].astype(str), fixed["label"].astype(int)))
            for p in files:
                df = pd.read_parquet(p)
                m = df["label"] == -1
                mapped = df.loc[m, "text"].astype(str).map(fix)
                df.loc[m, "label"] = mapped
                df = df[df["label"].isin((0, 1))].reset_index(drop=True)
                df.to_parquet(p, index=False)
            print(f"  applied {len(fixed):,} manual fixes from {csv.name}")
    unsure = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    unsure = unsure[unsure["label"] == -1]
    if not len(unsure):
        return
    unsure.to_csv(csv, index=False,
                  columns=["text", "origin_label", "label"])
    print(f"\n  wrote {csv} ({len(unsure):,} rows) for manual review.")
    print("  Fix the 'label' column to 0 or 1, then re-run the same command "
          "to continue. Rows you fixed in place in the parquet are kept as-is "
          "on resume; leftover -1 rows are excluded by merge-ftan.")
    raise SystemExit(0)


def _parse_source_caps(raw: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad --source_caps token {part!r}; expected source=target")
        src, cap = part.rsplit("=", 1)
        out[src.strip()] = int(cap.strip())
    return out


def _stratified_sample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Downsample `df` to exactly `cap` rows, preserving label ratios."""
    total = len(df)
    if total <= cap:
        return df
    counts = df["label"].value_counts()
    targets = (counts / total * cap).astype(int)
    remainder = cap - targets.sum()
    if remainder > 0:
        for label in (counts / total * cap - targets).sort_values(ascending=False).index[:remainder]:
            targets[label] += 1
    parts = []
    for label, n in targets.items():
        g = df[df["label"] == label]
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def step_subsample(args) -> bool:
    caps = _parse_source_caps(args.source_caps)
    if not caps:
        _notice("no --source_caps given, skipping subsample")
        return False
    path = RAW_DIR / "community.parquet"
    if not path.exists():
        _notice(f"{path} not found, skipping subsample")
        return False
    df = pd.read_parquet(path)
    before = len(df)
    parts = []
    skipped = []
    for src, cap in caps.items():
        sub = df[df["source"] == src]
        if len(sub) == 0:
            skipped.append((src, cap, 0))
            continue
        sampled = _stratified_sample(sub, cap, args.subsample_seed)
        parts.append(sampled)
        if len(sampled) < len(sub):
            print(f"  {src}: {len(sub):,} -> {len(sampled):,} (cap {cap:,})")
        else:
            skipped.append((src, cap, len(sub)))
    for src, cap, n in skipped:
        print(f"  {src}: {n:,} rows (cap {cap:,}, unchanged)")
    # keep sources not in caps as-is
    uncapped = df[~df["source"].isin(caps)]
    if len(uncapped):
        parts.append(uncapped)
    if not parts:
        _notice("nothing to write after subsample")
        return False
    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    out_df.to_parquet(path, index=False)
    print(f"  community.parquet: {before:,} -> {len(out_df):,} rows")
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

    # reddit fetch
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

    # 4chan fetch
    ap.add_argument("--4chan", action="store_true", help="fetch 4chan candidates")
    ap.add_argument("--chan-mode", choices=["live", "archive"], default="live")
    ap.add_argument("--boards", default=None, help="comma-separated 4chan boards (live)")
    ap.add_argument("--chan-board", default=None, help="single board for 4chan archive mode")
    ap.add_argument("--chan-search", default=None, help="body search term (4chan archive)")
    ap.add_argument("--chan_delay", type=float, default=1.0)
    ap.add_argument("--chan-repeats", type=int, default=0,
                    help="re-scan 4chan catalogs this many times (0 = off). "
                         "Use with --chan-repeat-delay to accumulate threads over time.")
    ap.add_argument("--chan-repeat-delay", type=float, default=300.0,
                    help="seconds between catalog re-scans when --chan-repeats > 0")
    ap.add_argument("--chan-threads-per-board", type=int, default=250,
                    help="max threads to fetch per board per scan")
    ap.add_argument("--chan_out", default="data/4chan/raw")

    # hf dataset fetch (e.g. ylelauta/pol-4chan-augmented)
    ap.add_argument("--hf-dataset", default=None, help="HF dataset id to load as candidates")
    ap.add_argument("--hf-split", default="train", help="split to load (default: train)")
    ap.add_argument("--hf-text-col", default=None)
    ap.add_argument("--hf-source-col", default=None)
    ap.add_argument("--hf-time-col", default=None)
    ap.add_argument("--hf-score-col", default=None)
    ap.add_argument("--hf-id-col", default=None)
    ap.add_argument("--hf-source-name", default=None)
    ap.add_argument("--hf_out", default="data/hf/raw")
    ap.add_argument("--hf-ftan-model", default=None, help="use FTAN to label rows (path to model)")
    ap.add_argument("--hf-ftan-device", default=None, help="cuda device for FTAN, e.g. 0")
    ap.add_argument("--hf-ftan-batch-size", type=int, default=256)
    ap.add_argument("--hf-ftan-threshold", type=float, default=0.5)
    ap.add_argument("--hf-ftan-grey-low", type=float, default=0.30,
                    help="FTAN p_off below this -> label 0 (confident clean)")
    ap.add_argument("--hf-ftan-grey-high", type=float, default=0.70,
                    help="FTAN p_off above this -> label 1 (confident offensive)")
    ap.add_argument("--hf-exit-on-unsure", action=argparse.BooleanOptionalAction, default=True,
                    help="after fetch-hf, stop if any label==-1 rows remain (default: on)")
    ap.add_argument("--hf-fallback-regex", action="store_true",
                    help="classify remaining label==-1 rows with regex instead of stopping")
    ap.add_argument("--hf-ftan-max-length", type=int, default=128)
    ap.add_argument("--hf-checkpoint-every", type=int, default=150_000,
                    help="rows between FTAN checkpoints")
    ap.add_argument("--hf-resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume FTAN from checkpoint/parquet if present (default: on)")

    # youtube fetch
    ap.add_argument("--youtube", action="store_true", help="fetch YouTube comments")
    ap.add_argument("--yt-api-key", default=None, help="YouTube Data API v3 key")
    ap.add_argument("--yt-query", default="cat videos")
    ap.add_argument("--yt-max-videos", type=int, default=20)
    ap.add_argument("--yt-max-comments", type=int, default=200)
    ap.add_argument("--yt-clean-frac", type=float, default=1.0)
    ap.add_argument("--yt_out", default="data/youtube/raw")

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

    # subsample (after merge, before normalize)
    ap.add_argument("--source_caps", default=None,
                    help="comma-separated source=target caps, e.g. "
                         "'4chan=800000,youtube=500000,reddit=500000'. "
                         "Sources not listed are kept as-is. Empty = no caps.")
    ap.add_argument("--subsample_seed", type=int, default=42)

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

    raw_dirs = [args.raw_out]
    ran = {"fetch": False, "fetch-4chan": False, "fetch-hf": False, "fetch-youtube": False}

    if "download" in steps:
        step_download(args)

    if "fetch" in steps:
        ran["fetch"] = step_fetch(args)
    if "fetch-4chan" in steps:
        ran["fetch-4chan"] = step_fetch_4chan(args)
    if "fetch-hf" in steps:
        ran["fetch-hf"] = step_fetch_hf(args)
    if "fetch-youtube" in steps:
        ran["fetch-youtube"] = step_fetch_youtube(args)

    if "fetch-hf" in steps and args.hf_ftan_model:
        review_hf_candidates(args)

    raw_dirs = [
        d for d, ok in ((args.raw_out, ran["fetch"] or _has_parquets(args.raw_out)),
                        (args.chan_out, ran["fetch-4chan"] or _has_parquets(args.chan_out)),
                        (args.hf_out, ran["fetch-hf"] or _has_parquets(args.hf_out)),
                        (args.yt_out, ran["fetch-youtube"] or _has_parquets(args.yt_out)))
        if ok
    ]
    if raw_dirs:
        args.raw = ",".join(raw_dirs)

    if "curate" in steps:
        step_curate(args)
    if "merge" in steps:
        step_merge(args)
    if "merge-ftan" in steps:
        step_merge_ftan(args)
    if "subsample" in steps:
        step_subsample(args)

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