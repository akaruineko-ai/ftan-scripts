"""Step 11: load any Hugging Face dataset and convert it to the candidate schema.

Use this when a public HF dataset already has the data you want (4chan archives,
Reddit dumps, etc.) and you just need to regex-prefilter it into the same
candidate format the other fetchers produce.

Examples:

    # 4chan /pol/ with Perspective toxicity scores (134M posts)
    .venv/bin/python scripts/11_hf_fetch.py \
        --dataset ylelauta/pol-4chan-augmented \
        --text-col com --source-col board --time-col time --score-col replies \
        --max_rows 1000000 --out data/4chan/raw

    # 4chan /pol/ whole dataset (4M rows, train split)
    .venv/bin/python scripts/11_hf_fetch.py \
        --dataset fuzzy-g/4chan_pol_whole_ds --split train \
        --text-col text --source-col board --time-col timestamp \
        --max_rows 1000000 --out data/4chan/raw

    # Reddit from an HF dataset
    .venv/bin/python scripts/11_hf_fetch.py \
        --dataset some-user/reddit-dump \
        --text-col body --source-col subreddit --time-col created_utc --score-col score \
        --out data/reddit/raw

Output: same candidate schema as the other fetchers
    (text, source, subreddit, created_utc, score, id,
     category, profanity, insult, attack)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from candidate_common import Sink, _is_latin
from common import clean_text, UNIFIED_COLUMNS
from reddit_vocab import prefilter_keep

DEFAULT_TEXT = "text"
DEFAULT_SOURCE = "board"
DEFAULT_TIME = "timestamp"
DEFAULT_SCORE = "replies"
DEFAULT_ID = "no"


def _coerce_ts(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v)
    if s.isdigit():
        return int(s)
    # ISO-8601 fallback
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def _row_from_hf(r: dict, text_col: str, source_col: str, time_col: str,
                  score_col: str, id_col: str, default_source: str) -> dict | None:
    body = str(r.get(text_col) or "").strip()
    if not body:
        return None
    return {
        "text": body,
        "source": default_source,
        "subreddit": str(r.get(source_col) or ""),
        "created_utc": _coerce_ts(r.get(time_col)),
        "score": int(r.get(score_col) or 0),
        "id": str(r.get(id_col) or ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. ylelauta/pol-4chan-augmented")
    ap.add_argument("--split", default="train", help="split to load (default: train)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--source-col", default=None)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--source-name", default=None,
                    help="value for the `source` column (default: dataset name)")
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--clean_sample_frac", type=float, default=0.05)
    ap.add_argument("--require_latin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_rows", type=int, default=2_000_000)
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False,
                    help="stream from the hub (slow per-row). Default: download & use Arrow (fast)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/hf/raw")
    ap.add_argument("--flush_rows", type=int, default=500_000)
    ap.add_argument("--checkpoint_every", type=int, default=150_000,
                    help="rows processed between checkpoints (FTAN mode only)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume from checkpoint/parquet shards if present (default: on)")
    # FTAN labeling mode
    ap.add_argument("--ftan-model", default=None,
                    help="path to a ftan-2.0 model dir (e.g. data/final/model/model). "
                         "When given, FTAN labels every row as offensive/clean instead of regex.")
    ap.add_argument("--ftan-device", default=None, help="cuda device for FTAN, e.g. 0")
    ap.add_argument("--ftan-batch-size", type=int, default=1024)
    ap.add_argument("--ftan-max-length", type=int, default=128,
                    help="max token length for FTAN truncation (default 128; shorter = faster)")
    ap.add_argument("--ftan-threshold", type=float, default=0.6,
                    help="offensive probability above this -> label 1 (default 0.6)")
    ap.add_argument("--ftan-grey-low", type=float, default=0.30,
                    help="p_off below this -> label 0 (clean, confident)")
    ap.add_argument("--ftan-grey-high", type=float, default=0.70,
                    help="p_off above this -> label 1 (offensive, confident)")
    args = ap.parse_args()

    text_col = args.text_col or DEFAULT_TEXT
    source_col = args.source_col or DEFAULT_SOURCE
    time_col = args.time_col or DEFAULT_TIME
    score_col = args.score_col or DEFAULT_SCORE
    id_col = args.id_col or DEFAULT_ID
    source_name = args.source_name or args.dataset.split("/")[-1]

    print(f"loading {args.dataset} [{args.split}] (streaming={args.streaming}) ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    first = next(iter(ds))
    available = sorted(first.keys())
    print(f"  available columns: {available}")
    for col in (text_col, source_col, time_col, score_col, id_col):
        if col not in available:
            print(f"  WARNING: column {col!r} not found; will use empty/0 defaults")

    if not args.streaming and args.ftan_model and args.max_rows is not None:
        total = len(ds)
        if total > args.max_rows:
            ds = ds.shuffle(seed=args.seed).select(range(args.max_rows))
            print(f"  random subset: {total:,} -> {len(ds):,} rows (--max_rows)")

    out_dir = Path(args.out)
    sink = Sink(out_dir, flush_rows=args.flush_rows)
    rng = __import__("random").Random(args.seed)

    use_ftan = args.ftan_model is not None

    ckpt_path = out_dir / "checkpoint.json"
    resume_n = 0
    skip_digests: set[int] | None = None
    if use_ftan and args.resume:
        _sig = {"dataset": args.dataset, "max_rows": args.max_rows,
                "seed": args.seed, "threshold": args.ftan_threshold,
                "max_length": args.ftan_max_length,
                "grey_low": args.ftan_grey_low, "grey_high": args.ftan_grey_high}
        _match = False
        if ckpt_path.exists():
            try:
                _prev = json.loads(ckpt_path.read_text())
                _match = all(_prev.get(k) == v for k, v in _sig.items())
            except Exception:
                _match = False
        if ckpt_path.exists() and _match:
            loaded = sink.load_existing()
            resume_n = int(_prev.get("rows_processed", 0))
            skip_digests = set(sink.seen)
            print(f"  RESUME: {resume_n:,} rows processed, {loaded:,} rows on disk; "
                  f"skipping already-kept rows")
        elif ckpt_path.exists():
            print(f"  WARNING: checkpoint at {ckpt_path} doesn't match current run "
                  f"(dataset/seed/threshold); starting fresh. Use a different --out if this "
                  f"is intentional.")
        elif not ckpt_path.exists() and sink.load_existing() > 0:
            loaded = sink.kept
            resume_n = loaded
            skip_digests = set(sink.seen)
            print(f"  RESUME (shards only): {loaded:,} rows on disk; skipping already-kept rows")

    if use_ftan:
        print(f"FTAN mode: loading model from {args.ftan_model} "
              f"(threshold={args.ftan_threshold}, device={args.ftan_device or 'auto'})")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.ftan_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.ftan_model)
        dev = args.ftan_device
        if dev is not None and str(dev).isdigit():
            dev = int(dev)
        if dev is None:
            dev = "cpu"
        model.to(dev)
        model.eval()
        if isinstance(dev, int) and dev >= 0:
            torch.backends.cuda.matmul.allow_tf32 = True
        print(f"  device={dev} | batch={args.ftan_batch_size}")

    n = 0
    t0 = time.time()

    if use_ftan:
        import torch.utils.data as tdata
        from itertools import islice

        text_col_, source_col_, time_col_, score_col_, id_col_, source_name_ = (
            text_col, source_col, time_col, score_col, id_col, source_name)

        def _ftan_gen():
            info = tdata.get_worker_info()
            it = ds
            if info is not None and info.num_workers > 1:
                if args.streaming:
                    it = islice(it, info.id, None, info.num_workers)
                else:
                    it = ds.shard(num_shards=info.num_workers, index=info.id)
            skip = skip_digests
            rows = []
            for r in it:
                row = _row_from_hf(r, text_col_, source_col_, time_col_,
                                   score_col_, id_col_, source_name_)
                if row is None:
                    continue
                body = clean_text(str(row.get("text", "")).strip())
                if not body or len(body.split()) < args.min_words:
                    continue
                if args.require_latin and not _is_latin(body):
                    continue
                if skip is not None:
                    digest = int.from_bytes(hashlib.md5(body.encode("utf-8")).digest()[:8], "big")
                    if digest in skip:
                        continue
                rows.append((len(body.split()), body, row))
            rows.sort(key=lambda x: x[0])
            for _, body, row in rows:
                yield body, row

        class _FtanDS(tdata.IterableDataset):
            def __iter__(self):
                return _ftan_gen()

        def _ftan_collate(batch):
            texts = [b[0] for b in batch]
            rows = [b[1] for b in batch]
            enc = tok(texts, padding="longest", truncation=True, max_length=args.ftan_max_length,
                      return_tensors="pt")
            enc = {k: v.to(torch.int32) for k, v in enc.items()}
            return enc, texts, rows

        pbar = tqdm(total=args.max_rows, desc="FTAN", unit="row", leave=True,
                    initial=resume_n,
                    bar_format="{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        _loader = tdata.DataLoader(
            _FtanDS(),
            batch_size=args.ftan_batch_size,
            num_workers=min(8, max(1, os.cpu_count() - 1)),
            pin_memory=True,
            collate_fn=_ftan_collate,
        )
        last_ckpt = resume_n
        n = resume_n
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for enc, batch_texts, batch_rows in _loader:
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = logits.softmax(dim=-1).float().cpu().numpy()
                for body, row, p in zip(batch_texts, batch_rows, probs):
                    p_off = float(p[1] if len(p) > 1 else p[0])
                    if p_off >= args.ftan_grey_high:
                        label = 1
                    elif p_off <= args.ftan_grey_low:
                        label = 0
                    else:
                        label = -1
                    row.update({
                        "text": body,
                        "label": label,
                        "source": source_name,
                        "origin_label": f"ftan_{p_off:.3f}",
                        "split_origin": "train",
                    })
                    sink.add(row)
                n += len(batch_rows)
                pbar.update(len(batch_rows))
                if n - last_ckpt >= args.checkpoint_every:
                    sink.flush()
                    ckpt_path.write_text(json.dumps({
                        "rows_processed": n, "kept": sink.kept, "shard": sink.shard,
                        "dataset": args.dataset, "max_rows": args.max_rows,
                        "seed": args.seed, "threshold": args.ftan_threshold,
                        "max_length": args.ftan_max_length,
                        "grey_low": args.ftan_grey_low, "grey_high": args.ftan_grey_high,
                    }))
                    last_ckpt = n
                    print(f"  checkpoint @ {n:,} rows | kept {sink.kept:,} | shard {sink.shard}", flush=True)
                if n >= args.max_rows:
                    break
        pbar.close()
        if use_ftan and ckpt_path is not None:
            sink.flush()
            ckpt_path.write_text(json.dumps({
                "rows_processed": n, "kept": sink.kept, "shard": sink.shard,
                "dataset": args.dataset, "max_rows": args.max_rows,
                "seed": args.seed, "threshold": args.ftan_threshold,
                "max_length": args.ftan_max_length,
                "grey_low": args.ftan_grey_low, "grey_high": args.ftan_grey_high,
            }))
    else:
        pbar = tqdm(desc="scanning", unit="row", leave=False)
        for r in ds:
            row = _row_from_hf(r, text_col, source_col, time_col, score_col, id_col, source_name)
            if row is None:
                continue
            body = str(row.get("text", "")).strip()
            if not body:
                continue
            if len(body.split()) < args.min_words:
                continue
            if args.require_latin and not _is_latin(body):
                continue
            info, keep = prefilter_keep(body, args.clean_sample_frac, rng)
            if keep:
                row.update({
                    "text": clean_text(body),
                    "category": info["category"],
                    "profanity": int(info["profanity"]),
                    "insult": int(info["insult"]),
                    "attack": int(info["attack"]),
                })
                sink.add(row)
            n += 1
            if sink.kept >= args.max_rows:
                print("  reached --max_rows, stopping")
                break
            pbar.update(1)
            pbar.set_postfix(kept=sink.kept)
            if n % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"  scanned {n:,} rows | kept {sink.kept:,} | {n/max(elapsed,1):.0f} rows/s")
        pbar.close()
    elapsed = time.time() - t0
    print(f"  scanned {n:,} rows | kept {sink.kept:,} in {elapsed:.1f}s")

    sink.close()
    print(f"\nkept {sink.kept:,} rows | dropped {sink.dropped_dup:,} exact dups | shards={sink.shard}")
    if sink.kept:
        parts = [pd.read_parquet(p) for p in sorted(out_dir.glob("candidates_*.parquet"))]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if use_ftan:
            print(f"  FTAN labels: offensive={int((df.label==1).sum()):,} "
                  f"clean={int((df.label==0).sum()):,} "
                  f"manual(-1)={int((df.label==-1).sum()):,}")
            if int((df.label == -1).sum()):
                print("  NOTE: rows with label -1 (model unsure) are written to "
                      "manual_check.csv for review; fix them to 0/1 and re-run "
                      "to continue.")
        else:
            print(df.category.value_counts().to_string())
        print(f"  -> {len(df):,} rows in {out_dir}")


if __name__ == "__main__":
    main()
