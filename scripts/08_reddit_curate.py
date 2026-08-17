"""Step 8: turn raw reddit candidates into the three dataset banks.

Banks (from the ftan-2.0 expansion plan, regex-authoritative with FTAN for
the grey zone):

    bank_a  attack    - profanity/insult aimed at a person      -> label 1
    bank_b  emotional - profanity with no addressee ("fuck this
                        weather") -> the hard negatives that teach a
                        classifier that swearing != insulting    -> label 0
    bank_c  clean     - no profanity, no insult                 -> label 0
    manual_check      - grey rows the FTAN model cannot decide (conf in
                        [--grey_low, --grey_high]) OR verification
                        disagreements -> review by hand

FTAN (data/final/model/model) is only invoked on:
  * grey rows (insult word, no 2nd-person address) - confidence splits them
    into bank_a / bank_c / manual_check;
  * an optional random fraction of bank_a / bank_c when --verify_frac > 0,
    to catch regex misses (low-conf attacks, high-conf clean).

Usage:
    .venv/bin/python scripts/08_reddit_curate.py \
        --raw data/reddit/raw --out data/reddit/banks \
        --target_a 1000000 --target_b 1000000 --target_c 1000000
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path

from common import clean_text
from reddit_vocab import classify_regex

BANK_COLUMNS = [
    "text", "label", "source", "origin_label", "split_origin",
    "subreddit", "created_utc", "score", "category", "ftan_conf",
]


def _to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=BANK_COLUMNS)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).map(clean_text)
    return df[df.text != ""].reset_index(drop=True)


class FtanScorer:
    """Lazy batched wrapper around the ftan-2.0 model (GPU by default)."""

    def __init__(self, model_dir: str, device: str | None, batch_size: int):
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self._tok = None
        self._model = None
        self._dev = None

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        dev = self.device
        if dev is not None and str(dev).isdigit():
            dev = int(dev)
        self._dev = dev if dev is not None else "cpu"
        self._model.to(self._dev)
        self._model.eval()

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        """Return offensive probability (1 = offensive) for each text."""
        if not texts:
            return np.array([], dtype=np.float32)
        if self._model is None:
            self._load()
        scores = []
        with tqdm(total=len(texts), desc="FTAN", unit="row", leave=False) as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                enc = self._tok(batch, padding=True, truncation=True, max_length=128,
                                return_tensors="pt")
                enc = {k: v.to(self._dev) for k, v in enc.items()}
                with torch.amp.autocast("cuda"):
                    logits = self._model(**enc).logits
                probs = logits.softmax(dim=-1).cpu().numpy()
                for p in probs:
                    off_p = float(p[1] if len(p) > 1 else p[0])
                    scores.append(off_p)
                pbar.update(len(batch))
        return np.asarray(scores, dtype=np.float32)


def load_raw(raw: str) -> pd.DataFrame:
    """Load candidates from one or more dirs/files (comma-separated)."""
    parts = [p for p in (s.strip() for s in raw.split(",")) if p]
    files = []
    for part in parts:
        path = Path(part)
        if path.is_dir():
            files += sorted(path.glob("*.parquet"))
        else:
            files.append(path)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no parquet files found in {raw}")
    frames = []
    for f in files:
        print(f"  loading {f.name}")
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"loaded {len(df):,} unique candidates")
    return df


def bucket_rows(df: pd.DataFrame, rng) -> dict:
    buckets = {k: [] for k in ("attack", "emotional", "grey", "clean")}
    for text in df["text"].tolist():
        info = classify_regex(text)
        buckets[info["category"]].append(text)
    for k, v in buckets.items():
        print(f"  regex {k:<10}: {len(v):,}")
    return buckets


def _meta_lookup(df: pd.DataFrame) -> dict:
    meta = {}
    cols = [c for c in ("source", "subreddit", "created_utc", "score") if c in df.columns]
    if not cols:
        return meta
    for t, values in zip(df["text"], df[cols].itertuples(index=False, name=None)):
        meta[t] = dict(zip(cols, values))
    return meta


def _row(text: str, label: int, origin: str, category: str, meta: dict) -> dict:
    m = meta.get(text, {})
    return {
        "text": text, "label": label, "source": m.get("source", "reddit"),
        "origin_label": origin, "split_origin": "train",
        "subreddit": m.get("subreddit", ""),
        "created_utc": int(m.get("created_utc") or 0),
        "score": int(m.get("score") or 0),
        "category": category, "ftan_conf": np.nan,
    }


def assign_grey(buckets: dict, ftan: FtanScorer, grey_low: float, grey_high: float,
                meta: dict) -> tuple[list, list, list]:
    """Return (bank_a_rows, bank_c_rows, manual_rows) from grey + FTAN conf."""
    grey = buckets["grey"]
    if not grey:
        return [], [], []
    print(f"  running FTAN on {len(grey):,} grey rows ...")
    probs = ftan.score(grey)
    a, c, manual = [], [], []
    for text, p in zip(grey, probs):
        if p >= grey_high:
            a.append(_grey_row(text, p, label=1, origin="grey->attack", meta=meta))
        elif p <= grey_low:
            c.append(_grey_row(text, p, label=0, origin="grey->clean", meta=meta))
        else:
            manual.append(_grey_row(text, p, label=-1, origin="grey-manual", meta=meta))
    print(f"  grey -> attack={len(a):,} clean={len(c):,} manual={len(manual):,}")
    return a, c, manual


def _grey_row(text: str, prob: float, label: int, origin: str, meta: dict) -> dict:
    m = meta.get(text, {})
    return {
        "text": text, "label": label, "source": m.get("source", "reddit"),
        "origin_label": origin, "split_origin": "train",
        "subreddit": m.get("subreddit", ""),
        "created_utc": int(m.get("created_utc") or 0),
        "score": int(m.get("score") or 0),
        "category": "grey", "ftan_conf": float(prob),
    }


def verify_banks(rows: list[dict], ftan: FtanScorer, verify_frac: float,
                 low_conf_attack: float, high_conf_clean: float, rng, label: int):
    """Sample a fraction of a bank, re-check with FTAN, pull disagreements out."""
    if not rows or verify_frac <= 0:
        return rows, []
    k = max(1, int(len(rows) * verify_frac))
    idx = sorted(rng.sample(range(len(rows)), k))
    sample_texts = [rows[i]["text"] for i in idx]
    print(f"  verifying {len(idx):,} rows of label-{label} bank ...")
    probs = ftan.score(sample_texts)
    pulled = []
    for i, p in zip(idx, probs):
        keep = (p >= low_conf_attack) if label == 1 else (p <= high_conf_clean)
        if not keep:
            row = dict(rows[i])
            row["label"] = -1  # needs human decision
            row["ftan_conf"] = float(p)
            row["origin_label"] = "verify-disagree"
            pulled.append(row)
            rows[i] = None
    rows = [r for r in rows if r is not None]
    print(f"  pulled {len(pulled):,} rows into manual_check")
    return rows, pulled


def sample_to_target(rows: list[dict], target: int, rng) -> list[dict]:
    if target is None:
        return rows
    if len(rows) > target:
        rows = rng.sample(rows, target)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/reddit/raw")
    ap.add_argument("--out", default="data/reddit/banks")
    ap.add_argument("--model", default="data/final/model/model")
    ap.add_argument("--device", default=None, help="cuda device id, e.g. 0 (auto)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--target_a", type=int, default=1_000_000)
    ap.add_argument("--target_b", type=int, default=1_000_000)
    ap.add_argument("--target_c", type=int, default=1_000_000)
    ap.add_argument("--grey_low", type=float, default=0.30,
                    help="grey rows with FTAN conf below this -> clean (label 0)")
    ap.add_argument("--grey_high", type=float, default=0.70,
                    help="grey rows with FTAN conf above this -> attack (label 1)")
    ap.add_argument("--verify_frac", type=float, default=0.0,
                    help="fraction of bank_a/bank_c re-checked with FTAN (0 = off)")
    ap.add_argument("--verify_attack_conf", type=float, default=0.40,
                    help="regex attacks with FTAN conf below this -> manual")
    ap.add_argument("--verify_clean_conf", type=float, default=0.90,
                    help="regex clean rows with FTAN conf above this -> manual")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.raw)
    buckets = bucket_rows(df, rng)
    meta = _meta_lookup(df)

    ftan = FtanScorer(args.model, args.device, args.batch_size)

    # bank_a: regex attacks, optionally verified
    bank_a = [_row(t, 1, "attack", "attack", meta) for t in buckets["attack"]]
    bank_a, pulled_a = verify_banks(
        bank_a, ftan, args.verify_frac, args.verify_attack_conf, 0.0, rng, label=1
    )

    # bank_b: emotional profanity (no addressee) -> label 0, no FTAN needed
    bank_b = [_row(t, 0, "emotional", "emotional", meta) for t in buckets["emotional"]]

    # bank_c: regex-clean rows, optionally verified
    bank_c = [_row(t, 0, "clean", "clean", meta) for t in buckets["clean"]]
    bank_c, pulled_c = verify_banks(
        bank_c, ftan, args.verify_frac, 0.0, args.verify_clean_conf, rng, label=0
    )

    # grey zone -> FTAN decides
    grey_a, grey_c, grey_manual = assign_grey(buckets, ftan, args.grey_low, args.grey_high, meta)
    bank_a += grey_a
    bank_c += grey_c

    manual = grey_manual + pulled_a + pulled_c

    print(f"\npre-sampling counts:")
    print(f"  bank_a (attack,   label 1): {len(bank_a):,}")
    print(f"  bank_b (emotional,label 0): {len(bank_b):,}")
    print(f"  bank_c (clean,    label 0): {len(bank_c):,}")
    print(f"  manual_check:               {len(manual):,}")

    bank_a = sample_to_target(bank_a, args.target_a, rng)
    bank_b = sample_to_target(bank_b, args.target_b, rng)
    bank_c = sample_to_target(bank_c, args.target_c, rng)

    for name, rows in (("bank_a", bank_a), ("bank_b", bank_b), ("bank_c", bank_c),
                       ("manual_check", manual)):
        out_path = out_dir / f"{name}.parquet"
        _to_df(rows).to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(rows):,} rows)")

    combined = _to_df(bank_a + bank_b + bank_c)
    combined = combined[combined.label != -1]
    print(f"\ncombined community dataset: {len(combined):,} rows "
          f"(pos={int((combined.label == 1).sum()):,} neg={int((combined.label == 0).sum()):,})")
    combined.to_parquet(out_dir / "community.parquet", index=False)


if __name__ == "__main__":
    main()