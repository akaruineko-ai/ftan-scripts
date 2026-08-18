"""Score one or more ftan moderation models on a benchmark parquet.

The benchmark parquet (see make_benchmark.py) holds held-out test rows tagged
with a `benchmark_split` column (`test` / `test_obfuscated`) plus the full
schema, so scoring is reported overall and sliced by benchmark_split, by
`mutated` (plain vs obfuscated), and by `source`.

Models can be local directories or Hugging Face Hub ids (any id accepted by
`AutoModelForSequenceClassification.from_pretrained`).

Multi-class models (more than the two ftan labels clean/offensive) are reduced
to binary predictions: every class that is not a clean synonym is mapped to
1/offensive. This is what lets 9-class moderation models such as
`KoalaAI/Text-Moderation` (S/H/V/HR/SH/S3/H2/V2/OK) be scored on the ftan
benchmark — the `OK` class becomes 0/clean, all harmful categories become 1.

Usage:
    .venv/bin/python scripts/benchmark.py \
        --model data/final/model/model \
        --model data/final/model/checkpoints/checkpoint-300000 \
        --model user/moderation-model

    # multi-class model with automatic reduction (OK -> clean, rest -> offensive)
    .venv/bin/python scripts/benchmark.py --model KoalaAI/Text-Moderation

    # or force an explicit class -> binary mapping (JSON object over class names)
    .venv/bin/python scripts/benchmark.py --model KoalaAI/Text-Moderation \
        --label_map '{"OK": 0, "S": 1, "H": 1, "V": 1, "HR": 1, "SH": 1, "S3": 1, "H2": 1, "V2": 1}'

Outputs:
    data/final/benchmark/results.json   metrics per model (all slices)
    a printed comparison table
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import FINAL_DIR

POS_LABEL = 1
METRICS = ["accuracy", "precision", "recall", "f1"]

CLEAN_SYNONYMS = {"ok", "clean", "normal", "benign", "non-offensive",
                  "not offensive", "neutral"}
OFFENSIVE_SYNONYMS = {"offensive", "toxic", "hate", "hateful", "abusive",
                      "explicit"}


def _metrics(y_true, y_pred) -> dict:
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _predict(model, tokenizer, texts, batch_size, max_length, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="infer",
                      unit="batch", leave=False):
            enc = tokenizer(
                texts[i:i + batch_size], truncation=True,
                max_length=max_length, padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return np.asarray(preds)


def _label_map(model, override=None):
    """Reduce argmax indices to binary 0/1 labels from config.id2label.

    Returns a dict {index: 0|1} when a reliable mapping can be built, else None
    (the caller then treats argmax indices as the labels, which is valid for
    binary ftan models whose id2label already is 0/1).

    For multi-class moderation models (e.g. KoalaAI/Text-Moderation's
    S/H/V/HR/SH/S3/H2/V2/OK taxonomy) every class that is not a clean synonym
    is reduced to 1/offensive, so the 9-class model can be scored on the binary
    ftan benchmark. `override` (a dict of class names -> 0/1) takes precedence
    for the names it covers; any class it misses falls back to the heuristic.
    """
    cfg = getattr(model, "config", None)
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None
    if not id2label:
        return None
    mapping = {}
    n_labels = len(id2label)
    for idx, name in id2label.items():
        idx = int(idx)
        if override is not None and str(name) in override:
            mapping[idx] = int(override[str(name)])
            continue
        try:
            mapping[idx] = int(name)  # config already stores numeric labels
            continue
        except (ValueError, TypeError):
            pass
        low = str(name).lower()
        if low in CLEAN_SYNONYMS:
            mapping[idx] = 0
        elif low in OFFENSIVE_SYNONYMS or n_labels > 2:
            mapping[idx] = 1
    return mapping or None


def _describe_label_map(model, label_map):
    """Human-readable {class name -> 0/1} view of a label mapping."""
    id2label = getattr(getattr(model, "config", None), "id2label", {})
    return {str(id2label.get(str(k), k)): v for k, v in sorted(label_map.items())}


def _apply_label_map(preds, label_map):
    if not label_map:
        return preds
    return np.asarray([label_map.get(int(p), p) for p in preds])


def _fmt(m) -> str:
    return f"acc={m['accuracy']:.4f} p={m['precision']:.4f} r={m['recall']:.4f} f1={m['f1']:.4f}"


def _fmt_or(groups, key) -> str:
    m = groups.get(key)
    return _fmt(m) if m else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="model to score: a local dir or a Hugging Face Hub id "
                         "(repeat for multiple models)")
    ap.add_argument("--benchmark", default=str(FINAL_DIR / "benchmark" / "benchmark.parquet"),
                    help="benchmark parquet (default: data/final/benchmark/benchmark.parquet)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512,
                    help="token truncation length for scoring")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="score only the first N rows (for quick smoke runs)")
    ap.add_argument("--label_map", default=None,
                    help="JSON object mapping model class names to 0/1, e.g. "
                         '{"OK": 0, "S": 1, "H": 1, ...} for '
                         "KoalaAI/Text-Moderation. When omitted a heuristic is "
                         "used: clean synonyms (OK/clean/normal/benign/neutral) "
                         "-> 0, every other class -> 1.")
    ap.add_argument("--device", default=None,
                    help="device id, e.g. 0 (auto if omitted)")
    ap.add_argument("--out", default=str(FINAL_DIR / "benchmark" / "results.json"))
    args = ap.parse_args()

    df = pd.read_parquet(args.benchmark)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    labels = df["label"].to_numpy()
    texts = df["text"].tolist()
    print(f"benchmark rows: {len(df):,} (offensive {int((labels == POS_LABEL).sum()):,})")

    label_override = None
    if args.label_map:
        label_override = json.loads(args.label_map)
        print(f"label override: {label_override}")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu" if str(args.device).lower() == "cpu" else int(args.device)
    print(f"device: {device}\n")

    results = {}
    for model_id in args.model:
        name = os.path.basename(model_id.rstrip("/")) \
            if os.path.isdir(model_id) else model_id
        print(f"[{name}] loading {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        n_labels = getattr(model.config, "num_labels", 2)
        model.to(device)
        label_map = _label_map(model, label_override)
        if label_map:
            print(f"  label reduction: {_describe_label_map(model, label_map)}")
        else:
            print("  no label reduction applied; argmax indices used as labels")
        if n_labels > 2:
            print(f"  model has {n_labels} classes; non-clean classes map to 1/offensive")
        preds = _apply_label_map(
            _predict(model, tokenizer, texts, args.batch_size,
                     args.max_length, device),
            label_map,
        )

        entry = {"overall": _metrics(labels, preds)}
        for group, col in (("by_benchmark_split", "benchmark_split"),
                           ("by_mutated", "mutated"),
                           ("by_source", "source")):
            entry[group] = {}
            for value, ix in df.groupby(col).groups.items():
                entry[group][str(value)] = _metrics(labels[ix], preds[ix])
        results[name] = entry
        print(f"  overall: {_fmt(entry['overall'])}")
        print(f"  test:            {_fmt_or(entry['by_benchmark_split'], 'test')}")
        print(f"  test_obfuscated: {_fmt_or(entry['by_benchmark_split'], 'test_obfuscated')}")
        print(f"  plain:   {_fmt_or(entry['by_mutated'], '0')}")
        print(f"  mutated: {_fmt_or(entry['by_mutated'], '1')}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
