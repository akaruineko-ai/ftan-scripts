"""Classify arbitrary text with a trained ftan-2.0 model.

Usage:
    echo "you are a fu4king 4ssh0le" | .venv/bin/python scripts/predict.py
    .venv/bin/python scripts/predict.py --model data/exps/final/model --text "I love this"
    .venv/bin/python scripts/predict.py --model data/exps/final/model -f texts.txt
    .venv/bin/python scripts/predict.py --model data/exps/final/model --threshold 0.5

Scoring mode is auto-detected from the saved model: multi-label models
(`problem_type="multi_label_classification"`) print every label whose sigmoid
score clears --threshold; binary models print the single top label + confidence.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import FINAL_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(FINAL_DIR / "model" / "model"))
    ap.add_argument("--text", default=None, help="single text to classify")
    ap.add_argument("-f", "--file", default=None, help="read texts, one per line")
    ap.add_argument("--device", default=None, help="device id, e.g. 0 (auto if omitted)")
    ap.add_argument("--multi_label", action=argparse.BooleanOptionalAction, default=None,
                    help="force multi-label vs binary scoring; auto-detected from "
                         "the model when omitted")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="sigmoid cutoff for activating multi-labels")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    multi = args.multi_label if args.multi_label is not None else (
        getattr(model.config, "problem_type", None) == "multi_label_classification"
    )

    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [line.strip() for line in open(args.file, encoding="utf-8")
                 if line.strip()]
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    id2label = model.config.id2label or {}

    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True,
                      max_length=512).to(device)
            logits = model(**enc).logits
            if multi:
                probs = torch.sigmoid(logits).squeeze(0)
                hits = [(id2label.get(i, str(i)), float(p))
                        for i, p in enumerate(probs) if p >= args.threshold]
                tags = " ".join(f"{lab}={score:.3f}" for lab, score in hits) \
                    if hits else "(none)"
                print(f"{tags} | {t}")
            else:
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                i = int(probs.argmax())
                print(f"{id2label.get(i, str(i)):<10} conf={probs[i]:.3f} | {t}")


if __name__ == "__main__":
    main()
