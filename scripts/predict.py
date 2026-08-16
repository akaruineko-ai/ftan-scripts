"""Classify arbitrary text with a trained ftan-2.0 model.

Usage:
    echo "you are a fu4king 4ssh0le" | .venv/bin/python scripts/predict.py
    .venv/bin/python scripts/predict.py --model data/final/model --text "I love this"
    .venv/bin/python scripts/predict.py --model data/final/model -f texts.txt
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from common import FINAL_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(FINAL_DIR / "model" / "model"))
    ap.add_argument("--text", default=None, help="single text to classify")
    ap.add_argument("-f", "--file", default=None, help="read texts, one per line")
    ap.add_argument("--device", default=None, help="device id, e.g. 0 (auto if omitted)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    clf = pipeline("text-classification", model=model, tokenizer=tok, device=args.device)

    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [line.strip() for line in open(args.file, encoding="utf-8")
                 if line.strip()]
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    for t in texts:
        out = clf(t)[0]
        print(f"{out['label']:<10} conf={out['score']:.3f} | {t}")


if __name__ == "__main__":
    main()
