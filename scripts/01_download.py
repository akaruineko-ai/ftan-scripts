"""Step 1: Download and normalize all source datasets into raw parquet files.

Each source produces a parquet with the unified schema:
    text, label, source, origin_label, split_origin
Outputs: data/raw/{source}.parquet

Sources:
    jigsaw     - tcapelle/jigsaw-toxic-comment-classification-challenge
    davidson   - contemmcm/hate-speech-and-offensive-language
    toxigen    - toxigen/toxigen-data
    hatexplain - Hate-speech-CNERG/hatexplain
    wikipedia  - wikimedia/wikipedia (20231101.en, capped sample, clean filler)
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd
from datasets import load_dataset

from common import RAW_DIR, clean_text

pd.set_option("future.no_silent_downcasting", True)


def _df(rows, columns):
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# Jigsaw toxic comment (Wikipedia comments)
# --------------------------------------------------------------------------- #
def load_jigsaw():
    rows = []
    for split, out_split in (("train", "train"), ("balanced_test", "test")):
        ds = load_dataset(
            "tcapelle/jigsaw-toxic-comment-classification-challenge",
            split=split,
        )
        cols = list(ds.features)
        for r in ds:
            flags = [c for c in ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate") if r[c] == 1]
            rows.append({
                "text": clean_text(r["comment_text"]),
                "label": 1 if flags else 0,
                "source": "jigsaw",
                "origin_label": "+".join(flags) if flags else "clean",
                "split_origin": out_split,
            })
    df = _df(rows, ["text", "label", "source", "origin_label", "split_origin"])
    df = df[df.text != ""]
    df.to_parquet(RAW_DIR / "jigsaw.parquet", index=False)
    print(f"jigsaw: {len(df):,} rows | pos={int((df.label == 1).sum()):,} neg={int((df.label == 0).sum()):,}")


# --------------------------------------------------------------------------- #
# Davidson et al. hate speech / offensive language (tweets)
# --------------------------------------------------------------------------- #
def load_davidson():
    ds = load_dataset("contemmcm/hate-speech-and-offensive-language", split="train")
    rows = []
    for r in ds:
        cls = int(r["label"])
        rows.append({
            "text": clean_text(r["text"]),
            "label": 1 if cls in (0, 1) else 0,  # hate + offensive vs neither
            "source": "davidson",
            "origin_label": {0: "hate", 1: "offensive", 2: "neither"}[cls],
            "split_origin": "train",
        })
    df = _df(rows, ["text", "label", "source", "origin_label", "split_origin"])
    df = df[df.text != ""]
    df.to_parquet(RAW_DIR / "davidson.parquet", index=False)
    print(f"davidson: {len(df):,} rows | pos={int((df.label == 1).sum()):,} neg={int((df.label == 0).sum()):,}")


# --------------------------------------------------------------------------- #
# ToxiGen (annotated + machine-generated)
# --------------------------------------------------------------------------- #
TOXIGEN_THRESHOLD = 3.5


def load_toxigen():
    # annotated subset: human toxicity scores
    ds = load_dataset("toxigen/toxigen-data", "annotated", split="train", trust_remote_code=True)
    rows = []
    for r in ds:
        t = clean_text(r["text"])
        if not t:
            continue
        try:
            score = float(r.get("toxicity_human", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        rows.append({
            "text": t,
            "label": 1 if score >= TOXIGEN_THRESHOLD else 0,
            "source": "toxigen",
            "origin_label": f"{score:.1f}",
            "split_origin": "train",
        })
    df = _df(rows, ["text", "label", "source", "origin_label", "split_origin"])
    df.to_parquet(RAW_DIR / "toxigen.parquet", index=False)
    print(f"toxigen: {len(df):,} rows | pos={int((df.label == 1).sum()):,} neg={int((df.label == 0).sum()):,}")

    # machine-generated subset: prompt labels (1=toxic, 0=benign)
    ds = load_dataset("toxigen/toxigen-data", "train", split="train", trust_remote_code=True)
    rows = []
    for r in ds:
        t = clean_text(r["generation"])
        if not t:
            continue
        try:
            lab = int(r["prompt_label"])
        except (TypeError, ValueError):
            continue
        rows.append({
            "text": t,
            "label": 1 if lab == 1 else 0,
            "source": "toxigen_machine",
            "origin_label": "toxic" if lab == 1 else "benign",
            "split_origin": "train",
        })
    df = _df(rows, ["text", "label", "source", "origin_label", "split_origin"])
    df.to_parquet(RAW_DIR / "toxigen_machine.parquet", index=False)
    print(f"toxigen_machine: {len(df):,} rows | pos={int((df.label == 1).sum()):,} neg={int((df.label == 0).sum()):,}")


# --------------------------------------------------------------------------- #
# HateXplain (rationale-annotated hate/offensive/normal)
# --------------------------------------------------------------------------- #
def load_hatexplain():
    ds = load_dataset("dataspoof/HateXplain", split="train")
    rows = []
    for r in ds:
        raw = r["text"]
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                raw = " ".join(json.loads(raw))
            except json.JSONDecodeError:
                pass
        t = clean_text(raw)
        if not t:
            continue
        lab = str(r["label"]).lower()
        if lab in ("0", "normal"):
            label, ol = 0, "normal"
        elif lab in ("1", "hate"):
            label, ol = 1, "hate"
        elif lab in ("2", "offensive"):
            label, ol = 1, "offensive"
        else:
            continue
        rows.append({
            "text": t,
            "label": label,
            "source": "hatexplain",
            "origin_label": ol,
            "split_origin": "test",
        })
    df = _df(rows, ["text", "label", "source", "origin_label", "split_origin"])
    df = df[df.text != ""]
    df.to_parquet(RAW_DIR / "hatexplain.parquet", index=False)
    print(f"hatexplain: {len(df):,} rows | pos={int((df.label == 1).sum()):,} neg={int((df.label == 0).sum()):,}")


# --------------------------------------------------------------------------- #
# Wikipedia clean filler
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_wikipedia(cap_sentences: int, min_words: int = 4, max_words: int = 60):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    rows = []
    count = 0
    for art in ds:
        text = art.get("text") or ""
        sentences = _SENT_SPLIT.split(text.replace("\n", " "))
        for s in sentences:
            s = clean_text(s)
            n = len(s.split())
            if n < min_words or n > max_words:
                continue
            rows.append({
                "text": s,
                "label": 0,
                "source": "wikipedia",
                "origin_label": "neutral",
                "split_origin": "train",
            })
            count += 1
            if count >= cap_sentences:
                break
        if count >= cap_sentences:
            break
    df = _df(rows, ["text", "label", "source", "origin_label", "split_origin"])
    df.to_parquet(RAW_DIR / "wikipedia.parquet", index=False)
    print(f"wikipedia: {len(df):,} rows | pos={int((df.label == 1).sum()):,} neg={int((df.label == 0).sum()):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*",
                    default=["jigsaw", "davidson", "toxigen", "hatexplain", "wikipedia"])
    ap.add_argument("--wiki_cap", type=int, default=350_000)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for s in args.sources:
        print(f"\n=== {s} ===")
        if s == "jigsaw":
            load_jigsaw()
        elif s == "davidson":
            load_davidson()
        elif s == "toxigen":
            load_toxigen()
        elif s == "hatexplain":
            load_hatexplain()
        elif s == "wikipedia":
            load_wikipedia(args.wiki_cap)
        else:
            print(f"  unknown source {s!r}")


if __name__ == "__main__":
    main()
