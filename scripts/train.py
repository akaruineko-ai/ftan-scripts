"""Fine-tune a DistilBERT-style model (binary) on the ftan-2.0 dataset.

Defaults target the bundled `data/final/dataset`. Training runs on GPU if
available; subsampling (`--max_train_rows`) keeps first runs fast on small
GPUs like the GTX 1660 Ti.

Usage:
    .venv/bin/python scripts/train.py
    .venv/bin/python scripts/train.py --max_train_rows 200000 --epochs 1 --batch_size 32
    .venv/bin/python scripts/train.py --data_dir /path/to/dataset --output_dir runs/my-run

Outputs (under --output_dir):
    model/              best checkpoint + tokenizer
    eval_metrics.json   final metrics on validation / test / test_obfuscated
    training_args.json  the run configuration
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common import FINAL_DIR

ID2LABEL = {0: "clean", 1: "offensive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def int_or_none(s):
    """argparse type: int, or None for 'None' / 'all' (use the full split)."""
    if s is None or str(s).strip().lower() in ("none", "all", ""):
        return None
    return int(s)


def resolve_max_length(tokenizer, ds, max_length, n=50_000, seed=42):
    """Return an int length; 'auto' = 95th percentile of token lengths.

    Most of this dataset is short social-media text, so truncating at the p95
    (rather than 256) keeps nearly all tokens while cutting attention cost a lot.
    """
    if isinstance(max_length, int) or str(max_length).isdigit():
        return int(max_length)
    rng = np.random.default_rng(seed)
    n = min(n, len(ds))
    sample = ds.select(rng.choice(len(ds), size=n, replace=False))
    lengths = []
    for batch in sample.batch(1024):
        enc = tokenizer(batch["text"], truncation=True, max_length=512)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    p95 = int(np.percentile(lengths, 95))
    auto = max(16, min(p95, 512))
    print(f"[auto max_length] p95 of {len(lengths):,} texts = {p95} tokens "
          f"-> using max_length={auto}")
    return auto


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--data_dir", default=str(FINAL_DIR / "dataset"))
    ap.add_argument("--output_dir", default=str(FINAL_DIR / "model"))
    ap.add_argument("--max_train_rows", type=int_or_none, default=200_000,
                    help="subsample train rows (None or 'all' = use everything)")
    ap.add_argument("--max_val_rows", type=int_or_none, default=10_000)
    ap.add_argument("--max_length", type=str, default="auto",
                    help="token truncation length, or 'auto' for the 95th "
                         "percentile length on a sample (usually 2-4x faster)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="use a multiple of 8 for fastest GEMMs")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--grad_checkpoint", action="store_true",
                    help="gradient checkpointing (less VRAM, a bit more compute)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eval_steps", type=int, default=500,
                    help="used only when --eval_strategy steps")
    ap.add_argument("--save_steps", type=int, default=1000,
                    help="used only when --save_strategy steps")
    ap.add_argument("--eval_strategy", choices=["steps", "epoch", "no"],
                    default="epoch", help="'epoch' evaluates once per epoch")
    ap.add_argument("--save_strategy", choices=["steps", "epoch", "no"],
                    default="epoch")
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true", default=True)
    ap.add_argument("--logging_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true",
                    help="mixed precision (may not speed up a GTX 1660 Ti)")
    ap.add_argument("--push_to_hub", default=None,
                    help="push the trained model to this repo id (e.g. user/ftan-distilbert)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dsd: DatasetDict = load_from_disk(args.data_dir)
    print("splits:", {k: len(v) for k, v in dsd.items()})

    def stratified_select(ds, n, seed):
        """Random subsample of n rows preserving the label ratio (no cache writes)."""
        if n is None or len(ds) <= n:
            return ds
        labels = np.asarray(ds["label"])
        rng = np.random.default_rng(seed)
        pos_ix = np.flatnonzero(labels == 1)
        neg_ix = np.flatnonzero(labels == 0)
        n_pos = max(1, int(round(n * len(pos_ix) / len(ds))))
        n_neg = max(0, n - n_pos)
        chosen = np.concatenate([
            rng.choice(pos_ix, size=min(n_pos, len(pos_ix)), replace=False),
            rng.choice(neg_ix, size=min(n_neg, len(neg_ix)), replace=False),
        ])
        return ds.select(np.sort(chosen))

    train_ds = stratified_select(dsd["train"], args.max_train_rows, args.seed)
    val_ds = stratified_select(dsd["validation"], args.max_val_rows, args.seed)
    print(f"train: {len(train_ds):,} | val: {len(val_ds):,}")

    args.max_length = resolve_max_length(tokenizer, train_ds, args.max_length)

    out = args.output_dir
    os.makedirs(f"{out}/.cache", exist_ok=True)

    cols = dsd["train"].column_names

    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True,
                        max_length=args.max_length, padding=False)
        enc["labels"] = batch["label"]
        return enc

    def tok_map(ds, name):
        # cache to the run dir so we never write cache-*.arrow next to the dataset
        return ds.map(tok, batched=True, remove_columns=cols,
                      cache_file_name=f"{out}/.cache/{name}.arrow",
                      load_from_cache_file=False)

    train_ds = tok_map(train_ds, "train")
    val_ds = tok_map(val_ds, "validation")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    warmup_steps = int(args.warmup_ratio * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=f"{out}/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(8, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        gradient_checkpointing=args.grad_checkpoint,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=args.eval_strategy != "no",
        metric_for_best_model="f1",
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.pin_memory,
        report_to=[],
    )

    print(f"[config] {len(train_ds):,} rows x {args.epochs} epochs "
          f"= {steps_per_epoch * args.epochs:,} steps (warmup {warmup_steps})")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    model_dir = f"{out}/model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    results = {}
    for split_name in ("validation", "test", "test_obfuscated"):
        if split_name not in dsd:
            continue
        ev = tok_map(dsd[split_name], split_name)
        res = trainer.evaluate(ev, metric_key_prefix=split_name)
        results[split_name] = {k.replace(f"{split_name}_", ""): v for k, v in res.items()}
        print(f"[{split_name}] " +
              " ".join(f"{k}={v:.4f}" for k, v in results[split_name].items()))

    with open(f"{out}/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{out}/training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\nsaved model + metrics to {out}/")

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()
