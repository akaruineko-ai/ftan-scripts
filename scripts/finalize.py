"""Finalize a trained run: pick the best checkpoint and export the final model.

Evaluates on the full validation / test / test_obfuscated splits, saves the
model + tokenizer to `{output_dir}/model`, and writes eval_metrics.json.

Usage:
    .venv/bin/python scripts/finalize.py
    .venv/bin/python scripts/finalize.py --checkpoint data/final/model/checkpoints/checkpoint-82546
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
from train import compute_metrics, int_or_none, resolve_max_length


def best_checkpoint(checkpoints_dir):
    """Return the checkpoint dir whose OWN epoch had the best eval f1.

    Each checkpoint's trainer_state logs accumulate evals up to its own epoch,
    so we take the LAST eval_f1 per checkpoint, not the max over the whole log.
    """
    best_dir, best_f1 = None, -1.0
    for name in sorted(os.listdir(checkpoints_dir)):
        state_file = os.path.join(checkpoints_dir, name, "trainer_state.json")
        if not os.path.isfile(state_file):
            continue
        with open(state_file) as f:
            state = json.load(f)
        f1s = [e.get("eval_f1") for e in state.get("log_history", [])
               if "eval_f1" in e]
        if not f1s:
            continue
        own_f1 = f1s[-1]
        if own_f1 > best_f1:
            best_f1 = own_f1
            best_dir = os.path.join(checkpoints_dir, name)
    if best_dir is None:
        raise SystemExit(f"no checkpoints found in {checkpoints_dir}")
    print(f"[best checkpoint] {os.path.basename(best_dir)} (its val f1={best_f1:.4f})")
    return best_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to a specific checkpoint; default = best val f1")
    ap.add_argument("--data_dir", default=str(FINAL_DIR / "dataset"))
    ap.add_argument("--output_dir", default=str(FINAL_DIR / "model"))
    ap.add_argument("--max_length", type=str, default="auto",
                    help="truncation length for eval, or 'auto' (p95)")
    ap.add_argument("--max_val_rows", type=int_or_none, default=None,
                    help="limit validation rows (None = full split)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    out = args.output_dir
    os.makedirs(f"{out}/.cache", exist_ok=True)

    ckpt_dir = args.checkpoint or best_checkpoint(f"{out}/checkpoints")
    print(f"loading checkpoint: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)

    dsd: DatasetDict = load_from_disk(args.data_dir)
    print("splits:", {k: len(v) for k, v in dsd.items()})

    args.max_length = resolve_max_length(tokenizer, dsd["validation"],
                                         args.max_length)

    cols = dsd["train"].column_names

    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True,
                        max_length=args.max_length, padding=False)
        enc["labels"] = batch["label"]
        return enc

    def tok_map(ds, name):
        return ds.map(tok, batched=True, remove_columns=cols,
                      cache_file_name=f"{out}/.cache/eval_{name}.arrow",
                      load_from_cache_file=False)

    training_args = TrainingArguments(
        output_dir=f"{out}/eval_tmp",
        per_device_eval_batch_size=args.batch_size,
        seed=42,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    results = {}
    for split_name in ("validation", "test", "test_obfuscated"):
        if split_name not in dsd:
            continue
        ev = tok_map(dsd[split_name], split_name)
        if split_name == "validation" and args.max_val_rows is not None \
                and len(ev) > args.max_val_rows:
            ev = ev.select(range(args.max_val_rows))
        res = trainer.evaluate(ev, metric_key_prefix=split_name)
        results[split_name] = {k.replace(f"{split_name}_", ""): v
                               for k, v in res.items()}
        print(f"[{split_name}] " +
              " ".join(f"{k}={v:.4f}" for k, v in results[split_name].items()))

    model_dir = f"{out}/model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(f"{out}/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{out}/training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\nfinal model saved to {model_dir}")


if __name__ == "__main__":
    main()