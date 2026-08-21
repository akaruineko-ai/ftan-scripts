"""Fine-tune a DistilBERT-style model on the ftan-2.0 dataset.

Defaults to **multi-label** (H / HR / H2) training against the bundled
`data/exps/final/dataset`. Pass `--no-multi_label` to fall back to the legacy
binary (clean / offensive) mode. Training runs on GPU if available;
subsampling (`--max_train_rows`) keeps first runs fast on small GPUs like the
GTX 1660 Ti.

Usage:
    .venv/bin/python scripts/train.py
    .venv/bin/python scripts/train.py --max_train_rows 200000 --epochs 1 --batch_size 32
    .venv/bin/python scripts/train.py --data_dir /path/to/dataset --output_dir runs/my-run
    .venv/bin/python scripts/train.py --output_dir runs/my-run --resume auto
    .venv/bin/python scripts/train.py --no-multi_label

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
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common import DATA_DIR, FINAL_DIR

# Legacy binary (clean / offensive) configuration.
BINARY_ID2LABEL = {0: "clean", 1: "offensive"}
BINARY_LABEL2ID = {v: k for k, v in BINARY_ID2LABEL.items()}

# Multi-label configuration. The order is the vector index order and matches
# derive_labels() in scripts/multi_label_dataset.py.
ML_LABELS = ["H", "HR", "H2"]
ML_ID2LABEL = {i: lab for i, lab in enumerate(ML_LABELS)}
ML_LABEL2ID = {lab: i for i, lab in enumerate(ML_LABELS)}
ML_DEFAULT_DIR = DATA_DIR / "exps/final/dataset"


def int_or_none(s):
    """argparse type: int, or None for 'None' / 'all' (use the full split)."""
    if s is None or str(s).strip().lower() in ("none", "all", ""):
        return None
    return int(s)


def resolve_resume(out, resume):
    """Return the checkpoint path to resume from (None = fresh run)."""
    if resume is None or str(resume).strip().lower() in ("none", "false", ""):
        return None
    if str(resume).strip().lower() == "auto":
        ckpts = sorted(
            (d for d in os.listdir(f"{out}/checkpoints")
             if d.startswith("checkpoint-")),
            key=lambda d: int(d.split("-")[-1]),
        )
        if not ckpts:
            raise SystemExit(f"no checkpoints found in {out}/checkpoints "
                             "for --resume auto")
        resume = os.path.join(out, "checkpoints", ckpts[-1])
        print(f"[resume] auto -> {resume}")
    if not os.path.isdir(resume):
        raise SystemExit(f"resume checkpoint not found: {resume}")
    return resume


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


def compute_metrics_multi(eval_pred, threshold=0.5):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.as_tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels)
    out = {
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds),
    }
    for i, lab in enumerate(ML_LABELS):
        out[f"f1_{lab}"] = f1_score(
            labels[:, i], preds[:, i], zero_division=0
        )
    return out


def parse_pos_weight(spec, train_ds):
    """Return a length-3 float tensor (H/HR/H2) or None.

    spec: 'none' -> None; 'balanced' -> (n - pos)/pos per label computed from
    the tokenized train_ds multi-hot labels; or a 'w0,w1,w2' comma list.
    """
    if spec is None or str(spec).strip().lower() in ("none", ""):
        return None
    if str(spec).strip().lower() == "balanced":
        labels = np.asarray(train_ds["labels"], dtype=float)  # (n, 3)
        n = labels.shape[0]
        pos = labels.sum(axis=0)
        # avoid div-by-zero for labels with no positives
        w = np.where(pos > 0, (n - pos) / np.maximum(pos, 1e-8), 1.0)
        print(f"[pos_weight] balanced = {[round(float(x), 3) for x in w]}")
        return torch.as_tensor(w, dtype=torch.float32)
    try:
        vals = [float(x) for x in str(spec).split(",")]
    except ValueError:
        raise SystemExit(f"--pos_weight must be 'none', 'balanced', or "
                         f"comma floats, got: {spec!r}")
    if len(vals) != len(ML_LABELS):
        raise SystemExit(f"--pos_weight list must have {len(ML_LABELS)} "
                         f"values (H/HR/H2), got {len(vals)}")
    print(f"[pos_weight] manual = {vals}")
    return torch.as_tensor(vals, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer that applies a BCE pos_weight for multi-label losses."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(outputs.logits.device)
            if self.pos_weight is not None else None
        )
        loss = loss_fct(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--data_dir", default=str(ML_DEFAULT_DIR))
    ap.add_argument("--output_dir", default=str(FINAL_DIR / "model"))
    ap.add_argument("--multi_label", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="multi-label (H/HR/H2) training [default on]; use "
                         "--no-multi_label for the legacy binary clean/offensive task")
    ap.add_argument("--val_fraction", type=float, default=0.02,
                    help="fraction of train carved out for validation when the "
                         "dataset has no 'validation' split")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="sigmoid cutoff for multi-label predictions/metrics")
    ap.add_argument("--max_train_rows", type=int_or_none, default=200_000,
                    help="subsample train rows (None or 'all' = use everything)")
    ap.add_argument("--max_val_rows", type=int_or_none, default=10_000)
    ap.add_argument("--pos_weight", default="none",
                    help="multi-label class weights for BCEWithLogitsLoss: "
                         "'none' (default, no reweighting), 'balanced' "
                         "(auto (n-pos)/pos from the train split), or a "
                         "comma list of floats in H/HR/H2 order, e.g. '1,3,8'")
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
    ap.add_argument("--resume", default=None,
                    help="resume from this checkpoint path, or 'auto' to use "
                         "the latest checkpoint in --output_dir/checkpoints")
    ap.add_argument("--push_to_hub", default=None,
                    help="push the trained model to this repo id (e.g. user/ftan-distilbert)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dsd: DatasetDict = load_from_disk(args.data_dir)
    print("splits:", {k: len(v) for k, v in dsd.items()})

    multi = args.multi_label

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

    def random_select(ds, n, seed):
        """Seeded random subsample (no stratification; works for multi-label)."""
        if n is None or len(ds) <= n:
            return ds
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(ds), size=n, replace=False))
        return ds.select(chosen)

    select_fn = random_select if multi else stratified_select

    # resolve the validation split
    if "validation" in dsd:
        val_ds = dsd["validation"]
        print("[split] using bundled 'validation' split")
        train_ds = dsd["train"]
    elif multi:
        # no validation split: carve a deterministic fraction of train
        full = dsd["train"].shuffle(seed=args.seed)
        n_val = int(args.val_fraction * len(full))
        val_ds = full.select(range(n_val))
        train_ds = full.select(range(n_val, len(full)))
        print(f"[split] carved validation={n_val:,} from train "
              f"(val_fraction={args.val_fraction})")
    else:
        raise SystemExit("binary mode requires a 'validation' split in the "
                         "dataset; this dataset has none (use --multi_label)")

    train_ds = select_fn(train_ds, args.max_train_rows, args.seed)
    val_ds = select_fn(val_ds, args.max_val_rows, args.seed)
    print(f"train: {len(train_ds):,} | val: {len(val_ds):,}")

    args.max_length = resolve_max_length(tokenizer, train_ds, args.max_length)

    out = args.output_dir
    os.makedirs(f"{out}/.cache", exist_ok=True)

    cols = dsd["train"].column_names

    if multi:
        def to_multihot(label_list):
            vec = [0.0] * len(ML_LABELS)
            for lab in label_list:
                if lab in ML_LABEL2ID:
                    vec[ML_LABEL2ID[lab]] = 1.0
            return vec

        def tok(batch):
            enc = tokenizer(batch["text"], truncation=True,
                            max_length=args.max_length, padding=False)
            enc["labels"] = [to_multihot(labs) for labs in batch["labels"]]
            return enc
    else:
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

    if multi:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(ML_LABELS),
            problem_type="multi_label_classification",
            id2label=ML_ID2LABEL, label2id=ML_LABEL2ID,
        )
        compute = lambda ep: compute_metrics_multi(ep, threshold=args.threshold)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2,
            id2label=BINARY_ID2LABEL, label2id=BINARY_LABEL2ID
        )
        compute = compute_metrics

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

    pos_weight = parse_pos_weight(args.pos_weight, train_ds) if multi else None
    trainer_cls = WeightedTrainer if pos_weight is not None else Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute,
        processing_class=tokenizer,
        pos_weight=pos_weight,
    )

    trainer.train(resume_from_checkpoint=resolve_resume(out, args.resume))

    model_dir = f"{out}/model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    results = {}
    # validation: whether bundled or carved from train, it lives in val_ds
    res = trainer.evaluate(val_ds, metric_key_prefix="validation")
    results["validation"] = {k.replace("validation_", ""): v for k, v in res.items()}
    print(f"[validation] " +
          " ".join(f"{k}={v:.4f}" for k, v in results["validation"].items()))
    for split_name in ("test", "test_obfuscated"):
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
