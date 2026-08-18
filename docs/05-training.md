# Training & Inference

Once the dataset exists at `data/final/dataset`, this page covers fine-tuning,
finalizing, rebalancing and running inference.

The quickest path:

```bash
# fine-tune distilbert-base-uncased (defaults target data/final/dataset)
.venv/bin/python scripts/train.py

# classify some text with the result
printf "you are a f4gg0t and i hate you\nthanks for the help\n" \
    | .venv/bin/python scripts/predict.py --model data/final/model/model
```

---

## train.py — fine-tune a classifier

Fine-tunes a binary text classifier (default `distilbert-base-uncased`) on the
dataset. Runs on GPU if available; `--max_train_rows` keeps first runs fast on
small GPUs.

```bash
.venv/bin/python scripts/train.py
.venv/bin/python scripts/train.py --max_train_rows 200000 --epochs 1 --batch_size 32
.venv/bin/python scripts/train.py --data_dir /path/to/dataset --output_dir runs/my-run
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--model` | `distilbert-base-uncased` | base model (HF id or local dir) |
| `--data_dir` | `data/final/dataset` | the DatasetDict to train on |
| `--output_dir` | `data/final/model` | where checkpoints / model / metrics go |
| `--max_train_rows` | `200000` | subsample train rows; `None` or `all` = full ~1.32M-row split |
| `--max_val_rows` | `10000` | subsample validation rows; `None` = full |
| `--max_length` | `auto` | truncation length; `auto` = p95 of token lengths on a 50k sample (usually 2–4× faster than 256) |
| `--epochs` | `1` | training epochs |
| `--batch_size` | `32` | per-device batch size (multiple of 8 = fastest GEMMs) |
| `--grad_accum` | `1` | gradient accumulation steps |
| `--grad_checkpoint` | off | gradient checkpointing (less VRAM, more compute) |
| `--lr` | `2e-5` | learning rate |
| `--warmup_ratio` | `0.06` | warmup as a fraction of total steps |
| `--weight_decay` | `0.01` | weight decay |
| `--eval_steps` | `500` | evaluation interval when `--eval_strategy steps` |
| `--save_steps` | `1000` | checkpoint interval when `--save_strategy steps` |
| `--eval_strategy` | `epoch` | `steps` \| `epoch` \| `no` |
| `--save_strategy` | `epoch` | `steps` \| `epoch` \| `no` |
| `--dataloader_num_workers` | `4` | dataloader workers |
| `--pin_memory` | on | pin dataloader memory |
| `--logging_steps` | `200` | log interval |
| `--seed` | `42` | RNG seed |
| `--fp16` | off | mixed precision (may not speed up a GTX 1660 Ti) |
| `--resume` | `None` | resume training from a checkpoint dir, or `auto` to use the latest checkpoint in `--output_dir/checkpoints` |
| `--push_to_hub` | `None` | push the trained model to this repo id |
| `--private` | off | make the pushed repo private |

### What it does

- **Stratified subsampling** — `--max_train_rows` / `--max_val_rows` downsample
  while preserving the label ratio.
- **Resuming** — `--resume <ckpt>` continues from a saved checkpoint (model,
  optimizer, scheduler, RNG and dataloader state all restored). Use the *same*
  `--output_dir`, `--data_dir`, subsample args, `--max_length`, `--batch_size`,
  `--grad_accum` and `--lr` as the original run. `--epochs` is the **total**
  epochs of the whole run, not "extra" epochs — remaining epochs are derived
  from the checkpoint's `global_step`.
- **Tokenization caches** go to `--output_dir/.cache/`, never next to the
  dataset, so no stray `cache-*.arrow` files corrupt the Hub folder.
- Checkpoints go to `--output_dir/checkpoints/` (keeps the best 2 by `f1`).
- After training it evaluates on `validation`, held-out `test`, and
  `test_obfuscated` (mutated rows) and writes `eval_metrics.json` plus
  `training_args.json`.

### Outputs

```
output_dir/
├── checkpoints/          # Trainer checkpoints (best 2 kept by f1)
├── .cache/               # tokenized caches (gitignored)
├── model/                # final model + tokenizer
├── eval_metrics.json     # accuracy/precision/recall/f1 per split
└── training_args.json    # the run configuration
```

> **transformers v5 required** — the script uses v5 APIs (`processing_class`,
> `warmup_steps`). Keep `transformers>=5.0`.

---

## finalize.py — pick the best checkpoint and export

`train.py` already saves a `model/` at the end. `finalize.py` is for the case
where you want to (re)select the best checkpoint afterwards — it finds the
checkpoint whose **own epoch** had the best validation F1, loads it, evaluates
on the full `validation` / `test` / `test_obfuscated` splits, and exports the
model.

```bash
# use the best checkpoint automatically
.venv/bin/python scripts/finalize.py

# a specific checkpoint
.venv/bin/python scripts/finalize.py --checkpoint data/final/model/checkpoints/checkpoint-82546
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | best-by-f1 | specific checkpoint dir |
| `--data_dir` | `data/final/dataset` | dataset to evaluate on |
| `--output_dir` | `data/final/model` | run dir containing `checkpoints/` |
| `--max_length` | `auto` | truncation length, or `auto` (p95) |
| `--max_val_rows` | `None` | limit validation rows (`None` = full) |
| `--batch_size` | `64` | eval batch size |

### Outputs

`{output_dir}/model`, `{output_dir}/eval_metrics.json`,
`{output_dir}/training_args.json`.

`make_dataset.py` runs `train` → `finalize` automatically when `train` is in
the steps.

---

## predict.py — CLI inference

Classifies text with a trained model and prints one line per input:

```bash
# stdin (one text per line)
printf "you are a f4gg0t and i hate you\nthanks for the help\n" \
    | .venv/bin/python scripts/predict.py --model data/final/model/model

# single text
.venv/bin/python scripts/predict.py --model data/final/model --text "I love this"

# a file, one text per line
.venv/bin/python scripts/predict.py --model data/final/model -f texts.txt
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--model` | `data/final/model/model` | model directory |
| `--text` | `None` | single text to classify |
| `-f` / `--file` | `None` | file with one text per line |
| `--device` | auto | device id, e.g. `0` |

### Output format

```
offensive  conf=0.998 | you are a f4gg0t and i hate you
clean      conf=0.997 | thanks for the help
```

---

## make_benchmark.py — build the benchmark parquet

Exports the held-out rows into a single parquet for scoring models on a
consistent, balanced evaluation set:

```bash
.venv/bin/python scripts/make_benchmark.py
```

### What it does

- Reads `data/processed/split.parquet`, takes the `test` rows (`benchmark_split
  = "test"`), appends their mutated subset (the `test_obfuscated` rows,
  `benchmark_split = "test_obfuscated"`), and writes
  `data/final/benchmark/benchmark.parquet`.
- Keeps the full schema (`text`, `label`, `source`, `origin_label`,
  `split_origin`, `mutated`, `variant`, `cluster`, `benchmark_split`) so
  scoring can be sliced by split, by plain/obfuscated, or by source.

> `test_obfuscated` rows are a subset of `test`, so those texts appear twice in
> the merged parquet — once per `benchmark_split`. Slice by `benchmark_split`
> (or `mutated`) rather than summing rows across both.

### Arguments

| flag | default | meaning |
|---|---|---|
| `--split_file` | `data/processed/split.parquet` | pipeline parquet with a `split` column |
| `--out` | `data/final/benchmark/benchmark.parquet` | output parquet path |

---

## benchmark.py — score models on the benchmark

Runs one or more trained models over the benchmark parquet and reports
accuracy / precision / recall / f1, sliced by benchmark split, by
plain/obfuscated (`mutated`), and by source:

```bash
# one model
.venv/bin/python scripts/benchmark.py --model data/final/model/model

# compare several (final model + checkpoints)
.venv/bin/python scripts/benchmark.py \
    --model data/final/model/model \
    --model data/final/model/checkpoints/checkpoint-300000

# mix in Hugging Face Hub models
.venv/bin/python scripts/benchmark.py \
    --model data/final/model/model \
    --model user/moderation-model
```

Models are anything `AutoModelForSequenceClassification.from_pretrained`
accepts: a local directory or a Hugging Face Hub id. Argmax indices are mapped
to `0/1` labels via the model's `config.id2label` when the names are
recognizable (`clean`/`offensive`, `toxic`, `hate`, …); otherwise the index is
used as-is. A warning is printed if the model has anything other than 2
labels.

### Arguments

| flag | default | meaning |
|---|---|---|
| `--model` | *(required, repeatable)* | model to score: a local directory or a Hugging Face Hub id; pass once per model |
| `--benchmark` | `data/final/benchmark/benchmark.parquet` | benchmark parquet |
| `--batch_size` | `32` | inference batch size |
| `--max_length` | `512` | token truncation length |
| `--max_rows` | `None` | score only the first N rows (quick smoke runs) |
| `--device` | auto | device id, e.g. `0` |
| `--out` | `data/final/benchmark/results.json` | metrics output |

### Output

Per-model metrics (`overall`, `by_benchmark_split`, `by_mutated`,
`by_source`) written to `results.json`, plus a printed per-model summary.

---

## make_balanced.py — rebalance the train split

The default build's train split is positive-biased (~63/37). This script
downsamples the majority class **at the cluster level** — whole near-duplicate
clusters (a base sentence + all its obfuscation variants) are kept or dropped
together, preserving the near-duplicate invariant from step 4. The result is a
~50/50 train split.

```bash
.venv/bin/python scripts/make_balanced.py
.venv/bin/python scripts/make_balanced.py --seed 7 --no-backup
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--data_dir` | `data/final/dataset` | the DatasetDict to rebalance |
| `--seed` | `42` | RNG seed for which clusters to drop |
| `--no-backup` | off | skip the unbalanced backup copy |

### What it does

- Leaves `validation` / `test` / `test_obfuscated` untouched.
- Backs up the original dataset to `data/final/dataset_unbalanced` (unless
  `--no-backup`).
- Drops majority-class clusters until train is ~50/50.
- Regenerates `data/final/stats.json` and `data/final/dataset_card.md`.

### Output

```
data/final/dataset             # balanced DatasetDict (train ~50/50)
data/final/dataset_unbalanced  # pre-rebalance copy (unless --no-backup)
```