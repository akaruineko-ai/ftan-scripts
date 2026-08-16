# ftan-2.0

A large-scale, obfuscation-aware **offensive / clean English text dataset** for training
moderation / profanity classifiers, published as a Hugging Face `DatasetDict`.

## Pipeline

```bash
# 0. setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. download sources -> data/raw/*.parquet
.venv/bin/python scripts/01_download.py

# 2. merge + exact-dedupe + filter -> data/processed/normalized.parquet
.venv/bin/python scripts/02_normalize.py

# 3. obfuscation mutations -> data/processed/mutated.parquet
.venv/bin/python scripts/03_mutate.py --offensive_variants 3 --clean_variant_frac 0.4

# 4. orthographic clustering + exact-dup removal -> data/processed/deduped.parquet
.venv/bin/python scripts/04_dedup.py

# 5. cluster-stratified splits -> data/processed/split.parquet
.venv/bin/python scripts/05_split.py

# 6. export HF DatasetDict -> data/final/dataset + stats + card
.venv/bin/python scripts/06_export.py --eval_obfuscation

# optional: push to the Hub
.venv/bin/python scripts/06_export.py --hub_id youruser/ftan-2.0 --private
```

## Sources

| source | id | role |
|---|---|---|
| Jigsaw Toxic Comment | `tcapelle/jigsaw-toxic-comment-classification-challenge` | offensive + clean (in-domain) |
| Davidson et al. | `contemmcm/hate-speech-and-offensive-language` | offensive tweets + clean |
| ToxiGen | `toxigen/toxigen-data` | subtle toxicity (machine-generated, human scores) |
| HateXplain | `Hate-speech-CNERG/hatexplain` | held-out test + train |
| Wikipedia | `wikimedia/wikipedia` 20231101.en | neutral filler |

## Output schema

`text` (str), `label` (1=offensive / 0=clean), `source`, `origin_label`,
`split_origin`, `mutated` (0/1), `variant`, `cluster` (orthographic-dup id for safe splitting).

Splits: `train`, `validation`, `test`, plus `test_obfuscated` (mutated test rows).

Current build (~1.37M rows):

| split | rows | offensive | clean |
|---|---|---|---|
| train | 1,320,712 | 559,852 | 760,860 |
| validation | 20,001 | 9,996 | 10,005 |
| test | 30,001 | 14,998 | 15,003 |
| test_obfuscated | 14,084 | 10,793 | 3,291 |

## Design notes

- **Obfuscation engine** mutates curse words (offensive rows) *and* innocent words
  (clean rows), so the model learns that leet-speak / censoring / homoglyphs are
  orthographic variation — not a semantic signal. See `scripts/common.py`.
- **Orthographic clustering**: each text is canonicalized (homoglyph -> latin,
  leet digits -> letters, symbols stripped, repeats collapsed) and rows sharing a
  canonical form share a `cluster` id. Splitting happens at the cluster level so a
  sentence and its mutated variants never leak across train/validation/test.
- The `test` split is a balanced sample drawn from held-out sources (HateXplain,
  Jigsaw balanced_test); `test_obfuscated` holds their mutated rows for robustness
  evaluation.
- Generated dataset is saved under `data/final/` (gitignored).

## Publishing

Two robust ways to upload `data/final/dataset` to the Hugging Face Hub:

```bash
# 1. from the pipeline (writes parquet, no stray arrow cache files)
.venv/bin/python scripts/06_export.py --hub_id youruser/ftan-2.0 --private

# 2. upload the on-disk DatasetDict folder directly
huggingface_hub upload-folder data/final/dataset --repo-type=dataset youruser/ftan-2.0
```

Do **not** upload `data/processed/*.parquet` or any `cache-*.arrow` files —
their stray pandas index columns break the Hub's auto-conversion
(`DatasetGenerationError` / `CastError` on an `indices` column).

## Fine-tuning

Fine-tune `distilbert-base-uncased` on the dataset:

```bash
# quick run (~200k rows, ~1 epoch on a GTX 1660 Ti)
.venv/bin/python scripts/train.py

# full control
.venv/bin/python scripts/train.py \
    --model distilbert-base-uncased \
    --max_train_rows 200000 --max_length 128 \
    --epochs 1 --batch_size 32 --lr 2e-5 \
    --output_dir data/final/model
```

- The training run evaluates on `validation`, held-out `test`, and
  `test_obfuscated` (mutated rows) and writes `eval_metrics.json`.
- `--max_train_rows None` uses the full ~1.32M-row train split.
- The scripts use transformers **v5** APIs (`processing_class`,
  `warmup_steps`) — keep `transformers>=5.0`.

### Inference

```bash
printf "you are a f4gg0t and i hate you\nthanks for the help\n" \
    | .venv/bin/python scripts/predict.py --model data/final/model/model
```

## License / disclaimer

Sources have varied licenses (CC0 for Jigsaw train; research terms for Davidson /
HateXplain; CC-BY-SA for ToxiGen). The dataset **contains raw offensive language** and
is intended for moderation research only.
