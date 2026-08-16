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

Or run the whole thing with the one-shot orchestrator (steps below run in
dependency order and stream their output; `download` is skipped automatically
if `data/raw/` already has sources):

```bash
# build the full core dataset (download -> ... -> export)
.venv/bin/python scripts/make_dataset.py

# core + reddit expansion in one pass
.venv/bin/python scripts/make_dataset.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0

# a specific plan: presets are core / reddit / all / train, or any comma list
.venv/bin/python scripts/make_dataset.py --steps reddit --dump RC_2025-01.zst
.venv/bin/python scripts/make_dataset.py --steps all,train --device 0
```

`make_dataset.py` steps: `download, fetch, curate, merge, normalize, mutate,
dedup, split, export, train, finalize`. `merge` converts the curated reddit
banks into `data/raw/reddit.parquet` so the core pipeline treats them like any
other source (mutated, cluster-deduped and split together).

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

## Reddit expansion (profanity / direct-insult banks)

The 1.37M-row core dataset is trained mostly on "every profanity counts as
offensive" sources. To teach a model the difference between *swearing at a
person* (`fuck you`) and *swearing about a situation* (`fuck i lost my keys`,
`this is fucking awesome`), collect raw Reddit comments and split them with
regex + the current FTAN model into three banks:

| bank | bucket | label | meaning |
|---|---|---|---|
| `bank_a` | attack | 1 | profanity/insult aimed at a person (2nd-person in a small window, or explicit pattern like `kys`, `fuck off`) |
| `bank_b` | emotional | 0 | expletive profanity with no addressee — the hard negatives FTAN gets wrong |
| `bank_c` | clean | 0 | no profanity, no insult |
| `manual_check` | grey / verify | ? | FTAN cannot decide (conf in `[--grey_low, --grey_high]`) or disagrees with regex — review by hand |

```bash
# 1. fetch + regex-prefilter comments (Arctic Shift API, the Pushshift successor)
.venv/bin/python scripts/07_reddit_fetch.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --after 2024-01-01 --max_rows 5000000 --out data/reddit/raw

#    ... or stream a monthly .zst dump (download via torrent, then point at the file)
.venv/bin/python scripts/07_reddit_fetch.py --dump RC_2025-01.zst \
    --clean_sample_frac 0.05 --out data/reddit/raw

# 2. regex + FTAN -> banks
.venv/bin/python scripts/08_reddit_curate.py --raw data/reddit/raw \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0
```

Or run the whole expansion in one go (banks are then merged into the core
pipeline via the `merge` step of `make_dataset.py`):

```bash
# one-shot: fetch -> curate -> merge -> normalize -> ... -> export
.venv/bin/python scripts/make_dataset.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0

# one-shot from an already-downloaded monthly dump
.venv/bin/python scripts/make_dataset.py --steps reddit --dump RC_2025-01.zst --device 0

# curate only (reuse existing candidates), fetch only, respectively
.venv/bin/python scripts/make_dataset.py --skip-fetch --device 0
.venv/bin/python scripts/make_dataset.py --skip-curate --subreddits leagueoflegends
```

- Regex is the authoritative splitter (`scripts/reddit_vocab.py`); it tolerates
  light obfuscation (`f*ck`, `f**k`, `sh*t`, `b*tch`) so censored curses are
  still caught. FTAN (`data/final/model/model`) is only invoked on the **grey**
  zone — insult words without an addressee (`what an idiot`) — and splits it by
  confidence into `bank_a` / `bank_c` / `manual_check`.
- `--verify_frac 0.05` re-checks a random slice of `bank_a`/`bank_c` with FTAN
  and pulls disagreements into `manual_check` (catch regex misses).
- The dumps are hosted on Academic Torrents (see
  `download_links.md` in the arctic-shift repo); `--clean_sample_frac` keeps
  only a fraction of regex-clean comments so the clean bank doesn't balloon.

## License / disclaimer

Sources have varied licenses (CC0 for Jigsaw train; research terms for Davidson /
HateXplain; CC-BY-SA for ToxiGen). The dataset **contains raw offensive language** and
is intended for moderation research only.
