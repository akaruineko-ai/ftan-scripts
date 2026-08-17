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

## Community expansion (profanity / direct-insult banks)

The 1.37M-row core dataset is trained mostly on "every profanity counts as
offensive" sources. To teach a model the difference between *swearing at a
person* (`fuck you`) and *swearing about a situation* (`fuck i lost my keys`,
`this is fucking awesome`), collect real forum comments and split them with
regex + the current FTAN model into three banks:

| bank | bucket | label | meaning |
|---|---|---|---|
| `bank_a` | attack | 1 | profanity/insult aimed at a person (2nd-person in a small window, or explicit pattern like `kys`, `fuck off`) |
| `bank_b` | emotional | 0 | expletive profanity with no addressee — the hard negatives FTAN gets wrong |
| `bank_c` | clean | 0 | no profanity, no insult |
| `manual_check` | grey / verify | ? | FTAN cannot decide (conf in `[--grey_low, --grey_high]`) or disagrees with regex — review by hand |

Three complementary sources:

| source | why | how |
|---|---|---|
| **Reddit** (`07_reddit_fetch.py`) | huge volume, conflict subreddits | Arctic Shift API or monthly `.zst` dump |
| **4chan** (`09_4chan_fetch.py`) | highest profanity density anywhere — fastest way to fill `bank_a`/`bank_b` (tens of thousands of candidates per board in a minute) | official keyless a.4cdn.org live API (`--live`), or 4plebs archive JSON API (`--archive`) |
| **YouTube** (`10_youtube_fetch.py`) | casual register where nobody swears — clean filler for `bank_c` (cat-video comment sections) | YouTube Data API v3 (free key, 10k units/day) |

All fetchers write the same candidate schema and keep `source` (`reddit` /
`4chan` / `youtube`) through the banks into the final dataset.

```bash
# reddit
.venv/bin/python scripts/07_reddit_fetch.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --after 2024-01-01 --max_rows 5000000 --out data/reddit/raw
#   ... or stream a monthly .zst dump (download via torrent)
.venv/bin/python scripts/07_reddit_fetch.py --dump RC_2025-01.zst \
    --clean_sample_frac 0.05 --out data/reddit/raw

# 4chan live (no key): snapshot every open thread on the boards
.venv/bin/python scripts/09_4chan_fetch.py --live \
    --boards b,pol,r9k,v,gaming,trash --max_rows 2000000 --out data/4chan/raw
#   4chan archive (historical, 4plebs)
.venv/bin/python scripts/09_4chan_fetch.py --archive \
    --board pol --search "fuck you" --out data/4chan/raw
#   NOTE: live mode only returns currently-open threads (~100-300 per board).
#   To accumulate hundreds of thousands of rows over time, use --repeat:
.venv/bin/python scripts/09_4chan_fetch.py --live \
    --boards b,pol --max_rows 1000000 --repeat 100 --repeat_delay 300 \
    --out data/4chan/raw

# youtube clean filler
.venv/bin/python scripts/10_youtube_fetch.py \
    --api_key AIza... --query "cat videos" \
    --max_videos 20 --max_comments_per_video 200 --out data/youtube/raw
#   NOTE: YouTube Data API v3 free tier is 10k units/day.
#   search costs 100 units, commentThreads list costs 1 unit per page.
#   --max_videos 200 + --max_comments_per_video 500 ≈ 3k-5k units per run.

# 2. regex + FTAN -> banks (accepts several --raw dirs, comma-separated)
.venv/bin/python scripts/08_reddit_curate.py \
    --raw data/reddit/raw,data/4chan/raw,data/youtube/raw \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0
```

### Controlling the source mix

By default each fetcher is capped by its own `--max_rows`, and the curate step
samples each bank down to `--target_a/b/c`. If you want *more* YouTube / 4chan
in the final dataset, raise the fetcher caps **and** cap the other sources at
the `merge` step so the core pipeline doesn't drown them out:

```bash
# fetch more raw rows from 4chan / youtube
.venv/bin/python scripts/09_4chan_fetch.py --live --boards b,pol,r9k,v \
    --threads_per_board 500 --max_rows 5000000 --out data/4chan/raw

.venv/bin/python scripts/10_youtube_fetch.py \
    --api_key AIza... --query "cat videos" \
    --max_videos 200 --max_comments_per_video 500 --out data/youtube/raw

# after curate: cap each source BEFORE the core pipeline normalizes/mutates
.venv/bin/python scripts/make_dataset.py --steps merge,subsample,normalize,mutate,dedup,split,export \
    --source_caps "4chan=800000,youtube=500000,reddit=500000"
```

`--source_caps` is **stratified by label**, so the offensive/clean ratio of each
source is preserved. Sources not listed are kept as-is. Use it with
`make_dataset.py` or directly:

```bash
.venv/bin/python scripts/make_dataset.py --api --4chan --youtube \
    --yt-api-key AIza... \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 \
    --source_caps "4chan=800000,youtube=500000,reddit=500000" \
    --device 0
```

```bash
# reddit only, from an already-downloaded monthly dump
.venv/bin/python scripts/make_dataset.py --steps reddit --dump RC_2025-01.zst --device 0

# 4chan + youtube only
.venv/bin/python scripts/make_dataset.py --steps community --4chan --youtube \
    --yt-api-key AIza... --device 0
```

## Using pre-built HF datasets (4chan, Reddit, etc.)

If a Hugging Face dataset already has the data you want, skip the fetchers and
point `11_hf_fetch.py` at it. It streams the dataset, applies the same regex
prefilter, and writes the standard candidate schema.

Real 4chan datasets on the Hub:

| dataset | rows | boards | notes |
|---|---|---|---|
| `vmfunc/4chan-pol-extensive` | ~50k+ | /pol/ | active + archived threads, `text`/`board`/`timestamp`/`replies` |
| `ylelauta/pol-4chan-augmented` | 134M | /pol/ | Perspective toxicity scores, `com`/`board`/`time`/`replies` |
| `fuzzy-g/4chan_pol_whole_ds` | 4M | /pol/ | train/val/test splits, `text`/`board`/`timestamp` |
| `u84u/4chan-pol` | 265M | /pol/ | raw posts, `com`/`time`/`no`/`replies` |

```bash
# 4chan /pol/ via HF (fast, no API limits)
.venv/bin/python scripts/11_hf_fetch.py \
    --dataset ylelauta/pol-4chan-augmented \
    --text-col com --source-col board --time-col time --score-col replies \
    --max_rows 1000000 --out data/4chan/raw

# via make_dataset
.venv/bin/python scripts/make_dataset.py --steps fetch-hf \
    --hf-dataset ylelauta/pol-4chan-augmented \
    --hf-text-col com --hf-source-col board --hf-time-col time --hf-score-col replies \
    --hf_out data/4chan/raw --max_rows 1000000
```

Column mapping flags (`--hf-text-col`, `--hf-source-col`, `--hf-time-col`,
`--hf-score-col`, `--hf-id-col`) let you adapt to any dataset schema.

When labeling with FTAN (`--ftan-model`), the script checkpoints every
`--checkpoint_every` rows (default 150,000) by writing `checkpoint.json` next to
the output shards, and **resumes by default** on the next run — already-kept rows
are skipped without re-running the GPU. Tune with `--checkpoint_every` and force
a clean start with `--no-resume` (or `HF_RESUME_OFF=1` in `run_4chan_ftan.sh`).
The checkpoint is keyed to the run's dataset/seed/threshold/grey/max_length, so
changing those starts fresh with a warning.

### Grey zone: rows the model is unsure about

Instead of forcing every row to 0/1, FTAN labels confident rows and leaves a
**grey zone** for manual review:

- `p_off >= --ftan-grey-high` (default 0.70) -> `label 1` (offensive)
- `p_off <= --ftan-grey-low`  (default 0.30) -> `label 0` (clean)
- otherwise -> `label -1` (unsure)

After labeling, `make_dataset` checks the output shards. If any `-1` rows
remain (default `--hf-exit-on-unsure`):

1. It writes `data/4chan/raw/manual_check.csv` (`text, origin_label, label`).
2. It prints a summary and **exits** — fix the `label` column to `0` or `1`
   (either edit the CSV or the parquet in place), then re-run the same command.
   The fixes are applied on the next run; `-1` rows never reach training.

Alternatives:

- `--hf-fallback-regex` (or `HF_FALLBACK_REGEX=1`): instead of stopping,
  classify the `-1` rows with regex (`attack` -> 1, `emotional`/`clean` -> 0,
  regex-grey dropped) and continue.
- `--hf-no-exit-on-unsure` (or `HF_EXIT_ON_UNSURE_OFF=1`): keep going even if
  `-1` rows remain — they are then excluded by `merge-ftan`.

### FTAN-labeled candidates -> community data

The `merge-ftan` step (included in `community`/`all` presets and
`run_4chan_ftan.sh`) appends FTAN-labeled candidates from `--hf_out`
(e.g. `data/4chan/raw`) into `data/raw/community.parquet`, preserving the FTAN
labels — no regex re-bucketing. Rows still labeled `-1` are excluded, and the
`--source_caps` subsample (e.g. `4chan=800000`) then applies as usual.

```bash
# label with FTAN, stop for manual review of unsure rows, then continue
bash scripts/run_4chan_ftan.sh          # first pass: labels, dumps manual_check.csv, exits
# ...edit data/4chan/raw/manual_check.csv (label -> 0/1)...
bash scripts/run_4chan_ftan.sh          # resumes, applies fixes, merges + builds dataset
```

## License / disclaimer

Sources have varied licenses (CC0 for Jigsaw train; research terms for Davidson /
HateXplain; CC-BY-SA for ToxiGen). The dataset **contains raw offensive language** and
is intended for moderation research only.
