# ftan-2.0

A large-scale, obfuscation-aware **offensive / clean English text dataset** for
training moderation / profanity classifiers, published as a Hugging Face
`DatasetDict` — plus the full pipeline that builds it (community-data fetchers,
curation, training, inference).

The dataset's core idea: real moderation input is rarely plain text. Users
write `f4gg0t`, `f u c k`, `f**k`, `ｆｕｃｋ` and Cyrillic homoglyphs. ftan-2.0
mutates **both** offensive and clean rows with an obfuscation engine, so the
model learns that leet-speak / censoring / homoglyphs are *orthographic
variation* — not a *semantic signal*.

> This README is a landing page. Detailed, script-by-script documentation lives
> in the [`docs/`](docs/00-overview.md) folder.

## Quickstart

```bash
# 0. setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. build the whole dataset with the orchestrator (download -> ... -> export)
.venv/bin/python scripts/make_dataset.py

# 2. train a classifier on it
.venv/bin/python scripts/train.py

# 3. run inference
printf "you are a f4gg0t and i hate you\nthanks for the help\n" \
    | .venv/bin/python scripts/predict.py --model data/final/model/model
```

## Documentation index

| doc | contents |
|---|---|
| [Overview](docs/00-overview.md) | what it is, pipeline diagram, schema, splits, design notes |
| [Setup & installation](docs/01-setup.md) | requirements, install, project structure |
| [Core pipeline (01–06)](docs/02-core-pipeline.md) | download, normalize, mutate, dedup, split, export — script by script |
| [Community data (07–11)](docs/03-community-data.md) | Reddit / 4chan / YouTube / HF fetchers + curate into banks |
| [Orchestrator](docs/04-orchestrator.md) | `make_dataset.py` steps, presets, every flag, grey-zone loop |
| [Training & inference](docs/05-training.md) | `train.py`, `finalize.py`, `predict.py`, `make_balanced.py` |
| [Publishing](docs/06-publishing.md) | pushing the dataset/model to the Hub |
| [Reference & FAQ](docs/07-reference.md) | shared modules, helper scripts, troubleshooting |

## Scripts at a glance

| script | stage | output |
|---|---|---|
| `01_download.py` | download core sources | `data/raw/*.parquet` |
| `02_normalize.py` | merge + exact-dedupe + filter | `data/processed/normalized.parquet` |
| `03_mutate.py` | obfuscation mutations | `data/processed/mutated.parquet` |
| `04_dedup.py` | orthographic clustering | `data/processed/deduped.parquet` |
| `05_split.py` | cluster-stratified splits | `data/processed/split.parquet` |
| `06_export.py` | export DatasetDict (+ Hub push) | `data/final/dataset` |
| `07_reddit_fetch.py` | reddit → candidates | `data/reddit/raw/*.parquet` |
| `08_reddit_curate.py` | candidates → banks | `data/reddit/banks/bank_{a,b,c}.parquet` |
| `09_4chan_fetch.py` | 4chan → candidates | `data/4chan/raw/*.parquet` |
| `10_youtube_fetch.py` | youtube → candidates | `data/youtube/raw/*.parquet` |
| `11_hf_fetch.py` | any HF dataset → candidates (+ FTAN labels) | `data/hf/raw/*.parquet` |
| `make_dataset.py` | one-shot orchestrator | everything above |
| `train.py` / `finalize.py` | fine-tune + export best checkpoint | `data/final/model/` |
| `predict.py` | CLI inference | stdout |
| `make_balanced.py` | rebalance the train split (~50/50) | rewritten `data/final/dataset` |
| `make_benchmark.py` | export held-out test rows → benchmark parquet | `data/final/benchmark/benchmark.parquet` |
| `benchmark.py` | score one/more models (local or HF Hub, incl. multi-class like `KoalaAI/Text-Moderation`) on the benchmark, compare | `data/final/benchmark/results.json` |

## Typical larger workflows

```bash
# core + reddit expansion in one pass
.venv/bin/python scripts/make_dataset.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0

# 4chan + youtube, then cap the source mix before the core pipeline
.venv/bin/python scripts/make_dataset.py --api --4chan --youtube \
    --yt-api-key AIza... \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 \
    --source_caps "4chan=800000,youtube=500000,reddit=500000" \
    --device 0

# label 4chan via HF dataset + FTAN, review unsure rows, merge, build, train
bash scripts/run_4chan_ftan.sh
```

## Output schema

`text` (str), `label` (1=offensive / 0=clean), `source`, `origin_label`,
`split_origin`, `mutated` (0/1), `variant`, `cluster` (orthographic-dup id for
safe splitting).

Splits: `train`, `validation`, `test`, plus `test_obfuscated` (mutated test
rows for robustness evaluation). Current build (~1.37M rows):

| split | rows | offensive | clean |
|---|---|---|---|
| train | 1,320,712 | 559,852 | 760,860 |
| validation | 20,001 | 9,996 | 10,005 |
| test | 30,001 | 14,998 | 15,003 |
| test_obfuscated | 14,084 | 10,793 | 3,291 |

## Sources

| source | id | role |
|---|---|---|
| Jigsaw Toxic Comment | `tcapelle/jigsaw-toxic-comment-classification-challenge` | offensive + clean (in-domain) |
| Davidson et al. | `contemmcm/hate-speech-and-offensive-language` | offensive tweets + clean |
| ToxiGen | `toxigen/toxigen-data` | subtle toxicity (machine-generated, human scores) |
| HateXplain | `Hate-speech-CNERG/hatexplain` | held-out test + train |
| Wikipedia | `wikimedia/wikipedia` 20231101.en | neutral filler |

Plus optional community data (Reddit / 4chan / YouTube / any HF dataset) —
see the [community-data docs](docs/03-community-data.md).

## License / disclaimer

Sources have varied licenses (CC0 for Jigsaw train; research terms for
Davidson / HateXplain; CC-BY-SA for ToxiGen). The dataset **contains raw
offensive language** and is intended for moderation research only.