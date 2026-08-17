# Overview

ftan-2.0 is a large-scale, **obfuscation-aware** English offensive / clean text
dataset for training moderation and profanity classifiers, plus a full
training / inference stack. It is published as a Hugging Face `DatasetDict`.

The project solves a specific problem that plain profanity datasets miss:
real-world moderation input is rarely plain text. Users write `f4gg0t`,
`f u c k`, `f**k`, `ｆｕｃｋ` and Cyrillic homoglyphs. If a classifier only
sees clean words, it will either miss censored profanity or (worse) flag
innocent leet-speak like `ex4mp1e` as offensive.

## What this repo does

1. **Builds a ~1.37M-row binary dataset** (`1 = offensive`, `0 = clean`) by
   merging public sources, then mutating rows with an obfuscation engine.
2. **Collects community data** (Reddit, 4chan, YouTube, or any Hugging Face
   dataset) to teach the model the difference between *swearing at a person*
   (`fuck you`) and *swearing about a situation* (`fuck, I lost my keys`).
3. **Trains and evaluates** a DistilBERT-style classifier on the result,
   including a dedicated `test_obfuscated` split for robustness.
4. **Publishes** the dataset (and optionally the model) to the Hugging Face Hub.

## The pipeline at a glance

```
                    ┌────────────────────────────────────────────┐
                    │   Community data (optional)                │
                    │                                            │
   core sources ────► 01_download ──► 07/09/10/11_fetch ──► 08_curate
   (HF datasets)    data/raw/*.parquet   data/*/raw/       data/*/banks
                    │                            │
                    └──────────────┬─────────────┘
                                   ▼
                        merge ──► data/raw/community.parquet
                                   │   (subsample via --source_caps)
                                   ▼
              02_normalize ──► data/processed/normalized.parquet
                                   │
              03_mutate    ──► data/processed/mutated.parquet
                                   │   (obfuscation engine)
              04_dedup     ──► data/processed/deduped.parquet
                                   │   (orthographic clusters)
              05_split     ──► data/processed/split.parquet
                                   │   (cluster-stratified)
              06_export    ──► data/final/dataset  (HF DatasetDict)
                                   │
              train / finalize ──► data/final/model  + eval_metrics.json
```

| stage | script | output |
|---|---|---|
| download core sources | `01_download.py` | `data/raw/*.parquet` |
| merge + normalize | `02_normalize.py` | `data/processed/normalized.parquet` |
| obfuscation mutations | `03_mutate.py` | `data/processed/mutated.parquet` |
| cluster dedup | `04_dedup.py` | `data/processed/deduped.parquet` |
| cluster-stratified split | `05_split.py` | `data/processed/split.parquet` |
| export DatasetDict | `06_export.py` | `data/final/dataset` + `stats.json` + `dataset_card.md` |
| reddit fetch | `07_reddit_fetch.py` | `data/reddit/raw/candidates_*.parquet` |
| curate into banks | `08_reddit_curate.py` | `data/reddit/banks/bank_{a,b,c}.parquet` |
| 4chan fetch | `09_4chan_fetch.py` | `data/4chan/raw/candidates_*.parquet` |
| youtube fetch | `10_youtube_fetch.py` | `data/youtube/raw/candidates_*.parquet` |
| HF dataset fetch | `11_hf_fetch.py` | `data/hf/raw/candidates_*.parquet` |
| one-shot orchestrator | `make_dataset.py` | runs any combination of the above |
| train | `train.py` | `data/final/model/checkpoints/` |
| finalize / evaluate | `finalize.py` | `data/final/model/model` + `eval_metrics.json` |
| inference | `predict.py` | stdout |
| rebalance train split | `make_balanced.py` | rewrites `data/final/dataset` |

You can run every stage by hand (great for learning what each step does), or
drive the whole thing with `make_dataset.py` (see
[the orchestrator page](04-orchestrator.md)).

## Data layout

| path | contents |
|---|---|
| `data/raw/` | downloaded core sources, one parquet each, unified schema |
| `data/processed/` | `normalized` → `mutated` → `deduped` → `split` parquet chain |
| `data/final/` | `dataset/` (DatasetDict), `model/`, `stats.json`, `dataset_card.md` |
| `data/reddit/` | fetched candidates (`raw/`) and curated banks (`banks/`) |
| `data/4chan/raw/` | 4chan candidates, `checkpoint.json`, `manual_check.csv` |
| `data/youtube/raw/` | youtube candidates |
| `data/hf/raw/` | candidates from arbitrary HF datasets |
| `data/seeds/` | `ldnoobw_en.txt`, `extra_en.txt` — the offensive seed-word list |

Everything under `data/final/` (and the raw candidates) is gitignored — it is
reproduced by the pipeline, not stored in the repo.

## Output schema

The final dataset has these columns:

| column | type | meaning |
|---|---|---|
| `text` | string | input sentence |
| `label` | int8 | `1` = offensive, `0` = clean |
| `source` | string | originating dataset (`jigsaw`, `davidson`, `toxigen`, `hatexplain`, `wikipedia`, `reddit`, `4chan`, `youtube`, …) |
| `origin_label` | string | label as given by the source (`toxic`, `clean`, `hate`, `offensive`, `neutral`, `ftan_0.982`, …) |
| `split_origin` | string | original train/test split of the source |
| `mutated` | int8 | `1` = obfuscation-engine variant, `0` = original |
| `variant` | int8 | variant index within the source row |
| `cluster` | int64 | near-duplicate cluster id (splitting is done at this level) |

## Splits

Current build (~1.37M rows):

| split | rows | offensive | clean |
|---|---|---|---|
| train | 1,320,712 | 559,852 | 760,860 |
| validation | 20,001 | 9,996 | 10,005 |
| test | 30,001 | 14,998 | 15,003 |
| test_obfuscated | 14,084 | 10,793 | 3,291 |

- `test` is a balanced sample drawn from **held-out sources** (HateXplain plus
  Jigsaw's `balanced_test`).
- `test_obfuscated` holds the mutated rows of the test set — the robustness
  evaluation split. The model never sees these variants during training, so
  its score on them measures real-world obfuscation tolerance.

## Design notes

### The obfuscation engine (`common.py`)

`Mutator` applies eight word-level styles, seeded and deterministic:

| style | example |
|---|---|
| leet | `fuck` → `fuсk`/`fuck` → `f4ck` |
| separators | `fuck` → `f.u.c.k` |
| censor | `fuck` → `f***` |
| repeat | `fuck` → `fuuuuck` |
| case | `fuck` → `FuCk` |
| homoglyph | `fuck` → `fисk` (Cyrillic lookalikes) |
| fullwidth | `fuck` → `ｆｕｃｋ` |
| combo | stacked transforms |

It mutates **both classes**:
- **offensive rows** get `--offensive_variants` mutated copies of their curse
  words;
- **clean rows** get innocuous words mutated (`example` → `ex4mp1e`) in a
  fraction (`--clean_variant_frac`) of rows.

This is the key trick: the model learns that leet-speak / censoring /
homoglyphs are *orthographic variation*, not a *semantic signal*. Obfuscation
alone never means "offensive".

### Orthographic clustering (`04_dedup.py`)

Each text is canonicalized (homoglyphs → latin, leet digits → letters,
symbols stripped, repeated chars collapsed), and rows sharing a canonical form
share a `cluster` id. Splitting happens at the **cluster level**, so a
sentence and all its obfuscated variants never leak across
train/validation/test.

### Why three banks for community data

The core dataset labels "any profanity = offensive". To teach the model the
difference between insults and expletives, community comments are split into:

| bank | label | meaning |
|---|---|---|
| `bank_a` | 1 | profanity/insult **aimed at a person** |
| `bank_b` | 0 | expletive profanity with **no addressee** — the hard negatives |
| `bank_c` | 0 | no profanity, no insult |
| `manual_check` | ? | FTAN cannot decide — review by hand |

See [community data](03-community-data.md) for the full flow.

## Further reading

- [Setup & installation](01-setup.md)
- [Core pipeline (scripts 01–06)](02-core-pipeline.md)
- [Community data (scripts 07–11)](03-community-data.md)
- [The orchestrator (`make_dataset.py`)](04-orchestrator.md)
- [Training & inference](05-training.md)
- [Publishing to the Hub](06-publishing.md)
- [Reference & FAQ](07-reference.md)
