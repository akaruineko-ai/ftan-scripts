# The Orchestrator (`make_dataset.py`)

`make_dataset.py` runs the whole ftan-2.0 pipeline in one command: it resolves
a plan of steps, runs each in dependency order with subprocesses, streams
their output, and prints a dataset summary at the end.

```bash
.venv/bin/python scripts/make_dataset.py [flags]
```

## Steps and presets

Steps run in this order:

```
download → fetch → fetch-4chan → fetch-hf → fetch-youtube → curate → merge
        → merge-ftan → subsample → normalize → mutate → dedup → split
        → export → train → finalize
```

`--steps` accepts a comma-separated list **or** a preset name:

| preset | steps |
|---|---|
| `core` | `download,normalize,mutate,dedup,split,export` |
| `reddit` | `fetch,curate,merge,subsample,normalize,mutate,dedup,split,export` |
| `community` | `fetch,fetch-4chan,fetch-hf,fetch-youtube,curate,merge,merge-ftan,subsample,normalize,mutate,dedup,split,export` |
| `all` | `download` + `community` |
| `train` | `train,finalize` |

```bash
# the whole thing, including community fetchers
.venv/bin/python scripts/make_dataset.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0

# a specific plan: presets are core / reddit / community / all / train,
# or any comma list
.venv/bin/python scripts/make_dataset.py --steps reddit --dump RC_2025-01.zst
.venv/bin/python scripts/make_dataset.py --steps all,train --device 0
.venv/bin/python scripts/make_dataset.py --steps 4chan,merge-ftan,export
```

### Skips are automatic

- `download` is skipped automatically if `data/raw/` already has sources
  (force with `--re-download`).
- `fetch` (reddit) only runs with `--api` or `--dump`.
- `fetch-4chan` only runs with `--4chan`.
- `fetch-youtube` only runs with `--youtube` **and** `--yt-api-key`.
- `fetch-hf` only runs with `--hf-dataset`.
- `curate` only runs if candidate parquets exist in the raw dirs.
- `merge` / `merge-ftan` only run if banks / FTAN shards exist.
- `subsample` only runs if `--source_caps` is given.

### Manual skip flags

| flag | effect |
|---|---|
| `--skip-download` | never run the download step |
| `--skip-fetch` | skip the reddit `fetch` step |
| `--skip-curate` | skip the `curate` step |

---

## Full flag reference

### Global

| flag | default | meaning |
|---|---|---|
| `--steps` | `all` | comma-separated steps or a preset name |
| `--skip-download` | off | don't run download |
| `--skip-fetch` | off | don't run reddit fetch |
| `--skip-curate` | off | don't run curate |
| `--re-download` | off | force download even if `data/raw` has parquets |
| `--seed` | `42` | global RNG seed (mutate, split, fetch, subsample) |

### download

| flag | default | meaning |
|---|---|---|
| `--sources` | `None` | which sources to download (space-separated list) |
| `--wiki_cap` | `350000` | wikipedia clean-filler sentence cap |

### reddit fetch (`fetch`)

| flag | default | meaning |
|---|---|---|
| `--api` | off | use the Arctic Shift API |
| `--dump` | `None` | local path or URL of a `.zst` / `.jsonl` dump |
| `--subreddits` | `None` | comma-separated subreddits (API mode) |
| `--after` | `2024-01-01` | only comments after this date (API) |
| `--before` | `None` | only comments before this date (API) |
| `--max_pages` | `5000` | max API pages |
| `--max_rows` | `5000000` | stop after this many kept rows |
| `--min_words` | `4` | drop shorter comments |
| `--clean_sample_frac` | `0.05` | keep-clean probability |
| `--require_latin` / `--no-require_latin` | on | drop non-Latin comments |
| `--api_delay` | `0.3` | seconds between API pages |
| `--retries` | `6` | retries per request |
| `--raw_out` | `data/reddit/raw` | candidate output dir |
| `--flush_rows` | `500000` | rows per parquet shard |

### 4chan fetch (`fetch-4chan`)

| flag | default | meaning |
|---|---|---|
| `--4chan` | off | enable the 4chan fetch step |
| `--chan-mode` | `live` | `live` (a.4cdn.org) or `archive` (4plebs) |
| `--boards` | `None` | comma-separated boards (live) |
| `--chan-board` | `None` | single board (archive) |
| `--chan-search` | `None` | body search term (archive) |
| `--chan_delay` | `1.0` | seconds between thread fetches |
| `--chan-repeats` | `0` | re-scan catalogs N times (`0` = off) |
| `--chan-repeat-delay` | `300.0` | seconds between catalog re-scans |
| `--chan-threads-per-board` | `250` | max threads per board per scan |
| `--chan_out` | `data/4chan/raw` | candidate output dir |

### HF dataset fetch (`fetch-hf`)

| flag | default | meaning |
|---|---|---|
| `--hf-dataset` | `None` | HF dataset id to load as candidates |
| `--hf-split` | `train` | split to load |
| `--hf-text-col` | `None` | comment body column |
| `--hf-source-col` | `None` | board/source column |
| `--hf-time-col` | `None` | timestamp column |
| `--hf-score-col` | `None` | score column |
| `--hf-id-col` | `None` | id column |
| `--hf-source-name` | `None` | override `source` value |
| `--hf_out` | `data/hf/raw` | candidate output dir |
| `--hf-ftan-model` | `None` | label rows with FTAN (path to model) |
| `--hf-ftan-device` | `None` | CUDA device, e.g. `0` |
| `--hf-ftan-batch-size` | `256` | FTAN batch size |
| `--hf-ftan-threshold` | `0.5` | keep row if p_off ≥ threshold |
| `--hf-ftan-grey-low` | `0.30` | p_off below → label `0` |
| `--hf-ftan-grey-high` | `0.70` | p_off above → label `1` |
| `--hf-ftan-max-length` | `128` | token truncation length |
| `--hf-exit-on-unsure` / `--hf-no-exit-on-unsure` | on | stop if label `-1` rows remain |
| `--hf-fallback-regex` | off | regex-classify `-1` rows instead of stopping |
| `--hf-checkpoint-every` | `150000` | rows between FTAN checkpoints |
| `--hf-resume` / `--hf-no-resume` | on | resume from checkpoint / shards |

### YouTube fetch (`fetch-youtube`)

| flag | default | meaning |
|---|---|---|
| `--youtube` | off | enable the YouTube fetch step |
| `--yt-api-key` | `None` | YouTube Data API v3 key (required) |
| `--yt-query` | `"cat videos"` | video search query |
| `--yt-max-videos` | `20` | max videos |
| `--yt-max-comments` | `200` | max comments per video |
| `--yt-clean-frac` | `1.0` | keep-clean probability |
| `--yt_out` | `data/youtube/raw` | candidate output dir |

### curate (`curate`)

| flag | default | meaning |
|---|---|---|
| `--raw` | `data/reddit/raw` | candidate dir(s)/file(s), comma-separated (auto-filled from the fetch dirs that ran) |
| `--banks_out` | `data/reddit/banks` | bank output dir |
| `--model` | `data/final/model/model` | FTAN model |
| `--device` | `None` | CUDA device id, e.g. `0` |
| `--batch_size` | `256` | FTAN batch size |
| `--target_a` | `1000000` | `bank_a` (attack) cap |
| `--target_b` | `1000000` | `bank_b` (emotional) cap |
| `--target_c` | `1000000` | `bank_c` (clean) cap |
| `--grey_low` | `0.30` | FTAN clean threshold |
| `--grey_high` | `0.70` | FTAN offensive threshold |
| `--verify_frac` | `0.0` | fraction of banks re-scored to catch regex misses |
| `--verify_attack_conf` | `0.40` | low-conf attack → `manual_check` |
| `--verify_clean_conf` | `0.90` | low-conf clean → `manual_check` |

### merge / merge-ftan / subsample

`merge` reads `bank_{a,b,c}.parquet` from `--banks_out` and writes
`data/raw/community.parquet` (label `-1` manual rows excluded, deduped on
text, unified schema).

`merge-ftan` appends FTAN-labeled candidates (label `0`/`1`) from `--hf_out`
into `data/raw/community.parquet`, preserving FTAN labels — no regex
re-bucketing. Rows still labeled `-1` are excluded.

| flag | default | meaning |
|---|---|---|
| `--source_caps` | `None` | comma-separated `source=target` caps, e.g. `4chan=800000,youtube=500000,reddit=500000` |
| `--subsample_seed` | `42` | seed for the stratified downsample |

`--source_caps` is **stratified by label**, so the offensive/clean ratio of
each source is preserved. Sources not listed are kept as-is.

### normalize / mutate / split / export

| flag | default | meaning |
|---|---|---|
| `--min_len` | `3` | normalize min tokens |
| `--max_len` | `512` | normalize max tokens |
| `--offensive_variants` | `3` | obfuscation copies per offensive row (`0` = none) |
| `--clean_variant_frac` | `0.4` | fraction of clean rows mutated |
| `--max_pos_per_source` | `0` | per-source offensive cap (`0` = uncapped) |
| `--max_neg_per_source` | `0` | per-source clean cap (`0` = uncapped) |
| `--test_target` | `30000` | test split target |
| `--val_target` | `20000` | validation split target |
| `--eval_obfuscation` / `--no-eval_obfuscation` | on | add `test_obfuscated` split |
| `--hub_id` | `None` | push dataset to this Hub repo id |
| `--private` | off | make the Hub repo private |

### train / finalize

| flag | default | meaning |
|---|---|---|
| `--data_dir` | `data/final/dataset` | dataset to train on |
| `--output_dir` | `data/final/model` | model output dir |
| `--max_train_rows` | `None` | subsample train rows (`None` = all) |
| `--max_val_rows` | `None` | subsample validation rows |
| `--max_length` | `auto` | truncation length (`auto` = p95 of a sample) |
| `--epochs` | `1` | training epochs |
| `--lr` | `2e-5` | learning rate |
| `--fp16` | off | mixed precision |
| `--push_to_hub` | `None` | push the trained model to this repo id |

---

## Controlling the source mix

By default each fetcher is capped by its own `--max_rows`, and curate samples
each bank down to `--target_a/b/c`. To get *more* YouTube / 4chan in the final
dataset, raise the fetcher caps **and** cap the other sources at the
`subsample` step so the core pipeline doesn't drown them out:

```bash
# fetch more raw rows from 4chan / youtube
.venv/bin/python scripts/make_dataset.py --steps fetch-4chan,fetch-youtube \
    --4chan --boards b,pol,r9k,v --chan-threads-per-board 500 \
    --youtube --yt-api-key AIza... \
    --yt-max-videos 200 --yt-max-comments 500

# after curate: cap each source BEFORE the core pipeline normalizes/mutates
.venv/bin/python scripts/make_dataset.py --steps merge,subsample,normalize,mutate,dedup,split,export \
    --source_caps "4chan=800000,youtube=500000,reddit=500000"
```

## The grey-zone review loop

With `--hf-ftan-model`, the `fetch-hf` step labels rows and leaves unsure
(`-1`) rows for review. By default the run **stops** and writes
`manual_check.csv`:

```bash
# first pass: labels, dumps manual_check.csv, exits
bash scripts/run_4chan_ftan.sh
# ...edit data/4chan/raw/manual_check.csv (label -> 0/1)...
# re-run the same command: resumes, applies fixes, merges + builds dataset
bash scripts/run_4chan_ftan.sh
```

Alternatives to stopping:

- `--hf-fallback-regex` — classify the `-1` rows with regex and continue.
- `--hf-no-exit-on-unsure` — keep going; `-1` rows are excluded by
  `merge-ftan`.

## `run_4chan_ftan.sh` — the 4chan + FTAN wrapper

A convenience wrapper that labels `fuzzy-g/4chan_pol_whole_ds` with FTAN and
builds the dataset. Everything is configurable via environment variables:

```bash
# default run (fetch-hf,merge-ftan,subsample,normalize,mutate,dedup,split,export)
bash scripts/run_4chan_ftan.sh

# customize
HF_DATASET=ylelauta/pol-4chan-augmented \
HF_TEXT=com HF_SOURCE=board HF_TIME=time HF_SCORE=replies HF_ID=no \
FTAN_DEVICE=0 MAX_ROWS=1000000 SOURCE_CAPS=4chan=500000 \
bash scripts/run_4chan_ftan.sh
```

| env var | default |
|---|---|
| `STEPS` (arg 1) | `fetch-hf,merge-ftan,subsample,normalize,mutate,dedup,split,export` |
| `HF_DATASET` | `fuzzy-g/4chan_pol_whole_ds` |
| `HF_SPLIT` | `train` |
| `HF_TEXT` / `HF_SOURCE` / `HF_TIME` / `HF_SCORE` / `HF_ID` | `text` / `flag` / `__index_level_0__` ×3 |
| `HF_SOURCE_NAME` | `4chan` |
| `FTAN_MODEL` | `data/final/model/model` |
| `FTAN_DEVICE` | `0` |
| `FTAN_THRESHOLD` | `0.6` |
| `FTAN_BATCH` | `1024` |
| `FTAN_MAX_LENGTH` | `64` |
| `FTAN_GREY_LOW` / `FTAN_GREY_HIGH` | `0.30` / `0.70` |
| `CHECKPOINT_EVERY` | `150000` |
| `MAX_ROWS` | `2000000` |
| `SOURCE_CAPS` | `4chan=800000` |
| `HF_OUT` | `data/4chan/raw` |
| `HF_RESUME_OFF` (set to any value) | pass `--hf-no-resume` |
| `HF_EXIT_ON_UNSURE_OFF` | pass `--hf-no-exit-on-unsure` |
| `HF_FALLBACK_REGEX` | pass `--hf-fallback-regex` |

## Common one-shot examples

```bash
# full default run (core sources; reddit auto-skipped unless --api/--dump)
.venv/bin/python scripts/make_dataset.py --device 0

# core + reddit, 1M per bank
.venv/bin/python scripts/make_dataset.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0

# reddit only, from an already-downloaded monthly dump
.venv/bin/python scripts/make_dataset.py --steps reddit --dump RC_2025-01.zst --device 0

# 4chan + youtube only
.venv/bin/python scripts/make_dataset.py --steps community --4chan --youtube \
    --yt-api-key AIza... --device 0

# reuse already-fetched candidates (skip fetch, keep curate)
.venv/bin/python scripts/make_dataset.py --skip-fetch --device 0

# build the dataset, then train the model on it
.venv/bin/python scripts/make_dataset.py --steps all,train --device 0

# label 4chan via HF + FTAN, stop for manual review of unsure rows
bash scripts/run_4chan_ftan.sh
```