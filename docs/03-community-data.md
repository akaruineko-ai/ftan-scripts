# Community Data (scripts 07–11)

The core dataset is trained mostly on "every profanity counts as offensive"
sources. To teach a model the difference between **swearing at a person**
(`fuck you`) and **swearing about a situation** (`fuck, I lost my keys`),
this part of the pipeline collects real forum comments and splits them into
banks.

## The bank scheme

| bank | bucket | label | meaning |
|---|---|---|---|
| `bank_a` | attack | 1 | profanity/insult aimed at a person (2nd-person pronoun in a small window, or an explicit pattern like `kys`, `fuck off`) |
| `bank_b` | emotional | 0 | expletive profanity with no addressee — the hard negatives FTAN gets wrong |
| `bank_c` | clean | 0 | no profanity, no insult |
| `manual_check` | grey / verify | ? | FTAN cannot decide (conf in `[grey_low, grey_high]`) or disagrees with regex — review by hand |

Three complementary sources:

| source | script | why | how |
|---|---|---|---|
| **Reddit** | `07_reddit_fetch.py` | huge volume, conflict subreddits | Arctic Shift API or monthly `.zst` dump |
| **4chan** | `09_4chan_fetch.py` | highest profanity density anywhere — fastest way to fill `bank_a`/`bank_b` | official keyless `a.4cdn.org` live API, or 4plebs archive JSON API |
| **YouTube** | `10_youtube_fetch.py` | casual register where nobody swears — clean filler for `bank_c` | YouTube Data API v3 (free key, 10k units/day) |

All fetchers write the **same candidate schema**, so any of them can feed the
curate step, and `source` (`reddit` / `4chan` / `youtube`) is preserved
through the banks into the final dataset:

```
text, source, subreddit, created_utc, score, id, category, profanity, insult, attack
```

### Regex prefiltering

Every fetcher applies a cheap regex prefilter (`reddit_vocab.prefilter_keep`)
at fetch time:

- **attack / grey / emotional** candidates are always kept;
- **clean** rows are kept with probability `--clean_sample_frac`
  (default `0.05`) so the clean bank doesn't balloon to the whole dump.

This keeps the on-disk parquets a small fraction of the raw source size.
The regexes tolerate light obfuscation (`f*ck`, `f4ck`, `f u c k`); heavier
mutations (homoglyphs, leet, fullwidth) are handled later by the FTAN model.

---

## 07_reddit_fetch.py — Reddit → candidates

Two data modes, mutually exclusive:

### `--api` — Arctic Shift API

The Pushshift successor (`arctic-shift.photon-reddit.com`). Pulls recent
comments from conflict-heavy subreddits.

```bash
.venv/bin/python scripts/07_reddit_fetch.py --api \
    --subreddits PublicFreakout,leagueoflegends,AmItheAsshole \
    --after 2024-01-01 --max_rows 5000000 --out data/reddit/raw
```

### `--dump` — monthly `.zst` dump

Stream a local (or HTTP(S)) comment dump. Download a monthly dump via torrent
first, then point at the file:

```bash
.venv/bin/python scripts/07_reddit_fetch.py --dump RC_2025-01.zst \
    --clean_sample_frac 0.05 --out data/reddit/raw
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--api` | off | fetch via the Arctic Shift API |
| `--dump` | `None` | local path or HTTP(S) URL of a dump (`.zst` / `.jsonl`) |
| `--subreddits` | default list | comma-separated subreddits (API mode) |
| `--after` | `2024-01-01` | only comments after this date (API mode) |
| `--before` | `None` | only comments before this date (API mode) |
| `--max_pages` | `5000` | max API pages |
| `--max_rows` | `5000000` | stop after this many kept rows |
| `--min_words` | `4` | drop comments with fewer words |
| `--clean_sample_frac` | `0.05` | probability of keeping a clean row |
| `--require_latin` / `--no-require_latin` | on | drop mostly non-Latin comments |
| `--api_delay` | `0.3` | seconds between API pages |
| `--retries` | `6` | retries per request |
| `--seed` | `42` | RNG seed |
| `--out` | `data/reddit/raw` | output directory |
| `--flush_rows` | `500000` | rows per parquet shard |

### Output

`data/reddit/raw/candidates_000.parquet`, `candidates_001.parquet`, … in the
candidate schema with `source="reddit"`.

---

## 08_reddit_curate.py — candidates → banks

Regex is **authoritative** for `attack` / `emotional` / `clean`. The FTAN
model (`data/final/model/model`) is only invoked on:

- **grey rows** (an insult word but no 2nd-person address) — its confidence
  splits them into `bank_a` / `bank_c` / `manual_check`;
- an optional random fraction of `bank_a` / `bank_c` when `--verify_frac > 0`,
  to catch regex misses.

```bash
.venv/bin/python scripts/08_reddit_curate.py \
    --raw data/reddit/raw,data/4chan/raw,data/youtube/raw \
    --target_a 1000000 --target_b 1000000 --target_c 1000000 --device 0
```

`--raw` accepts **several directories or files, comma-separated**, so one
curate run can mix reddit / 4chan / youtube candidates.

### Arguments

| flag | default | meaning |
|---|---|---|
| `--raw` | `data/reddit/raw` | candidate parquet dir(s)/file(s), comma-separated |
| `--out` | `data/reddit/banks` | output bank directory |
| `--model` | `data/final/model/model` | FTAN model directory |
| `--device` | auto | CUDA device id, e.g. `0` |
| `--batch_size` | `256` | FTAN batch size |
| `--target_a` | `1000000` | cap for `bank_a` (attack) |
| `--target_b` | `1000000` | cap for `bank_b` (emotional) |
| `--target_c` | `1000000` | cap for `bank_c` (clean) |
| `--grey_low` | `0.30` | FTAN p_off below this → clean |
| `--grey_high` | `0.70` | FTAN p_off above this → attack |
| `--verify_frac` | `0.0` | fraction of `bank_a`/`bank_c` to re-score for regex misses |
| `--verify_attack_conf` | `0.40` | low-conf attack below this → `manual_check` |
| `--verify_clean_conf` | `0.90` | high-conf clean below this → `manual_check` |
| `--seed` | `42` | RNG seed |

### Output

`data/reddit/banks/bank_a.parquet`, `bank_b.parquet`, `bank_c.parquet`,
`manual_check.parquet` — unified schema
(`text, label, source, origin_label, split_origin, subreddit, created_utc,
score, category, ftan_conf`).

The `merge` step in `make_dataset.py` converts these banks into
`data/raw/community.parquet`, where the core pipeline picks them up like any
other source.

---

## 09_4chan_fetch.py — 4chan → candidates

4chan has the highest profanity density of any public forum: a few boards
yield tens of thousands of attack/emotional candidates in a minute, with **no
API key**. Two modes:

### `--live` — official keyless API

Snapshots every currently-open thread on the given boards via `a.4cdn.org`:

```bash
.venv/bin/python scripts/09_4chan_fetch.py --live \
    --boards b,pol,r9k,v,gaming,trash --max_rows 2000000 --out data/4chan/raw
```

> Live mode only returns **currently-open** threads (~100–300 per board).
> To accumulate hundreds of thousands of rows over time, use `--repeat`:
> it re-scans the catalogs, dedupes against what it already kept, and appends.

```bash
.venv/bin/python scripts/09_4chan_fetch.py --live \
    --boards b,pol --max_rows 1000000 --repeat 100 --repeat_delay 300 \
    --out data/4chan/raw
```

### `--archive` — 4plebs archive

Historical bulk search via the 4plebs JSON API (no key; only mirrors some
boards: `adv b hr o pol s4s tg trv x y`):

```bash
.venv/bin/python scripts/09_4chan_fetch.py --archive \
    --board pol --search "fuck you" --out data/4chan/raw
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--live` | off | use the official a.4cdn.org live API |
| `--archive` | off | use the 4plebs archive JSON API |
| `--boards` | default list | comma-separated boards (live mode) |
| `--board` | `None` | single board (archive mode) |
| `--search` | `None` | body search term (archive mode) |
| `--threads_per_board` | `250` | max threads fetched per board per scan (live) |
| `--delay` | `1.0` | seconds between thread fetches |
| `--repeat` | `0` | re-scan catalogs this many times (`0` = off) |
| `--repeat_delay` | `300.0` | seconds between catalog re-scans |
| `--min_words` | `4` | drop posts with fewer words |
| `--clean_sample_frac` | `0.05` | probability of keeping a clean row |
| `--require_latin` / `--no-require_latin` | on | drop mostly non-Latin posts |
| `--max_rows` | `2000000` | stop after this many kept rows |
| `--per_page` | `100` | archive page size |
| `--max_pages` | `200` | archive pagination cap |
| `--api_delay` | `0.4` | seconds between archive pages |
| `--retries` | `4` | retries per request |
| `--seed` | `42` | RNG seed |
| `--out` | `data/4chan/raw` | output directory |
| `--flush_rows` | `500000` | rows per parquet shard |

### Output

`data/4chan/raw/candidates_*.parquet`, candidate schema with
`source="4chan"`, `subreddit=<board>`.

---

## 10_youtube_fetch.py — YouTube → candidates

YouTube comment sections (cat videos, cooking, gaming let's-plays) are a
relaxed register where people almost never swear at each other — the direct
counterweight to flame-heavy Reddit / 4chan, and the natural source of clean
filler for `bank_c`.

Requires a **YouTube Data API v3 key** (free tier: 10k units/day).

```bash
.venv/bin/python scripts/10_youtube_fetch.py \
    --api_key AIza... --query "cat videos" \
    --max_videos 20 --max_comments_per_video 200 --out data/youtube/raw
```

Because the whole point is *clean* filler, every clean comment is kept by
default (`--clean_sample_frac 1.0`).

### Arguments

| flag | default | meaning |
|---|---|---|
| `--api_key` | `None` | YouTube Data API v3 key (required) |
| `--query` | `"cat videos"` | search query for videos |
| `--published_before` | `None` | only videos published before this date |
| `--max_videos` | `20` | max videos to fetch comments from |
| `--max_comments_per_video` | `200` | max comments per video |
| `--min_words` | `3` | drop comments with fewer words |
| `--clean_sample_frac` | `1.0` | probability of keeping a clean row |
| `--require_latin` / `--no-require_latin` | on | drop mostly non-Latin comments |
| `--max_rows` | `1000000` | stop after this many kept rows |
| `--api_delay` | `0.4` | seconds between API calls |
| `--retries` | `4` | retries per request |
| `--seed` | `42` | RNG seed |
| `--out` | `data/youtube/raw` | output directory |
| `--flush_rows` | `500000` | rows per parquet shard |

> **Quota math:** a search costs 100 units, `commentThreads.list` costs 1 unit
> per page. `--max_videos 200 --max_comments_per_video 500` ≈ 3–5k units per
> run — well inside the free 10k/day.

### Output

`data/youtube/raw/candidates_*.parquet`, candidate schema with
`source="youtube"`, `subreddit=<video title> [videoId]`.

---

## 11_hf_fetch.py — any Hugging Face dataset → candidates

If a public HF dataset already has the data you want (4chan archives, Reddit
dumps, …), skip the platform fetchers and stream it through the same
regex-prefilter instead.

```bash
.venv/bin/python scripts/11_hf_fetch.py \
    --dataset ylelauta/pol-4chan-augmented \
    --text-col com --source-col board --time-col time --score-col replies \
    --max_rows 1000000 --out data/4chan/raw
```

### Real 4chan datasets on the Hub

| dataset | rows | boards | notes |
|---|---|---|---|
| `vmfunc/4chan-pol-extensive` | ~50k+ | /pol/ | active + archived threads, `text`/`board`/`timestamp`/`replies` |
| `ylelauta/pol-4chan-augmented` | 134M | /pol/ | Perspective toxicity scores, `com`/`board`/`time`/`replies` |
| `fuzzy-g/4chan_pol_whole_ds` | 4M | /pol/ | train/val/test splits, `text`/`board`/`timestamp` |
| `u84u/4chan-pol` | 265M | /pol/ | raw posts, `com`/`time`/`no`/`replies` |

### Column mapping

The flags let you adapt to any dataset schema:

| flag | default | meaning |
|---|---|---|
| `--text-col` | `text` | the comment body column |
| `--source-col` | `board` | subreddit/board column |
| `--time-col` | `timestamp` | timestamp column |
| `--score-col` | `replies` | score/likes column |
| `--id-col` | `no` | post id column |
| `--source-name` | `None` | override `source` value in output |

### Plain arguments

| flag | default | meaning |
|---|---|---|
| `--dataset` | *(required)* | HF dataset id, e.g. `ylelauta/pol-4chan-augmented` |
| `--split` | `train` | split to load |
| `--min_words` | `4` | drop rows with fewer words |
| `--clean_sample_frac` | `0.05` | probability of keeping a clean row |
| `--require_latin` / `--no-require_latin` | on | drop mostly non-Latin rows |
| `--max_rows` | `2000000` | stop after this many kept rows |
| `--streaming` / `--no-streaming` | off | stream instead of downloading fully |
| `--seed` | `42` | RNG seed |
| `--out` | `data/hf/raw` | output directory |
| `--flush_rows` | `500000` | rows per parquet shard |

### FTAN labeling (`--ftan-model`)

Pass a model path to label rows with FTAN instead of regex (regex prefilter is
still applied first):

```bash
.venv/bin/python scripts/11_hf_fetch.py \
    --dataset fuzzy-g/4chan_pol_whole_ds --split train \
    --text-col text --source-col board --time-col timestamp \
    --ftan-model data/final/model/model --ftan-device 0 \
    --ftan-threshold 0.6 --ftan-max-length 64 \
    --max_rows 2000000 --out data/4chan/raw
```

| flag | default | meaning |
|---|---|---|
| `--ftan-model` | `None` | path to the FTAN model to label rows |
| `--ftan-device` | auto | CUDA device, e.g. `0` |
| `--ftan-batch-size` | `1024` | FTAN batch size |
| `--ftan-max-length` | `128` | token truncation length |
| `--ftan-threshold` | `0.6` | row is kept as a candidate if p_off ≥ threshold |
| `--ftan-grey-low` | `0.30` | p_off below → label `0` (confident clean) |
| `--ftan-grey-high` | `0.70` | p_off above → label `1` (confident offensive) |
| `--checkpoint_every` | `150000` | rows between FTAN checkpoints |
| `--resume` / `--no-resume` | on | resume from checkpoint / existing shards |

**Checkpointing & resume.** The script writes `checkpoint.json` next to the
output shards every `--checkpoint_every` rows, keyed to the run's
dataset/seed/threshold/grey/max_length. On the next run it **resumes by
default** — already-kept rows are skipped without re-running the GPU. Change
one of the keyed settings and it starts fresh (with a warning). Force a clean
start with `--no-resume`.

### The grey zone

Instead of forcing every row to 0/1, FTAN labels confident rows and leaves a
grey zone for manual review:

- `p_off >= --ftan-grey-high` → label `1`
- `p_off <= --ftan-grey-low` → label `0`
- otherwise → label `-1` (unsure)

After labeling, `make_dataset` checks the output shards. If any `-1` rows
remain (default `--hf-exit-on-unsure`):

1. It writes `data/4chan/raw/manual_check.csv` (`text, origin_label, label`).
2. It prints a summary and **exits** — fix the `label` column to `0` or `1`
   (edit the CSV or the parquet in place), then re-run the same command. The
   fixes are applied on the next run; `-1` rows never reach training.

**Alternatives:**

- `--hf-fallback-regex` (or `HF_FALLBACK_REGEX=1`): classify the `-1` rows
  with regex (`attack` → 1, `emotional`/`clean` → 0, regex-grey dropped) and
  continue.
- `--hf-no-exit-on-unsure` (or `HF_EXIT_ON_UNSURE_OFF=1`): keep going even if
  `-1` rows remain — they are then excluded by the `merge-ftan` step.

### Output

`data/hf/raw/candidates_*.parquet` in the candidate schema. With FTAN, each
row carries `label` (`0`/`1`/`-1`), `origin_label` like `ftan_0.982`, and
`source` set from `--source-name`.

---

## Wiring it together

The candidates from any/all fetchers are turned into `data/raw/community.parquet`
by `make_dataset.py` (`curate` → `merge` for banks, or `merge-ftan` for
FTAN-labeled shards), then subsampled with `--source_caps` and fed into the
core pipeline. See [the orchestrator page](04-orchestrator.md) for the full
flag reference.