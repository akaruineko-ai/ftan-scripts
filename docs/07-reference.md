# Reference & FAQ

Reference material for the shared modules and helper scripts, plus common
questions.

---

## Shared modules

### `scripts/common.py`

Path constants and the pipeline's shared helpers:

| symbol | purpose |
|---|---|
| `ROOT` / `DATA_DIR` / `RAW_DIR` / `PROCESSED_DIR` / `FINAL_DIR` / `SEEDS_DIR` | canonical paths used by every script |
| `UNIFIED_COLUMNS` | `text, label, source, origin_label, split_origin` |
| `clean_text(s)` | collapse whitespace, strip NUL bytes |
| `load_seeds()` | `(offensive_seeds, clean_targets)` from `data/seeds/ldnoobw_en.txt` + `extra_en.txt` |
| `Mutator` | the deterministic obfuscation engine (see below) |

**The seed-word list** lives in `data/seeds/`:
- `ldnoobw_en.txt` — the offensive word list (curse words, slurs, insults).
- `extra_en.txt` — your own additions.

Lines shorter than 3 chars or starting with `#` are ignored; everything is
lowercased.

**`Mutator`** applies eight word-level styles (`leet`, `sep`, `censor`,
`repeat`, `case`, `homoglyph`, `fullwidth`, `combo`) plus two sentence-level
fallbacks (`mutate_random_word`, `mutate_sentence_style`). It is seeded and
deterministic — same seed + input = same variants.

### `scripts/candidate_common.py`

Helpers shared by the community fetchers:

| symbol | purpose |
|---|---|
| `prefilter_keep` (re-export from `reddit_vocab`) | fetch-time keep decision |
| `_is_latin(text, min_ratio=0.5)` | skip mostly non-Latin comments |
| `Sink` | collects rows, dedupes on an md5 digest, flushes to `candidates_*.parquet` shards |
| `Sink.load_existing()` | rebuilds the dedup set + shard counter from on-disk shards (resume) |
| `clean_4chan_html(com)` | converts a 4chan post `com` field (escaped HTML) to plain text |

### `scripts/reddit_vocab.py`

The regex vocabulary + sentence classifier at the heart of the community
pipeline. It is **deliberately narrow**: it only detects *explicit* English
profanity and direct insults — not passive aggression or sarcasm.

Every comment is assigned one of four buckets:

| bucket | example | meaning |
|---|---|---|
| `attack` | `fuck you`, `kys`, `shut up` | profanity/insult aimed at a person → label 1 |
| `emotional` | `fuck this weather`, `this is fucking awesome` | expletive with no addressee → label 0 |
| `grey` | `what an idiot`, `he's a moron` | person insult, no 2nd-person pronoun → FTAN decides |
| `clean` | — | no profanity, no insult → label 0 |

Key functions:

| symbol | purpose |
|---|---|
| `classify_regex(text)` | bucket a single comment → `{profanity, insult, attack, category}` |
| `prefilter_keep(text, clean_sample_frac, rng)` | fetch-time decision: keep all non-clean rows, sample clean rows |
| `ATTACK_RE`, `SECOND_PERSON_RE`, `CENSORED_RE` | compiled patterns used internally |

The profanity patterns tolerate light obfuscation (`f*ck`, `f4ck`,
`f u c k`, `f**k`, `sh*t`) so the fetch stage still catches censored curse
words. Heavier mutations (homoglyphs, leet digits, fullwidth) are handled by
the FTAN model later.

How a comment is classified:
1. **Attack pattern** matches (`fuck you`, `kys`, `shut the fuck up`, …) → `attack`.
2. Otherwise, profanity/insult within **6 tokens** of a 2nd-person pronoun
   (`you`, `your`, `u`, `ur`, `ya`, …) → `attack`.
3. Otherwise: profanity alone → `emotional`; insult alone → `grey`; neither → `clean`.

> **Editing note:** if you add words to the vocab, keep the *expletive* vs
> *person-insult* distinction — that split is what teaches the model
> "swearing ≠ insulting".

---

## Helper scripts

### `scripts/autopilot-4chan.py`

A quick heuristic auto-labeler for the manual-review CSV. Labels rows with a
profanity word **and** a 2nd-person pronoun (`you`/`your`/`u`/`ya`/`yall`) as
offensive (`1`), everything else clean (`0`):

```bash
.venv/bin/python scripts/autopilot-4chan.py
```

Reads `data/4chan/raw/manual_check.csv`, writes `data/4chan/labeled_auto.csv`.
Useful for a first pass before a human review — the regex is a rough stand-in,
not a substitute for FTAN.

### `scripts/run_4chan_ftan.sh`

Wrapper around `make_dataset.py` for the "4chan via HF + FTAN labels"
workflow. See [the orchestrator page](04-orchestrator.md#run_4chan_ftansh--the-4chan--ftan-wrapper).

---

## FAQ

### Which steps need a GPU?

`curate` (FTAN on the grey zone), `fetch-hf` with `--ftan-model`, `train` and
`finalize`. Everything else is pure CPU pandas / numpy. Pass `--device 0` to
use CUDA where a model is involved.

### How do I quickly rebuild just the core dataset?

```bash
.venv/bin/python scripts/make_dataset.py --steps core
```

`download` is skipped automatically if `data/raw/` already has parquets. Force
a re-download with `--re-download`.

### Why is `test_obfuscated` separate from `test`?

`test` is the clean, balanced held-out evaluation. `test_obfuscated` holds the
**mutated** variants of test rows — the model never sees them in training, so
its score there measures real-world obfuscation tolerance. A model that trains
only on plain text often collapses on this split.

### Why does the run stop and write `manual_check.csv`?

FTAN leaves rows it is unsure about (p_off inside `[grey_low, grey_high]`)
labeled `-1`. `make_dataset.py` won't silently train on them — it writes them
for review and exits. Fix the `label` column to `0`/`1` and re-run; or use
`--hf-fallback-regex` to classify by regex, or `--hf-no-exit-on-unsure` to
have `merge-ftan` just drop them.

### My run seems to redo FTAN from scratch. Why?

The resume checkpoint is keyed to the dataset / seed / threshold / grey bounds
/ max_length. Change any of those and it deliberately starts fresh (with a
warning). Use `--no-resume` to force a clean start.

### How do I get more of one community source in the dataset?

Raise that fetcher's caps (`--max_rows`, `--yt-max-videos`, …) and cap the
others at the `subsample` step:

```bash
.venv/bin/python scripts/make_dataset.py \
    --api --4chan --youtube --yt-api-key AIza... \
    --source_caps "4chan=800000,youtube=500000,reddit=500000" --device 0
```

Caps are stratified by label, so each source keeps its offensive/clean ratio.

### `datasets` / `transformers` version errors?

- `datasets` must stay `<3` (the toxigen/wikipedia loader scripts were dropped
  in v3).
- `transformers` must be `>=5` (v5 Trainer APIs).
- `torch`: install a CUDA build matching your driver for GPU speed.

### Where does everything get saved?

| artifact | path |
|---|---|
| core sources | `data/raw/*.parquet` |
| processed chain | `data/processed/{normalized,mutated,deduped,split}.parquet` |
| final dataset | `data/final/dataset` |
| stats + card | `data/final/stats.json`, `data/final/dataset_card.md` |
| model | `data/final/model/` |
| community candidates | `data/{reddit,4chan,youtube,hf}/raw/candidates_*.parquet` |
| banks | `data/reddit/banks/bank_{a,b,c}.parquet` |
| seeds | `data/seeds/` |

All generated outputs under `data/` are gitignored.

### Is the dataset safe to publish?

It **contains raw offensive language**. Do not publish it in contexts that
require a safe-for-work dataset. The license section is
[here](06-publishing.md#license--disclaimer).