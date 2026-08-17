# Core Pipeline (scripts 01–06)

This page documents the six core pipeline scripts, in order. Each one reads
the previous script's output and writes its own. The chain is:

```
data/raw/*.parquet
  → 02_normalize → data/processed/normalized.parquet
  → 03_mutate    → data/processed/mutated.parquet
  → 04_dedup     → data/processed/deduped.parquet
  → 05_split     → data/processed/split.parquet
  → 06_export    → data/final/dataset  (DatasetDict)
```

> All scripts are run from the repo root with `.venv/bin/python`.
> Every script prints per-source / per-class stats as it goes.

---

## 01_download.py — download core sources

Downloads the five public source datasets and writes each to a parquet with
the unified schema `text, label, source, origin_label, split_origin`.

```bash
.venv/bin/python scripts/01_download.py
```

### Sources

| source id | HF dataset | role |
|---|---|---|
| `jigsaw` | `tcapelle/jigsaw-toxic-comment-classification-challenge` | offensive + clean (in-domain), train + `balanced_test` |
| `davidson` | `contemmcm/hate-speech-and-offensive-language` | offensive tweets + clean |
| `toxigen` | `toxigen/toxigen-data` (annotated) | subtle toxicity, human scores ≥ 3.5 → offensive |
| `toxigen_machine` | `toxigen/toxigen-data` (train) | machine-generated, `prompt_label` |
| `hatexplain` | `dataspoof/HateXplain` | held-out test + train |
| `wikipedia` | `wikimedia/wikipedia` (20231101.en, streaming) | neutral clean filler |

Jigsaw rows are offensive if any of the six labels (`toxic`, `severe_toxic`,
`obscene`, `threat`, `insult`, `identity_hate`) is set; `origin_label` records
which ones (e.g. `obscene+insult`).

### Arguments

| flag | default | meaning |
|---|---|---|
| `--sources` | `jigsaw davidson toxigen hatexplain wikipedia` | which sources to download (space-separated) |
| `--wiki_cap` | `350000` | max sentences to sample from Wikipedia |

### Examples

```bash
# only jigsaw + davidson
.venv/bin/python scripts/01_download.py --sources jigsaw davidson

# a smaller wikipedia filler for a quick run
.venv/bin/python scripts/01_download.py --wiki_cap 50000
```

### Output

`data/raw/{jigsaw,davidson,toxigen,toxigen_machine,hatexplain,wikipedia}.parquet`

---

## 02_normalize.py — merge, dedupe, filter

Concatenates every `data/raw/*.parquet`, removes exact duplicates on
lowercased text (keep first), and drops rows outside the length bounds.

```bash
.venv/bin/python scripts/02_normalize.py
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--min_len` | `3` | minimum token count (by whitespace split) |
| `--max_len` | `512` | maximum token count |

### Output

`data/processed/normalized.parquet` + per-source × per-class stats on stdout.

---

## 03_mutate.py — the obfuscation engine

Generates obfuscated variants of every row (see
[the mutation table in the overview](00-overview.md#the-obfuscation-engine-commonpy)).

- **Offensive rows**: the original is kept (`mutated=0, variant=0`) plus
  `--offensive_variants` mutated copies with curse words replaced
  (`leet / separators / censor / repeat / case / homoglyph / fullwidth / combo`).
  If a row contains no seed word, a random-word or sentence-style mutation is
  used instead so variants stay diverse.
- **Clean rows**: the original is kept; a fraction (`--clean_variant_frac`) of
  rows get one innocuous-word mutation (e.g. `example` → `ex4mp1e`), so
  obfuscation is not a cue for offense.

```bash
.venv/bin/python scripts/03_mutate.py --offensive_variants 3 --clean_variant_frac 0.4
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--offensive_variants` | `3` | obfuscation copies per offensive row (`0` = no mutation) |
| `--clean_variant_frac` | `0.4` | fraction of clean rows that get a mutated variant |
| `--max_pos_per_source` | `0` | per-source cap on offensive rows (`0` = uncapped) |
| `--max_neg_per_source` | `0` | per-source cap on clean rows (`0` = uncapped) |
| `--seed` | `42` | RNG seed (deterministic mutations) |

`--max_pos_per_source` / `--max_neg_per_source` cap how many offensive / clean
rows survive **per source** before mutation — useful to keep one aggressive
source from dominating the build.

### Output

`data/processed/mutated.parquet` — adds `mutated` (0/1) and `variant` columns.

---

## 04_dedup.py — orthographic clustering

Assigns a `cluster` id to near-duplicate rows so splits never separate a
sentence from its own obfuscated variants.

Each text is **canonicalized**: homoglyphs → latin, leet digits → letters,
fullwidth → ASCII, separators/punctuation stripped, repeated chars collapsed
(`fuuuuck` → `fuck`). Rows sharing a canonical form share a cluster.

Exact text duplicates (same lowercased text) are dropped, keeping the first.
Mutation variants are *kept* — they are valuable training data — just grouped.

```bash
.venv/bin/python scripts/04_dedup.py
```

No arguments.

### Output

`data/processed/deduped.parquet` — adds the `cluster` column, and prints
cluster statistics (`multi-row clusters: N`).

---

## 05_split.py — cluster-stratified splits

Splits at the **cluster** level into `train` / `validation` / `test`:

1. **test** — a balanced sample (`test_target // 2` each class) drawn first,
   from held-out sources (HateXplain + any row whose source split was `test`).
2. **validation** — a balanced sample (`val_target // 2` each class) from the
   remainder.
3. **train** — everything left.

```bash
.venv/bin/python scripts/05_split.py --test_target 30000 --val_target 20000
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--test_target` | `30000` | target test rows (balanced, roughly half per class) |
| `--val_target` | `20000` | target validation rows (balanced) |
| `--seed` | `42` | RNG seed |

### Output

`data/processed/split.parquet` — adds a `split` column (`train` / `validation`
/ `test`).

---

## 06_export.py — build the DatasetDict

Reads the split parquet and builds a `datasets.DatasetDict` with typed
features, saves it to `data/final/dataset`, writes `stats.json` and
`dataset_card.md`, and optionally pushes to the Hub.

Also creates **`test_obfuscated`**: the mutated (`mutated == 1`) rows of the
test set, held out to evaluate obfuscation robustness.

```bash
.venv/bin/python scripts/06_export.py
.venv/bin/python scripts/06_export.py --eval_obfuscation --hub_id youruser/ftan-2.0 --private
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--eval_obfuscation` / `--no-eval_obfuscation` | `on` | add the `test_obfuscated` split |
| `--hub_id` | `None` | push the dataset to this Hub repo id (e.g. `user/name`) |
| `--private` | off | create the Hub repo as private |

### Output

- `data/final/dataset/` — the `DatasetDict` (gitignored)
- `data/final/stats.json` — `splits`, `total_rows`, `label_distribution`,
  `source_distribution`, `mutated_share`
- `data/final/dataset_card.md` — auto-generated dataset card

> **Hub caveat:** the script deletes stray `cache-*.arrow` files from the
> dataset directory before pushing. Do the same if you upload the folder by
> hand — stray pandas index columns break the Hub's auto-conversion. See
> [publishing](06-publishing.md).

---

## Next steps

- Add community data (Reddit / 4chan / YouTube / any HF dataset) — see
  [community data](03-community-data.md).
- Or drive the whole chain (including the community steps) with
  [`make_dataset.py`](04-orchestrator.md).
- Or train on the result — see [training & inference](05-training.md).
