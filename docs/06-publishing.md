# Publishing to the Hub

Two robust ways to upload `data/final/dataset` (the on-disk `DatasetDict`) to
the Hugging Face Hub.

## 1. From the pipeline

`06_export.py` builds the dataset, cleans stray cache files, and pushes:

```bash
.venv/bin/python scripts/06_export.py --hub_id youruser/ftan-2.0 --private
```

Or through the orchestrator:

```bash
.venv/bin/python scripts/make_dataset.py --steps export --hub_id youruser/ftan-2.0 --private
```

This path writes fresh parquet and removes `cache-*.arrow` files before
pushing, so the Hub's auto-conversion never trips on stray pandas index
columns.

## 2. Upload the on-disk folder directly

If you already have a built dataset:

```bash
huggingface_hub upload-folder data/final/dataset --repo-type=dataset youruser/ftan-2.0
```

> **Do not** upload `data/processed/*.parquet` or any `cache-*.arrow` files.
> Their stray pandas index columns break the Hub's auto-conversion
> (`DatasetGenerationError` / `CastError` on an `indices` column).

## What gets written to `data/final/`

| file | contents |
|---|---|
| `dataset/` | the `DatasetDict` (train / validation / test / test_obfuscated) |
| `stats.json` | `splits`, `total_rows`, `label_distribution`, `source_distribution`, `mutated_share` |
| `dataset_card.md` | auto-generated card with schema, splits, sources and design notes |

## Publishing the model

`train.py` (or `finalize.py`) can push the trained model straight to the Hub:

```bash
.venv/bin/python scripts/train.py --push_to_hub youruser/ftan-distilbert --private
.venv/bin/python scripts/finalize.py   # then push manually if needed
```

## License / disclaimer

Sources have varied licenses (CC0 for Jigsaw train; research terms for
Davidson / HateXplain; CC-BY-SA for ToxiGen). The dataset **contains raw
offensive language** and is intended for moderation research only.