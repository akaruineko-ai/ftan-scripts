# Setup & Installation

## Requirements

- Python **3.10+** (tested with 3.13).
- A GPU is **recommended** for the curate / train / FTAN-label steps
  (`--device 0`), but every stage has a CPU fallback — it is just slower.
- Disk: the full build needs several GB (`data/raw/` sources are ~1–2 GB,
  processed intermediates another few GB).
- Network access to Hugging Face (for source datasets, and optionally the Hub
  upload).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## requirements.txt notes

The pinned constraints matter:

- **`datasets>=2.21,<3`** — several sources (`toxigen`, `wikipedia`) ship
  loader scripts that were dropped in `datasets` v3. Do **not** upgrade past
  v2.
- **`transformers>=5.0`** — the training scripts use v5 APIs
  (`processing_class`, `warmup_steps`). Older versions will fail.
- **`torch>=2.0`** — install the CUDA build matching your driver if you want
  GPU speed:

  ```bash
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
  ```

- **`zstandard>=0.22`** — only needed for Reddit `.zst` dump parsing
  (`07_reddit_fetch.py --dump`).

## Project structure

```
ftan-2.0/
├── README.md              ← landing page
├── docs/                  ← this documentation
│   ├── 00-overview.md
│   ├── 01-setup.md
│   ├── 02-core-pipeline.md
│   ├── 03-community-data.md
│   ├── 04-orchestrator.md
│   ├── 05-training.md
│   ├── 06-publishing.md
│   └── 07-reference.md
├── requirements.txt
├── scripts/
│   ├── 01_download.py     # core sources → data/raw
│   ├── 02_normalize.py    # merge + dedupe + filter
│   ├── 03_mutate.py       # obfuscation engine
│   ├── 04_dedup.py        # orthographic clusters
│   ├── 05_split.py        # cluster-stratified splits
│   ├── 06_export.py       # DatasetDict + Hub push
│   ├── 07_reddit_fetch.py # reddit → candidates
│   ├── 08_reddit_curate.py# candidates → banks
│   ├── 09_4chan_fetch.py  # 4chan → candidates
│   ├── 10_youtube_fetch.py# youtube → candidates
│   ├── 11_hf_fetch.py     # any HF dataset → candidates
│   ├── make_dataset.py    # one-shot orchestrator
│   ├── train.py           # fine-tune classifier
│   ├── finalize.py        # pick best checkpoint, export
│   ├── predict.py         # CLI inference
│   ├── make_balanced.py   # rebalance train split
│   ├── run_4chan_ftan.sh  # 4chan FTAN-label wrapper
│   ├── autopilot-4chan.py # quick regex auto-label helper
│   └── common.py, candidate_common.py, reddit_vocab.py  # shared modules
└── data/
    ├── raw/               # core sources (generated)
    ├── processed/         # intermediate parquets (generated)
    ├── final/             # dataset + model + stats (generated)
    ├── reddit/ 4chan/ youtube/ hf/   # community candidates
    └── seeds/             # offensive seed-word lists
```

## Verify the install

```bash
.venv/bin/python -c "import datasets, transformers, pandas, numpy, requests; print('ok')"
```

## Two ways to run

1. **Script-by-script** — best for learning and for re-running a single stage.
   Each script is documented with its inputs, outputs and every flag on its
   own page ([core pipeline](02-core-pipeline.md),
   [community data](03-community-data.md)).
2. **`make_dataset.py`** — the orchestrator runs a plan of steps in dependency
   order, streaming output, and prints a dataset summary at the end
   ([orchestrator](04-orchestrator.md)).
