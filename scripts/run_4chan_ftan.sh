#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_STEPS="fetch-hf,merge-ftan,subsample,normalize,mutate,dedup,split,export"
DEFAULT_HF_DATASET="fuzzy-g/4chan_pol_whole_ds"
DEFAULT_HF_SPLIT="train"
DEFAULT_HF_TEXT="text"
DEFAULT_HF_SOURCE="flag"
DEFAULT_HF_TIME="__index_level_0__"
DEFAULT_HF_SCORE="__index_level_0__"
DEFAULT_HF_ID="__index_level_0__"
DEFAULT_HF_SOURCE_NAME="4chan"
DEFAULT_FTAN_MODEL="data/final/model/model"
DEFAULT_FTAN_DEVICE="0"
DEFAULT_FTAN_THRESHOLD="0.6"
DEFAULT_FTAN_BATCH="1024"
DEFAULT_FTAN_MAX_LENGTH="64"
DEFAULT_FTAN_GREY_LOW="0.30"
DEFAULT_FTAN_GREY_HIGH="0.70"
DEFAULT_CHECKPOINT_EVERY="150000"
DEFAULT_MAX_ROWS="2000000"
DEFAULT_SOURCE_CAPS="4chan=800000"
DEFAULT_HF_OUT="data/4chan/raw"

STEPS="${1:-$DEFAULT_STEPS}"

CMD=( .venv/bin/python scripts/make_dataset.py
      --steps "${STEPS}"
      --hf-dataset "${HF_DATASET:-$DEFAULT_HF_DATASET}"
      --hf-split "${HF_SPLIT:-$DEFAULT_HF_SPLIT}"
      --hf-text-col "${HF_TEXT:-$DEFAULT_HF_TEXT}"
      --hf-source-col "${HF_SOURCE:-$DEFAULT_HF_SOURCE}"
      --hf-time-col "${HF_TIME:-$DEFAULT_HF_TIME}"
      --hf-score-col "${HF_SCORE:-$DEFAULT_HF_SCORE}"
      --hf-id-col "${HF_ID:-$DEFAULT_HF_ID}"
      --hf-source-name "${HF_SOURCE_NAME:-$DEFAULT_HF_SOURCE_NAME}"
      --hf-ftan-model "${FTAN_MODEL:-$DEFAULT_FTAN_MODEL}"
      --hf-ftan-device "${FTAN_DEVICE:-$DEFAULT_FTAN_DEVICE}"
      --hf-ftan-batch-size "${FTAN_BATCH:-$DEFAULT_FTAN_BATCH}"
      --hf-ftan-threshold "${FTAN_THRESHOLD:-$DEFAULT_FTAN_THRESHOLD}"
      --hf-ftan-grey-low "${FTAN_GREY_LOW:-$DEFAULT_FTAN_GREY_LOW}"
      --hf-ftan-grey-high "${FTAN_GREY_HIGH:-$DEFAULT_FTAN_GREY_HIGH}"
      --hf-ftan-max-length "${FTAN_MAX_LENGTH:-$DEFAULT_FTAN_MAX_LENGTH}"
      --hf-checkpoint-every "${CHECKPOINT_EVERY:-$DEFAULT_CHECKPOINT_EVERY}"
      ${HF_RESUME_OFF:+--hf-no-resume}
      ${HF_EXIT_ON_UNSURE_OFF:+--hf-no-exit-on-unsure}
      ${HF_FALLBACK_REGEX:+--hf-fallback-regex}
      --max_rows "${MAX_ROWS:-$DEFAULT_MAX_ROWS}"
      --hf_out "${HF_OUT:-$DEFAULT_HF_OUT}"
      --source_caps "${SOURCE_CAPS:-$DEFAULT_SOURCE_CAPS}"
      --device "${FTAN_DEVICE:-$DEFAULT_FTAN_DEVICE}"
)

echo "=== 4chan_pol_whole_ds + FTAN pipeline ==="
echo "repo:  ${REPO_DIR}"
echo "steps: ${STEPS}"
echo "hf_dataset: ${HF_DATASET:-$DEFAULT_HF_DATASET}"
echo "ftan_model: ${FTAN_MODEL:-$DEFAULT_FTAN_MODEL}"
echo "grey zone: ${FTAN_GREY_LOW:-$DEFAULT_FTAN_GREY_LOW} .. ${FTAN_GREY_HIGH:-$DEFAULT_FTAN_GREY_HIGH}"
echo "source_caps: ${SOURCE_CAPS:-$DEFAULT_SOURCE_CAPS}"
echo "device: ${FTAN_DEVICE:-$DEFAULT_FTAN_DEVICE}"
echo
echo "hint: rows the model is unsure about (p_off in the grey zone) get"
echo "  label -1 and the run stops for review. Fix data/4chan/raw/manual_check.csv"
echo "  (label 0/1) and re-run to continue, or set HF_FALLBACK_REGEX=1 to"
echo "  classify them by regex instead."
echo

cd "${REPO_DIR}"
exec "${CMD[@]}"
