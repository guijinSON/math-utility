#!/bin/bash

# Run GenRM locally via vLLM. Ensure model weights are available (HF auth if
# needed) and GPUs are visible (e.g., CUDA_VISIBLE_DEVICES).

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"
QUESTION_COLUMN="question"
ANSWER_COLUMN="solution_text"
DUMMY_RESPONSE="N/A"

# GENERATION CONFIG
BATCH_SIZE=8
OUTPUT_TAG="genrm_eval"
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=16384

MODELS=(
  "nvidia/Qwen3-Nemotron-235B-A22B-GenRM"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for MODEL in "${MODELS[@]}"; do
  echo "Scoring with ${MODEL}..."
  python "${PROJECT_ROOT}/genrm_score.py" \
    --model_name "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --dataset_subset "$DATASET_SUBSET" \
    --question_column "$QUESTION_COLUMN" \
    --answer_column "$ANSWER_COLUMN" \
    --dummy_response "$DUMMY_RESPONSE" \
    --batch_size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --output_tag "$OUTPUT_TAG" \
    "$@"
done

