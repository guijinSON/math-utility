#!/bin/bash

# Export your OpenAI-compatible API key in the environment before running.
# Example: export OPENAI_API_KEY="sk-..."

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"
QUESTION_COLUMN="question"
ANSWER_COLUMN="solution_text"
DUMMY_RESPONSE="N/A"

BATCH_SIZE=8
OUTPUT_TAG="genrm_eval"

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
    --output_tag "$OUTPUT_TAG" \
    "$@"
done

