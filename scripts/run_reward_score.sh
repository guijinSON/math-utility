#!/bin/bash
set -euo pipefail

# Fill in if you need private weights
export HF_TOKEN=""

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"

# System template can reference dataset columns, e.g. {question} or {variant question}
SYSTEM_TEMPLATE=$'You are an impartial mathematical judge.\nYou will be given a math problem (question) and a proposed solution (assistant response).\nCarefully evaluate the solution for logical correctness, mathematical validity, completeness, and rigor.\nProvide a reward score that reflects the overall quality of the solution.'

QUESTION_COLUMN="question"
ANSWER_COLUMN="solution_text"

BATCH_SIZE=8
OUTPUT_TAG="rm_eval"

MODELS=(
  "nvidia/AceMath-72B-RM"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for MODEL in "${MODELS[@]}"; do
  echo "Scoring with ${MODEL}..."
  python "${PROJECT_ROOT}/reward_score.py" \
    --model_name "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --dataset_subset "$DATASET_SUBSET" \
    --system_template "$SYSTEM_TEMPLATE" \
    --question_column "$QUESTION_COLUMN" \
    --answer_column "$ANSWER_COLUMN" \
    --batch_size "$BATCH_SIZE" \
    --output_tag "$OUTPUT_TAG" \
    "$@"
done

