#!/bin/bash

export HF_TOKEN=""

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"

PROMPT_TEMPLATE=$'{question} \n\n {solution_text}.\n\nRefer to the question-solution set provided above. Solve the provided question below and output the final answer in the following format: \\boxed{{N}}. \n\n {variant question}'
N=64
BATCH_SIZE=25
MODELS=(
   "Qwen/Qwen3-235B-A22B-Thinking-2507"
  # "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

OUTPUT_TAG="suhak_variant_with_reference"
for MODEL in "${MODELS[@]}"; do
  echo "Running ${MODEL}..."
  python variants_solve.py \
    --model_name "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --dataset_subset "$DATASET_SUBSET" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --n "$N" \
    --batch_size "$BATCH_SIZE" \
    --output_tag "$OUTPUT_TAG"
done
