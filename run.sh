#!/bin/bash

export HF_TOKEN=""

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"
# PROMPT_TEMPLATE=$'Solve the given question and output the final answer in the following format: \\boxed{{N}}. \n\n{question}'
N=32
BATCH_SIZE=25

# MODELS=(
#   "openai/gpt-oss-120b"
#   "openai/gpt-oss-20b"
#   "Qwen/Qwen3-30B-A3B-Thinking-2507"
#   "Qwen/Qwen3-4B-Thinking-2507"
# )

# OUTPUT_TAG="original_question"
# for MODEL in "${MODELS[@]}"; do
#   echo "Running ${MODEL}..."
#   python variants_solve.py \
#     --model_name "$MODEL" \
#     --dataset_path "$DATASET_PATH" \
#     --dataset_subset "$DATASET_SUBSET" \
#     --prompt_template "$PROMPT_TEMPLATE" \
#     --n "$N" \
#     --batch_size "$BATCH_SIZE" \
#     --output_tag "$OUTPUT_TAG"
# done

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"
PROMPT_TEMPLATE=$'Solve the given question and output the final answer in the following format: \\boxed{{N}}. \n\n{variant question}'
N=32

MODELS=(
  "openai/gpt-oss-120b"
  # "openai/gpt-oss-20b"
  "Qwen/Qwen3-30B-A3B-Thinking-2507"
  # "Qwen/Qwen3-4B-Thinking-2507"
)

OUTPUT_TAG="variant_question"
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


PROMPT_TEMPLATE=$'{question} \n\n {solution_text}.\n\nRefer to the question-solution set provided above. Solve the provided question below and output the final answer in the following format: \\boxed{{N}}. \n\n {variant question}'
N=128
MODELS=(
  "openai/gpt-oss-120b"
  "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

OUTPUT_TAG="variant_with_reference"
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
