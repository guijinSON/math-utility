#!/bin/bash

# Example runner for variants_solve_litellm.py using LiteLLM async completions.
# Customize the variables below as needed.

set -euo pipefail

# API configuration
# Export your API key before running, e.g.:
#   export OPENAI_API_KEY="sk-..."
API_KEY_ENV="OPENAI_API_KEY"
API_BASE="${API_BASE:-}"  # optional, e.g. https://api.openai.com/v1

# Dataset configuration
DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="suhak_pivot"

# Prompt template and generation settings
PROMPT_TEMPLATE=$'{question}\n\nSolve the given question and output the final answer in the following format: \\boxed{{N}}.'
N=1
MAX_CONCURRENT=8
MAX_TOKENS=2048
TEMPERATURE=1.0
TOP_P=0.9

# Models to run (LiteLLM-compatible names)
MODELS=(
  "openai/gpt-oss-120b"
  # "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

OUTPUT_TAG="litellm_variants"

for MODEL in "${MODELS[@]}"; do
  echo "Running ${MODEL}..."
  python variants_solve_litellm.py \
    --model_name "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --dataset_subset "$DATASET_SUBSET" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --n "$N" \
    --max_concurrent "$MAX_CONCURRENT" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --api_base "$API_BASE" \
    --api_key_env "$API_KEY_ENV" \
    --output_tag "$OUTPUT_TAG"
done
