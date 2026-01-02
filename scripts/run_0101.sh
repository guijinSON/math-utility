#!/bin/bash

export HF_TOKEN=""

DATASET_PATH="amphora/suhak-variants"
DATASET_SUBSET="real_math_new_pivot"



PROMPT_TEMPLATE=$'{question} \n\n {solution_text}.\n\nRefer to the question-solution set provided above. Solve the provided question below and output the final answer in the following format: \\boxed{{N}}. \n\n {variant question}'
N=64
BATCH_SIZE=25
MODELS=(
     "Qwen/Qwen3-235B-A22B-Thinking-2507"
      "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

OUTPUT_TAG="real_math_new_pivot_variant_with_reference"
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



PROMPT_TEMPLATE=$"""You are an impartial mathematical judge.
You will be given a math problem and a proposed solution.
The solution may or may not be correct and does not explicitly state a final answer.

Your task is to carefully evaluate the solution for logical correctness, mathematical validity, completeness, and rigor, with special emphasis on whether the reasoning fully and correctly solves the given problem.

You must independently reason through the problem first, forming your own reference solution or partial verification, and then compare the given solution against that reasoning.

Evaluation Instructions:
1. Assess the solution step by step.
2. Verify all mathematical claims, derivations, and logical transitions.
3. Identify any gaps, unjustified steps, incorrect assumptions, or missing arguments.
4. Consider whether the solution fully resolves the question as stated.
5. Partial or high level arguments are insufficient unless explicitly justified.

Scoring Rubric:
Assign a single integer score from 1 to 10.

10: Completely correct, rigorous, logically sound, and fully solves the problem.
9: Essentially correct with very minor omissions that do not affect correctness.
7–8: Mostly correct but with minor logical gaps or unclear justifications.
5–6: Partially correct but missing key arguments or containing nontrivial ambiguity.
3–4: Significant errors or omissions, but some relevant ideas are present.
1–2: Largely incorrect with major logical or mathematical flaws.
1: Completely incorrect or irrelevant.

Required Output Format:
Summary:
<brief neutral summary of your evaluation>

Detailed Analysis:
<concise but precise discussion of correctness, gaps, or errors>

Score: <integer from 1 to 10>

---

Here is the question and solution:

Question:
{question}

Solution:
{solution_text}"""

N=64
BATCH_SIZE=25
MODELS=(
   "Qwen/Qwen3-235B-A22B-Thinking-2507"
  "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

OUTPUT_TAG="real_math_new_pivot_llm_judge_pointwise"
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


