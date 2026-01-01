import argparse
import os
import pickle
from typing import Any, Dict, List

import pandas as pd
from datasets import DatasetDict, load_dataset
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_dataset_to_df(path: str, subset: str | None = None) -> pd.DataFrame:
    if not path:
        raise ValueError("`dataset_path` is required; pass --dataset_path with a non-empty path.")
    if subset:
        dataset = load_dataset(path, subset, split="test")
    else:
        dataset = load_dataset(path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[next(iter(dataset))]
    return dataset.to_pandas()


def sanitize_tag(tag: str) -> str:
    import re

    if not tag:
        return ""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return clean.strip("_")


def args_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
        help="GenRM model id.",
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument(
        "--question_column",
        type=str,
        default="question",
        help="Column to send as the user message.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="solution_text",
        help="Column to send as response_1.",
    )
    parser.add_argument(
        "--dummy_response",
        type=str,
        default="N/A",
        help="Placeholder content for response_2.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_root", type=str, default="result")
    parser.add_argument("--output_tag", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16384)
    return parser.parse_args()


def row_to_messages(row: Dict[str, Any], user_col: str, answer_col: str, dummy: str) -> List[Dict[str, str]]:
    user_text = "" if pd.isna(row.get(user_col)) else str(row.get(user_col, ""))
    answer_text = "" if pd.isna(row.get(answer_col)) else str(row.get(answer_col, ""))
    return [
        {"role": "user", "content": user_text},
        {"role": "response_1", "content": answer_text},
        {"role": "response_2", "content": dummy},
    ]


def completion_text(content: str) -> str:
    # GenRM responses may include a <think> section. Strip it if present.
    return content.split("</think>")[-1].strip()


def main() -> None:
    args = args_parse()

    result_root = args.output_root
    os.makedirs(result_root, exist_ok=True)

    result_dir = os.path.join(result_root, args.model_name.split("/")[-1])
    tag = sanitize_tag(args.output_tag)
    if tag:
        result_dir = os.path.join(result_dir, tag)
    os.makedirs(result_dir, exist_ok=True)

    # Initialize vLLM model and tokenizer
    model = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    df = load_dataset_to_df(args.dataset_path, subset=args.dataset_subset)
    for required_col in [args.question_column, args.answer_column]:
        if required_col not in df.columns:
            raise ValueError(f"Required column {required_col!r} is missing from the dataset.")

    rows = df.to_dict(orient="records")

    for batch_index in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_index : batch_index + args.batch_size]
        messages_batch = [
            row_to_messages(row, args.question_column, args.answer_column, args.dummy_response)
            for row in batch_rows
        ]

        prompts_batch = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]

        generation_outputs = model.generate(prompts_batch, sampling_params)

        outputs = []
        raw_outputs = []
        for gen_out in generation_outputs:
            text = ""
            if gen_out.outputs:
                text = gen_out.outputs[0].text or ""
            outputs.append(completion_text(text))
            raw_outputs.append(gen_out)

        batch_payload = {
            "model_name": args.model_name,
            "question_column": args.question_column,
            "answer_column": args.answer_column,
            "dummy_response": args.dummy_response,
            "batch_index": batch_index // args.batch_size,
            "start_index": batch_index,
            "end_index": batch_index + len(batch_rows),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "messages": messages_batch,
            "dataset_rows": batch_rows,
            "outputs": outputs,
            "raw_outputs": raw_outputs,
        }

        file_name = f"batch_{batch_index // args.batch_size}.pkl"
        with open(os.path.join(result_dir, file_name), "wb") as handle:
            pickle.dump(batch_payload, handle)


if __name__ == "__main__":
    main()

