from datasets import DatasetDict, load_dataset
from transformers import AutoConfig
from vllm import LLM
from more_itertools import batched
import pandas as pd
import argparse
import pickle
import torch
import os
import string
import re


# Default system template mirrors the judging-style prompt. Users can override via CLI.
DEFAULT_SYSTEM_TEMPLATE = """You are an impartial mathematical judge.
You will be given a math problem (question) and a proposed solution (assistant response).
Carefully evaluate the solution for logical correctness, mathematical validity, completeness, and rigor.
Provide a reward score that reflects the overall quality of the solution."""


def extract_template_fields(template):
    formatter = string.Formatter()
    return {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name
    }


def build_prompt_from_row(row, template):
    row_values = {}
    for key, value in row.items():
        if pd.isna(value):
            row_values[key] = ""
        else:
            row_values[key] = str(value)

    try:
        return template.format(**row_values)
    except KeyError as exc:
        raise ValueError(
            f"Prompt template references column {exc.args[0]!r} which is missing from the dataset row."
        )


def sanitize_tag(tag):
    if not tag:
        return ""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return clean.strip("_")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/AceMath-72B-RM",
        help="Reward model to load (HF repo id).",
    )
    parser.add_argument("--dataset_path", type=str, default="ethz-spylab/RealMath")
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument(
        "--system_template",
        type=str,
        default=DEFAULT_SYSTEM_TEMPLATE,
        help="System prompt template. Can reference dataset columns (e.g., {question}).",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default="question",
        help="Dataset column name to use as the user query.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="solution_text",
        help="Dataset column name to use as the assistant response.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of conversations to score per batch.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="result",
        help="Base directory where result subfolders are created.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default="",
        help="Optional suffix applied to the result subdirectory for this run.",
    )
    return parser.parse_args()


def load_dataset_to_df(path, subset=None):
    if not path:
        raise ValueError("`dataset_path` is required; pass --dataset_path with a non-empty path.")
    if subset:
        dataset = load_dataset(path, subset, split="test")
    else:
        dataset = load_dataset(path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[next(iter(dataset))]
    return dataset.to_pandas()


def format_rm_prompts(df, tokenizer, system_template, question_col, answer_col):
    prompts = []
    sample_prompt = None
    for _, row in df.iterrows():
        system_text = build_prompt_from_row(row, system_template)
        question = "" if pd.isna(row.get(question_col)) else str(row.get(question_col, ""))
        answer = "" if pd.isna(row.get(answer_col)) else str(row.get(answer_col, ""))

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if sample_prompt is None:
            sample_prompt = prompt_text

        prompts.append(prompt_text)

    return prompts, sample_prompt


def main():
    args = args_parse()

    result_root = args.output_root
    os.makedirs(result_root, exist_ok=True)

    result_dir = os.path.join(result_root, args.model_name.split("/")[-1])
    tag = sanitize_tag(args.output_tag)
    if tag:
        result_dir = os.path.join(result_dir, tag)
    os.makedirs(result_dir, exist_ok=True)

    rope_theta = 1000000 
    original_max_position_embeddings = 4096
    factor = 2.0
    hf_overrides = { 
        "rope_parameters": { 
            "rope_theta": rope_theta, 
            "rope_type": "yarn", 
            "factor": factor, 
            "original_max_position_embeddings": original_max_position_embeddings, 
        }, 
        "max_model_len": int(original_max_position_embeddings * factor), 
    }
    model = LLM(
        args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        runner="pooling",
        trust_remote_code=True,           
        hf_overrides=hf_overrides
    )
    
    tokenizer = model.get_tokenizer()
    config = AutoConfig.from_pretrained(args.model_name)
    max_model_len = getattr(config, "max_position_embeddings", None)

    df = load_dataset_to_df(args.dataset_path, subset=args.dataset_subset)

    for required_col in [args.question_column, args.answer_column]:
        if required_col not in df.columns:
            raise ValueError(
                f"Required column {required_col!r} is missing from the dataset."
            )

    template_fields = extract_template_fields(args.system_template)
    missing_columns = [field for field in template_fields if field not in df.columns]
    if missing_columns:
        raise ValueError(
            f"System template references missing columns: {missing_columns}"
        )

    prompts, sample_prompt = format_rm_prompts(
        df,
        tokenizer,
        args.system_template,
        args.question_column,
        args.answer_column,
    )

    if sample_prompt:
        print("Sample formatted RM prompt (row 0):")
        print(sample_prompt)
        if max_model_len:
            print(f"Max model length: {max_model_len}")

    for batch_index, batch in enumerate(batched(prompts, args.batch_size)):
        start = batch_index * args.batch_size
        end = start + len(batch)
        batch_rows = df.iloc[start:end].reset_index(drop=True).to_dict(orient="records")

        results = model.reward(batch)  # batch can be a list[str]

        scores = []
        raw_outputs = []
        
        for out in results:
            data = getattr(out.outputs, "data", None)  # torch.Tensor or None
            if data is None:
                scores.append(None)
                raw_outputs.append(None)
                continue
        
            # Keep a JSON-friendly copy for logging
            try:
                raw_outputs.append(data.detach().cpu().tolist())
            except Exception:
                raw_outputs.append(repr(data))
        
            # Produce a single float "score" per prompt.
            # Most reward models give a scalar; if it’s a vector, take the last element.
            try:
                if hasattr(data, "numel") and data.numel() == 1:
                    scores.append(float(data.item()))
                else:
                    scores.append(float(data.reshape(-1)[-1].item()))
            except Exception:
                scores.append(None)

        batch_payload = {
            "model_name": args.model_name,
            "system_template": args.system_template,
            "question_column": args.question_column,
            "answer_column": args.answer_column,
            "batch_index": batch_index,
            "start_index": start,
            "end_index": end,
            "prompts": batch,
            "dataset_rows": batch_rows,
            "scores": scores,
            "raw_outputs": [r.outputs for r in results],
        }

        file_name = f"batch_{batch_index}.pkl"
        with open(os.path.join(result_dir, file_name), "wb") as pickle_file:
            pickle.dump(batch_payload, pickle_file)


if __name__ == "__main__":
    main()

