from datasets import DatasetDict, load_dataset
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from more_itertools import batched
import pandas as pd
import argparse
import pickle
import torch
import os
import string
import re

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)

os.makedirs("result", exist_ok=True)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

model_params = {
    "openai/gpt-oss-120b": {"temperature": 1.0, "stop_token_ids": encoding.stop_tokens_for_assistant_actions()},
    "openai/gpt-oss-20b": {"temperature": 1.0, "stop_token_ids": encoding.stop_tokens_for_assistant_actions()},
    "Qwen/Qwen3-30B-A3B-Thinking-2507": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    "Qwen/Qwen3-4B-Thinking-2507": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}

DEFAULT_PROMPT_TEMPLATE = """{question}

Solve the given question and output the final answer in the following format: \\boxed{{N}}."""


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

def safe_parse_oss(output, convo):
    result = []
    for out in output.outputs:
        try:
            entries = encoding.parse_messages_from_completion_tokens(out.token_ids, Role.ASSISTANT)
            result.append(convo.messages + entries)
            # result.append(entries)
        except Exception as e:
            print(e)
            result.append(None)

    return result

def safe_parse_original(output):
    result = []
    for out in output.outputs:
        try:
            result.append(out.text)
        except:
            result.append(None)

    return result

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_path", type=str, default="ethz-spylab/RealMath")
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help=(
            "Template used to format each prompt. Columns can be referenced by "
            "name (e.g. '### Question {col_a}\\n### Solution {col_b}')."
        ),
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of prompts to generate in a single batch.",
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
    if subset:
        dataset = load_dataset(path, subset, split='test')
    else:
        dataset = load_dataset(path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[next(iter(dataset))]
    return dataset.to_pandas()

def format_prompts_general(df, tokenizer, template):
    prompts = []
    sample_prompt = None
    for _, row in df.iterrows():
        prompt_text = build_prompt_from_row(row, template)
        if sample_prompt is None:
            sample_prompt = prompt_text
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts, sample_prompt

def format_prompts_oss(df, template):
    prompts, convos = [], []
    sample_prompt = None
    for _, row in df.iterrows():
        prompt_text = build_prompt_from_row(row, template)
        if sample_prompt is None:
            sample_prompt = prompt_text

        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(Role.USER, prompt_text),
            ]
        )

        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prompts.append(encoding.decode(prefill_ids))
        convos.append(convo)

    return prompts, convos, sample_prompt

def main():
    args = args_parse()

    result_root = args.output_root
    os.makedirs(result_root, exist_ok=True)

    result_dir = os.path.join(result_root, args.model_name.split("/")[-1])
    tag = sanitize_tag(args.output_tag)
    if tag:
        result_dir = os.path.join(result_dir, tag)
    os.makedirs(result_dir, exist_ok=True)

    model = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = model.get_tokenizer()
    config = AutoConfig.from_pretrained(args.model_name)
    max_model_len = config.max_position_embeddings

    params_dict = model_params[args.model_name]
    params = SamplingParams(n=args.n, max_tokens=16384, **params_dict)

    df = load_dataset_to_df(args.dataset_path, subset=args.dataset_subset)
    template_fields = extract_template_fields(args.prompt_template)
    missing_columns = [
        field for field in template_fields if field not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"prompt template references missing columns: {missing_columns}"
        )

    use_oss = "gpt-oss" in args.model_name
    if use_oss:
        prompts, convos, sample_prompt = format_prompts_oss(
            df, args.prompt_template
        )
    else:
        prompts, sample_prompt = format_prompts_general(
            df, tokenizer, args.prompt_template
        )

    if sample_prompt:
        print("Sample formatted prompt (row 0):")
        print(sample_prompt)

    for batch_index, batch in enumerate(batched(prompts, args.batch_size)):
        start = batch_index * args.batch_size
        end = start + len(batch)
        dfs = df.iloc[start:end]
        dfs.reset_index(inplace=True, drop=True)

        outputs = model.generate(batch, params)
        if use_oss:
            offset = start
            responses = [
                safe_parse_oss(output, convos[offset + ids])
                for ids, output in enumerate(outputs)
            ]
        else:
            responses = [safe_parse_original(output) for output in outputs]
        dfs[args.model_name] = responses
        file_name = f"result_{batch_index}.jsonl"
        dfs.to_json(os.path.join(result_dir, file_name), lines=True, orient="records")

if __name__ == "__main__":
    main()
