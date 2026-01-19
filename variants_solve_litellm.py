import argparse
import asyncio
import json
import os
import pickle
import re
import string
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import DatasetDict, load_dataset
from litellm import acompletion

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
)

# Ensure output directory exists ahead of time.
os.makedirs("result", exist_ok=True)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Default per-model parameters. Users can override via CLI flags.
model_params: Dict[str, Dict[str, Any]] = {
    "openai/gpt-oss-120b": {"temperature": 1.0},
    "openai/gpt-oss-20b": {"temperature": 1.0},
    "Qwen/Qwen3-235B-A22B-Thinking-2507": {"temperature": 0.7, "top_p": 0.8},
    "Qwen/Qwen3-30B-A3B-Thinking-2507": {"temperature": 0.7, "top_p": 0.8},
    "Qwen/Qwen3-4B-Thinking-2507": {"temperature": 0.7, "top_p": 0.8},
}

DEFAULT_PROMPT_TEMPLATE = """{question}

Solve the given question and output the final answer in the following format: \\boxed{{N}}."""


def extract_template_fields(template: str) -> set[str]:
    formatter = string.Formatter()
    return {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name
    }


def build_prompt_from_row(row: pd.Series, template: str) -> str:
    row_values: Dict[str, str] = {}
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


def sanitize_tag(tag: str) -> str:
    if not tag:
        return ""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return clean.strip("_")


def args_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
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
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions to request per prompt.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=8,
        help="Maximum concurrent API calls.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; overrides model defaults when provided.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p nucleus sampling; overrides model defaults when provided.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Stop sequences. Repeat flag to provide multiple sequences.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=os.environ.get("LITELLM_API_BASE"),
        help="Custom API base URL for litellm-compatible endpoints.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name holding the API key.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=60,
        help="Timeout (in seconds) for each completion request.",
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
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default=None,
        help="Optional path to append a JSONL log for each completion call.",
    )
    return parser.parse_args()


def load_dataset_to_df(path: str, subset: Optional[str] = None) -> pd.DataFrame:
    if not path:
        raise ValueError("`dataset_path` is required; pass --dataset_path with a non-empty path.")
    if subset:
        dataset = load_dataset(path, subset, split="test")
    else:
        dataset = load_dataset(path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[next(iter(dataset))]
    return dataset.to_pandas()


def format_prompts_general(df: pd.DataFrame, template: str) -> tuple[list[str], Optional[str]]:
    prompts: list[str] = []
    sample_prompt: Optional[str] = None
    for _, row in df.iterrows():
        prompt_text = build_prompt_from_row(row, template)
        if sample_prompt is None:
            sample_prompt = prompt_text
        prompts.append(prompt_text)
    return prompts, sample_prompt


def format_prompts_oss(df: pd.DataFrame, template: str) -> tuple[list[str], list[Conversation], Optional[str]]:
    prompts: list[str] = []
    convos: list[Conversation] = []
    sample_prompt: Optional[str] = None
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


def serialize_response(resp: Any) -> Dict[str, Any]:
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    if hasattr(resp, "dict"):
        try:
            return resp.dict()
        except Exception:
            pass
    if isinstance(resp, dict):
        return resp
    try:
        return resp.__dict__
    except Exception:
        return {"unserializable_response": str(resp)}


def extract_choice_texts(resp_dict: Dict[str, Any]) -> List[str]:
    choices = resp_dict.get("choices") or []
    texts: List[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if isinstance(message, dict):
            texts.append(message.get("content") or "")
            continue
        if "text" in choice:
            texts.append(choice.get("text") or "")
    return texts


def build_completion_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = model_params.get(args.model_name, {})
    api_key = os.environ.get(args.api_key_env)

    kwargs: Dict[str, Any] = {
        "model": args.model_name,
        # We manually repeat calls to achieve `n` completions per prompt.
        # Keeping n=1 here ensures provider compatibility.
        "n": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature if args.temperature is not None else defaults.get("temperature"),
        "top_p": args.top_p if args.top_p is not None else defaults.get("top_p"),
        "timeout": args.request_timeout,
    }
    if args.api_base:
        kwargs["api_base"] = args.api_base
    if api_key:
        kwargs["api_key"] = api_key
    if args.stop:
        kwargs["stop"] = args.stop

    # Remove unset parameters to avoid sending None to providers.
    return {k: v for k, v in kwargs.items() if v is not None}


async def append_jsonl(log_path: str, record: Dict[str, Any], lock: asyncio.Lock) -> None:
    """Append a JSON record to the log_path as a single line (thread-safe)."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    async with lock:
        await asyncio.to_thread(_append_line, log_path, line)


def _append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


async def request_completion(
    prompt_text: str,
    completion_kwargs: Dict[str, Any],
    use_prompt_param: bool,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    kwargs = dict(completion_kwargs)
    if use_prompt_param:
        kwargs["prompt"] = prompt_text
    else:
        kwargs["messages"] = [{"role": "user", "content": prompt_text}]

    async with semaphore:
        try:
            response = await acompletion(**kwargs)
            return serialize_response(response)
        except Exception as exc:  # Keep batch progress even if individual calls fail.
            return {"error": str(exc)}


async def generate_all(
    prompts: List[str],
    df: pd.DataFrame,
    args: argparse.Namespace,
    use_oss: bool,
    convos: Optional[List[Conversation]] = None,
) -> None:
    result_root = args.output_root
    os.makedirs(result_root, exist_ok=True)

    result_dir = os.path.join(result_root, args.model_name.split("/")[-1])
    tag = sanitize_tag(args.output_tag)
    if tag:
        result_dir = os.path.join(result_dir, tag)
    os.makedirs(result_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(args.max_concurrent)
    completion_kwargs = build_completion_kwargs(args)
    log_path = args.log_jsonl or os.path.join(result_dir, "responses.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_lock = asyncio.Lock()

    # Manually repeat calls per prompt to honor args.n (LiteLLM n may be unsupported).
    tasks = []
    for prompt_index, prompt_text in enumerate(prompts):
        for repetition_index in range(args.n):
            tasks.append(
                _request_log_wrapper(
                    prompt_text=prompt_text,
                    prompt_index=prompt_index,
                    repetition_index=repetition_index,
                    completion_kwargs=completion_kwargs,
                    use_oss=use_oss,
                    semaphore=semaphore,
                    log_path=log_path,
                    log_lock=log_lock,
                )
            )

    responses_flat = await asyncio.gather(*tasks)

    # Group back into per-prompt lists
    grouped_responses: List[List[Dict[str, Any]]] = []
    grouped_choices: List[List[List[str]]] = []
    idx = 0
    for _ in prompts:
        chunk = responses_flat[idx : idx + args.n]
        grouped_responses.append(chunk)
        grouped_choices.append([extract_choice_texts(resp) for resp in chunk])
        idx += args.n

    payload: Dict[str, Any] = {
        "model_name": args.model_name,
        "prompt_template": args.prompt_template,
        "use_oss": use_oss,
        "prompts": prompts,
        "dataset_rows": df.to_dict(orient="records"),
        "n": args.n,
        "choices": grouped_choices,
        "raw_responses": grouped_responses,
        "litellm_params": {k: v for k, v in completion_kwargs.items() if k != "api_key"},
        "log_jsonl": log_path,
    }

    if use_oss and convos is not None:
        payload["conversations"] = convos

    file_name = "all.pkl"
    with open(os.path.join(result_dir, file_name), "wb") as pickle_file:
        pickle.dump(payload, pickle_file)


async def _request_log_wrapper(
    prompt_text: str,
    prompt_index: int,
    repetition_index: int,
    completion_kwargs: Dict[str, Any],
    use_oss: bool,
    semaphore: asyncio.Semaphore,
    log_path: str,
    log_lock: asyncio.Lock,
) -> Dict[str, Any]:
    """Wrap completion request and log each instance as it returns."""
    response = await request_completion(prompt_text, completion_kwargs, use_oss, semaphore)
    record = {
        "prompt_index": prompt_index,
        "repetition_index": repetition_index,
        "prompt": prompt_text,
        "response": response,
    }
    await append_jsonl(log_path, record, log_lock)
    return response


def main() -> None:
    args = args_parse()

    df = load_dataset_to_df(args.dataset_path, subset=args.dataset_subset)
    template_fields = extract_template_fields(args.prompt_template)
    missing_columns = [field for field in template_fields if field not in df.columns]
    if missing_columns:
        raise ValueError(f"prompt template references missing columns: {missing_columns}")

    use_oss = "gpt-oss" in args.model_name

    if use_oss:
        prompts, convos, sample_prompt = format_prompts_oss(df, args.prompt_template)
    else:
        prompts, sample_prompt = format_prompts_general(df, args.prompt_template)
        convos = None

    if sample_prompt:
        print("Sample formatted prompt (row 0):")
        print(sample_prompt)

    asyncio.run(generate_all(prompts, df, args, use_oss, convos if use_oss else None))


if __name__ == "__main__":
    main()
