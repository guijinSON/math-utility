import pickle
from typing import Any
import re
from typing import Optional, Dict

SUMMARY_RE = re.compile(
    r"(?ims)^\s*\**\s*Summary\s*\**\s*:\s*(.*?)\s*(?=^\s*\**\s*Detailed\s+Analysis\s*\**\s*:|^\s*\**\s*Score\s*\**\s*:|\Z)"
)

ANALYSIS_RE = re.compile(
    r"(?ims)^\s*\**\s*Detailed\s+Analysis\s*\**\s*:\s*(.*?)\s*(?=^\s*\**\s*Score\s*\**\s*:|\Z)"
)

SCORE_RE = re.compile(
    r"(?im)^\s*(?:\*\*|__)?\s*Score\s*(?:\*\*|__)?\s*[:\-]\s*(?:\*\*|__)?\s*([1-9]|10)\s*(?:\*\*|__)?\b"
)

def parse_evaluation(text: str) -> Dict[str, Optional[object]]:
    summary_match = SUMMARY_RE.search(text)
    analysis_match = ANALYSIS_RE.search(text)
    score_match = SCORE_RE.search(text)

    summary = summary_match.group(1).strip() if summary_match else None
    detailed_analysis = analysis_match.group(1).strip() if analysis_match else None
    score = int(score_match.group(1)) if score_match else None

    return {
        "summary": summary,
        "detailed_analysis": detailed_analysis,
        "score": score,
    }

class SamplingParamsProxy:
    """Lightweight stand-in for vllm.sampling_params.SamplingParams."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setstate__(self, state):
        # Many objects pickle as a dict of attributes.
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class PatchedUnpickler(pickle.Unpickler):
    """Unpickler that redirects SamplingParams to SamplingParamsProxy."""

    def find_class(self, module, name):
        if module == "vllm.sampling_params" and name == "SamplingParams":
            return SamplingParamsProxy
        return super().find_class(module, name)


def fix_best_of(obj: Any) -> None:
    """Recursively coerce SamplingParamsProxy.best_of floats to ints."""
    if isinstance(obj, SamplingParamsProxy):
        if hasattr(obj, "best_of") and isinstance(obj.best_of, float):
            obj.best_of = int(obj.best_of)
    elif isinstance(obj, dict):
        for value in obj.values():
            fix_best_of(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            fix_best_of(value)


def load_pickle(path: str) -> Any:
    """Load a pickle file without any patching."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_pickle_with_patch(path: str) -> Any:
    """Load a pickle file using PatchedUnpickler and normalize best_of."""
    with open(path, "rb") as handle:
        data = PatchedUnpickler(handle).load()
    fix_best_of(data)
    return data

