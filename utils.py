import pickle
from typing import Any


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

