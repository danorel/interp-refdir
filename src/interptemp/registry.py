"""Resolve short names ("nnterp", "llm") or import paths ("pkg.mod:Class") to classes."""

from __future__ import annotations

import importlib
from typing import Any

# Lazily imported so optional deps (openai, nnterp) load only when used.
_BUILTINS: dict[str, dict[str, str]] = {
    "model": {
        "nnterp": "interptemp.models.nnterp_model:NnterpModel",
        "vllm": "interptemp.models.vllm_generator:VLLMGenerator",
    },
    "judge": {
        "llm": "interptemp.judges.llm:LLMJudge",
        "substring": "interptemp.judges.substring:SubstringJudge",
    },
    "task": {
        "jsonl": "interptemp.tasks.base:JsonlTask",
        "hf": "interptemp.tasks.base:HFDatasetTask",
    },
}


def import_object(path: str) -> Any:
    module, sep, attr = path.partition(":")
    if not sep:
        raise ValueError(f"Expected 'module:attr', got {path!r}")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def register(kind: str, name: str, path: str) -> None:
    _BUILTINS.setdefault(kind, {})[name] = path


def resolve(kind: str, name_or_path: str) -> Any:
    if ":" in name_or_path:
        return import_object(name_or_path)
    try:
        return import_object(_BUILTINS[kind][name_or_path])
    except KeyError:
        known = sorted(_BUILTINS.get(kind, {}))
        raise KeyError(
            f"Unknown {kind} {name_or_path!r}. Known: {known}, or pass 'module:Class'"
        ) from None
