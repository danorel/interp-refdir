"""Typed experiment configs loaded from YAML, with dotted CLI overrides.

Method-specific knobs go in `params` (free-form) so new experiments don't require
touching this file. Promote a knob to a typed field only once several experiments share it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    name: str = "Qwen/Qwen3-8B"
    backend: str = "nnterp"  # registry name or "module:Class"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    revision: str | None = None
    # Prepend tokenizer BOS if the text doesn't already start with it (chat templates of
    # Llama/Gemma already include it; Qwen has none). Avoids silent double-BOS.
    add_bos: bool = True
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. enable_thinking
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(_Strict):
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    batch_size: int = 16
    skip_special_tokens: bool = True


class JudgeConfig(_Strict):
    backend: str = "llm"
    model: str = "google/gemini-3.8-flash"
    temperature: float = 0.0
    max_concurrency: int = 16
    reasoning_effort: str | None = "low"  # OpenRouter unified reasoning param; None = off
    max_tokens: int = 4096
    cache_path: str = ".cache/judge.sqlite"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(_Strict):
    name: str
    target: str  # "module:Class" of the Experiment subclass
    seed: int = 0
    output_dir: str = "outputs"
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    judge: JudgeConfig | None = None
    params: dict[str, Any] = Field(default_factory=dict)


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ValueError(f"Override {key!r}: {p!r} is not a mapping")
    d[leaf] = value


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML config with dotted overrides (`a.b.c=value`, values parsed as YAML).

    `model:` may be a path to a model YAML (relative to cwd, else to the config file), and
    `model=<path>` on the CLI swaps the whole model before `model.x=...` overrides apply.
    """
    path = Path(path)
    raw = _load_yaml(path)
    parsed = []
    for ov in overrides or []:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        parsed.append((key.strip(), yaml.safe_load(val)))
    for key, val in parsed:
        if key == "model":
            raw["model"] = val
    if isinstance(raw.get("model"), str):
        mpath = Path(raw["model"])
        if not mpath.exists():
            mpath = path.parent / mpath
        raw["model"] = _load_yaml(mpath)
    for key, val in parsed:
        if key != "model":
            _set_dotted(raw, key, val)
    return ExperimentConfig.model_validate(raw)
