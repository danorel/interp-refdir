from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from interptemp.config import JudgeConfig
from interptemp.registry import resolve


@dataclass
class JudgeInput:
    prompt: str
    response: str
    extra: dict[str, Any] = field(default_factory=dict)  # extra template vars (e.g. forbidden word)


@dataclass
class JudgeResult:
    label: Any  # None = judge failed to produce a parseable verdict
    reasoning: str = ""
    raw: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Judge(ABC):
    @abstractmethod
    def judge(self, items: Sequence[JudgeInput]) -> list[JudgeResult]: ...


def build_judge(cfg: JudgeConfig, **overrides: Any) -> Judge:
    """Builds `cfg.backend` with cfg + cfg.kwargs + overrides (e.g. a task-specific rubric)."""
    cls = resolve("judge", cfg.backend)
    return cls(cfg, **{**cfg.kwargs, **overrides})
