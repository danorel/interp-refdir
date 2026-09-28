"""Tasks turn a data source into `Example`s with chat messages.

Subclass `Task` for anything custom (synthetic data, multi-turn, few-shot); override
`score` if the task has a programmatic metric so experiments don't need a judge.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from interptemp.registry import resolve
from interptemp.store import read_jsonl

Message = dict[str, str]


@dataclass
class Example:
    id: str
    messages: list[Message]
    label: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def user_text(self) -> str:
        return next(m["content"] for m in reversed(self.messages) if m["role"] == "user")


class Task(ABC):
    name: str = "task"

    @abstractmethod
    def load(self) -> list[Example]: ...

    def score(self, example: Example, completion: str) -> dict[str, Any]:
        """Programmatic metrics for one completion; empty if the task relies on a judge."""
        return {}


def _messages(text: str, system: str | None) -> list[Message]:
    msgs = [{"role": "system", "content": system}] if system else []
    return [*msgs, {"role": "user", "content": text}]


def split(
    examples: Sequence[Example], fracs: dict[str, float], seed: int = 0
) -> dict[str, list[Example]]:
    """Deterministic shuffle + split, e.g. {"train": .6, "val": .2, "test": .2}."""
    if abs(sum(fracs.values()) - 1) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fracs}")
    xs = list(examples)
    random.Random(seed).shuffle(xs)
    out, start = {}, 0
    names = list(fracs)
    for i, name in enumerate(names):
        end = len(xs) if i == len(names) - 1 else start + round(fracs[name] * len(xs))
        out[name], start = xs[start:end], end
    return out


class JsonlTask(Task):
    def __init__(
        self,
        path: str,
        text_field: str = "prompt",
        label_field: str | None = None,
        system: str | None = None,
        limit: int | None = None,
        name: str | None = None,
    ):
        self.path, self.text_field, self.label_field = path, text_field, label_field
        self.system, self.limit = system, limit
        self.name = name or path

    def load(self) -> list[Example]:
        rows = read_jsonl(self.path)[: self.limit]
        return [
            Example(
                id=str(r.get("id", i)),
                messages=_messages(r[self.text_field], self.system),
                label=r.get(self.label_field) if self.label_field else None,
                meta={k: v for k, v in r.items() if k not in (self.text_field, self.label_field)},
            )
            for i, r in enumerate(rows)
        ]


class HFDatasetTask(Task):
    def __init__(
        self,
        path: str,
        text_field: str,
        split: str = "train",
        subset: str | None = None,
        label_field: str | None = None,
        system: str | None = None,
        limit: int | None = None,
        name: str | None = None,
    ):
        self.path, self.subset, self.split = path, subset, split
        self.text_field, self.label_field = text_field, label_field
        self.system, self.limit = system, limit
        self.name = name or path

    def load(self) -> list[Example]:
        from datasets import load_dataset

        ds = load_dataset(self.path, self.subset, split=self.split)
        if self.limit is not None:
            ds = ds.select(range(min(self.limit, len(ds))))
        rows: list[dict[str, Any]] = ds.to_list()
        return [
            Example(
                id=f"{self.name}:{i}",
                messages=_messages(r[self.text_field], self.system),
                label=r[self.label_field] if self.label_field else None,
            )
            for i, r in enumerate(rows)
        ]


def build_task(spec: dict[str, Any]) -> Task:
    """spec = {"type": "jsonl" | "hf" | "module:Class", **kwargs}."""
    spec = dict(spec)
    return resolve("task", spec.pop("type"))(**spec)
