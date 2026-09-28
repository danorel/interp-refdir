from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from interptemp.sites import Site


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_activations(path: str | Path, acts: dict[Site, torch.Tensor], **meta: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": {str(k): v.cpu() for k, v in acts.items()}, "meta": meta}, path)
    return path


def load_activations(path: str | Path) -> tuple[dict[Site, torch.Tensor], dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return {Site.parse(k): v for k, v in blob["acts"].items()}, blob["meta"]
