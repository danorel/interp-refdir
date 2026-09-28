from __future__ import annotations

import logging
import os
import random
import subprocess
from collections.abc import Iterator, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code uses the global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def batched[T](xs: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    if n <= 0:
        raise ValueError(f"batch size must be positive, got {n}")
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def git_state() -> dict[str, str | bool]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"sha": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def get_logger(name: str = "interptemp") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def resolve_dtype(name: str) -> torch.dtype:
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {list(dtypes)}")
    return dtypes[name]
