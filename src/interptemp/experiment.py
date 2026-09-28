"""Experiment base class: one config in, one reproducible run directory out.

Run dir layout: outputs/<name>/<timestamp>/
    config.yaml   resolved config (after overrides)
    meta.json     git sha/dirty, argv, timing
    *.jsonl, *.pt results written via self.save_*
"""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
import yaml

from interptemp.config import ExperimentConfig
from interptemp.judges.base import Judge, build_judge
from interptemp.models.base import Generator, InterpModel, build_model
from interptemp.sites import Site
from interptemp.store import save_activations, write_jsonl
from interptemp.utils import get_logger, git_state, seed_everything


class Experiment(ABC):
    def __init__(self, cfg: ExperimentConfig, run_dir: str | Path | None = None):
        self.cfg = cfg
        self.params = cfg.params
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir or Path(cfg.output_dir) / cfg.name / stamp)
        self.log = get_logger(cfg.name)

    # ---- lazily built components (only what the experiment touches gets loaded) ---------

    @cached_property
    def model(self) -> Generator:
        self.log.info(f"loading {self.cfg.model.backend}:{self.cfg.model.name}")
        return build_model(self.cfg.model)

    @property
    def imodel(self) -> InterpModel:
        """`model`, asserting it exposes internals (not generation-only)."""
        if not isinstance(self.model, InterpModel):
            raise TypeError(f"{type(self.model).__name__} has no internals access")
        return self.model

    @cached_property
    def judge(self) -> Judge:
        if self.cfg.judge is None:
            raise ValueError("config has no `judge` section")
        return build_judge(self.cfg.judge, **self.judge_kwargs())

    def judge_kwargs(self) -> dict[str, Any]:
        """Override to inject a task-specific rubric/labels into the configured judge."""
        return {}

    # ---- outputs ------------------------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def save_jsonl(self, name: str, rows: list[dict[str, Any]]) -> Path:
        p = write_jsonl(self.path(name), rows)
        self.log.info(f"wrote {len(rows)} rows -> {p}")
        return p

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
        return p

    def save_tensor(
        self, name: str, t: torch.Tensor | dict[Site, torch.Tensor], **meta: Any
    ) -> Path:
        if isinstance(t, dict):
            return save_activations(self.path(name), t, **meta)
        torch.save(t.cpu(), self.path(name))
        return self.path(name)

    # ---- lifecycle ----------------------------------------------------------------------

    def setup(self) -> None:
        seed_everything(self.cfg.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(self.cfg.model_dump(), sort_keys=False)
        )
        self._t0 = time.time()
        self.save_json(
            "meta.json", {"git": git_state(), "argv": sys.argv, "started": datetime.now()}
        )
        self.log.info(f"run dir: {self.run_dir}")

    @abstractmethod
    def run(self) -> dict[str, Any] | None:
        """Do the work; return summary metrics (saved to summary.json)."""

    def execute(self) -> dict[str, Any] | None:
        self.setup()
        summary = self.run()
        if summary is not None:
            self.save_json("summary.json", summary)
            self.log.info(f"summary: {json.dumps(summary, default=str)}")
        self.log.info(f"done in {time.time() - self._t0:.1f}s -> {self.run_dir}")
        return summary
