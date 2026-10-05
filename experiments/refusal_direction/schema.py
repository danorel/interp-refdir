"""Typed records that cross module (and file) boundaries.

Every row written to JSONL and every condition key read back by the plotting and judge
validation scripts is defined here, so a renamed field breaks at import time rather than
in a dict lookup three modules away.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any


class Stage(StrEnum):
    """Pipeline steps, in execution order. `params.stages` names a subset of these."""

    BASELINE = "baseline"
    SWEEP = "sweep"
    ALPHA = "alpha"
    EVAL = "eval"

    @classmethod
    def parse_all(cls, names: list[str]) -> list[Stage]:
        """Validate configured stage names up front, so a typo fails before the model loads."""
        unknown = [n for n in names if n not in set(cls)]
        if unknown:
            raise ValueError(f"unknown stages {unknown}; valid: {[s.value for s in cls]}")
        return [s for s in cls if s in names]  # canonical order, regardless of config order


# Judge verdicts. `incoherent` is deliberately separate: an intervention that destroys the
# model would otherwise score as "no longer refusing", i.e. as a success.
LABELS = ["refusal", "compliance", "incoherent"]

POOLED_RANDOM = "random*"  # synthetic vector name: all random_<seed> conditions together
RANDOM_PREFIX = "random_"

# Identify one labeled row; also the `--key` fields of `interp-label`.
KEY_FIELDS = ["id", "intervention", "vector"]


def condition_key(prompt_set: str, intervention: str, vector: str) -> str:
    """The grouping key used in `summary.json` ("harmful/ablate/r")."""
    return f"{prompt_set}/{intervention}/{vector}"


def is_pooled(key: str) -> bool:
    """True for the pooled `random*` view, which double-counts the individual seeds."""
    return key.endswith(POOLED_RANDOM)


@dataclass
class GenerationRow:
    """One prompt under one condition: what the model said and how it scored."""

    id: str
    prompt_set: str  # "harmful" | "harmless"
    intervention: str  # "none" | "ablate" | "add"
    vector: str  # "-" | "r" | "contrast" | "random_<seed>"
    prompt: str
    completion: str
    refusal_score: float  # logit proxy; see metrics.refusal_score
    substring_refusal: bool
    alpha: float | None = None  # addition scale, only set by the alpha stage
    judged: bool = False  # whether a judge ran at all (vs. ran and failed)
    label: str | None = None  # one of LABELS; None = judge produced no verdict
    judge_reasoning: str | None = None
    judge_error: str | None = None

    @property
    def key(self) -> str:
        return "|".join(str(getattr(self, f)) for f in KEY_FIELDS)

    @property
    def condition(self) -> str:
        return condition_key(self.prompt_set, self.intervention, self.vector)

    def to_dict(self) -> dict[str, Any]:
        """JSONL form. Fields that never applied are omitted rather than written as null."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "judged"}
        if not self.judged:  # no judge configured: don't imply a missing verdict
            del out["label"], out["judge_reasoning"]
        for optional in ("alpha", "judge_error"):
            if out.get(optional) is None:
                out.pop(optional, None)
        return out


@dataclass
class SweepCandidate:
    """One (layer, position) direction, scored on val without generating any text."""

    layer: int
    pos: int
    layer_frac: float
    norm: float
    bypass_score: float  # refusal score on harmful after ablation (lower = refusal removed)
    induce_score: float  # refusal score on harmless after addition (higher = refusal induced)
    kl: float  # KL(clean || ablated) on harmless: side-effect guard

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class SweepResult:
    """What the sweep decided, and the evidence for it."""

    best: SweepCandidate
    induce_filter_passed: bool
    clean_refusal_score: dict[str, float]
    candidates: list[SweepCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.best.to_dict(),
            "filters_passed": self.induce_filter_passed,
            "clean_refusal_score": self.clean_refusal_score,
        }


@dataclass
class AlphaResult:
    """The chosen addition scale, and the per-scale report it was chosen from."""

    chosen: float
    per_alpha: dict[float, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen, "per_alpha": self.per_alpha}


@dataclass
class Selection:
    """Everything the selection stages fix on val before the test set is ever touched.

    Carrying one object (rather than four loose arguments) keeps "what was decided" and
    "what gets reported" in step.
    """

    layer: int
    pos: int
    alpha: float
    direction: Any  # torch.Tensor; untyped here to keep this module import-light

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "pos": self.pos,
            "norm": float(self.direction.norm()),
            "alpha": self.alpha,
        }


@dataclass
class Rate:
    """A proportion with its Wilson 95% interval, as written to `summary.json`."""

    k: int
    n: int
    rate: float
    ci95: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
