"""Judge-vs-human agreement. Validate every judge on hand labels before trusting it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any


def _paired(a: Sequence[Hashable | None], b: Sequence[Hashable | None]) -> list[tuple[Any, Any]]:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]


def cohen_kappa(a: Sequence[Hashable | None], b: Sequence[Hashable | None]) -> float:
    """Cohen's kappa over pairs where both labels are present (None = missing)."""
    pairs = _paired(a, b)
    if not pairs:
        raise ValueError("no overlapping labels")
    n = len(pairs)
    p_o = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    p_e = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n**2
    if p_e == 1:  # both raters constant and identical: kappa undefined, agreement perfect
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def agreement_report(
    judge: Sequence[Hashable | None], human: Sequence[Hashable | None]
) -> dict[str, Any]:
    pairs = _paired(judge, human)
    return {
        "n": len(judge),
        "n_scored": len(pairs),
        "n_judge_missing": sum(j is None for j in judge),
        "accuracy": sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan"),
        "kappa": cohen_kappa(judge, human) if pairs else float("nan"),
        "confusion": {
            f"judge={x}|human={y}": c for (x, y), c in sorted(Counter(pairs).items(), key=str)
        },
    }
