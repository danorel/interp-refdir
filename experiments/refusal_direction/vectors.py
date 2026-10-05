"""Building the candidate directions, their controls, and the geometry diagnostics.

Separated from the experiment so the "what vector do we intervene with" decisions sit in one
place: the mean-diff sweep grid, the two control families, and the check that a control is
not trivially passable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from interptemp.directions import mean_diff, random_like, unit
from interptemp.models.base import InterpModel
from interptemp.sites import Site

ZERO_NORM_TOL = 1e-6


def build_direction_grid(
    model: InterpModel,
    harmful_prompts: Sequence[str],
    harmless_prompts: Sequence[str],
    sites: Sequence[Site],
    positions: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    """mean(harmful) - mean(harmless) at every (site, position): [n_sites, n_positions, d].

    The two sets are averaged independently, so pairing between them is irrelevant: it is
    their topic *distributions* that need to match, not their row order.
    """
    acts = {
        name: model.activations(prompts, sites, positions=positions, batch_size=batch_size)
        for name, prompts in (("harmful", harmful_prompts), ("harmless", harmless_prompts))
    }
    return torch.stack([mean_diff(acts["harmful"][s], acts["harmless"][s]) for s in sites])


def is_degenerate(direction: torch.Tensor) -> bool:
    """Layer-0 input is the same embedding for every prompt at template positions, so its
    mean-diff is exactly zero — not a candidate."""
    return bool(direction.norm() <= ZERO_NORM_TOL)


def contrast_direction(
    model: InterpModel,
    question_prompts: Sequence[str],
    imperative_prompts: Sequence[str],
    site: Site,
    pos: int,
    norm: float,
) -> torch.Tensor:
    """A control built exactly like `r`, from a harmless contrast with no refusal in it.

    Harder to pass than a random vector: it is a real mean-diff direction with real
    structure, so an effect it fails to reproduce is unlikely to be "any direction works".
    Scaled to `norm` so the addition test compares like with like.
    """
    acts = {
        name: model.activations(prompts, [site], positions=[pos])[site][:, 0]
        for name, prompts in (("q", question_prompts), ("i", imperative_prompts))
    }
    return unit(mean_diff(acts["q"], acts["i"])) * norm


def control_vectors(r: torch.Tensor, n_random: int) -> dict[str, torch.Tensor]:
    """`r` plus norm-matched random directions, keyed by the names used in the reports."""
    return {"r": r} | {f"random_{seed}": random_like(r, seed=seed) for seed in range(n_random)}


def ablation_diagnostics(
    model: InterpModel,
    vectors: dict[str, torch.Tensor],
    prompts_by_set: dict[str, Sequence[str]],
    site: Site,
    pos: int,
) -> dict[str, Any]:
    """How much of ||h||² each vector's ablation removes, and each vector's cosine to `r`.

    An isotropic random direction removes ~1/d of the norm, so "random ablation changed
    nothing" is a trivially passed control; these numbers say how trivial.
    """
    out: dict[str, Any] = {}
    for set_name, prompts in prompts_by_set.items():
        h = model.activations(prompts, [site], positions=[pos])[site][:, 0].float()
        out[f"frac_removed/{set_name}"] = {
            name: ((h @ unit(v.float())) ** 2 / h.pow(2).sum(-1)).mean().item()
            for name, v in vectors.items()
        }
    out["cos_to_r"] = {
        name: torch.nn.functional.cosine_similarity(v.float(), vectors["r"].float(), dim=0).item()
        for name, v in vectors.items()
    }
    return out
