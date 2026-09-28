"""Activation edits. An Intervention declares the Sites it touches and maps h -> h'.

Interventions are pure tensor functions, so they are unit-testable without a model and
work unchanged across backends. `h` is [batch, seq, d]; during generation after the first
step seq == 1 (only the new token), so `positions` refer to the current chunk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from interptemp.sites import Site


class Intervention(ABC):
    def __init__(self, sites: Sequence[Site | str]):
        self.sites = [Site.parse(s) for s in sites]
        if not self.sites:
            raise ValueError(f"{type(self).__name__} needs at least one site")

    @abstractmethod
    def __call__(self, h: torch.Tensor, site: Site) -> torch.Tensor: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sites={[str(s) for s in self.sites]})"


def _select(h: torch.Tensor, positions: Sequence[int] | None) -> torch.Tensor:
    """Boolean mask [seq] selecting `positions` (negative ok); None -> all."""
    mask = torch.zeros(h.shape[1], dtype=torch.bool, device=h.device)
    if positions is None:
        mask[:] = True
    else:
        n = h.shape[1]
        idx = [p for p in positions if -n <= p < n]  # out-of-range (e.g. during decode) -> skip
        mask[idx] = True
    return mask


class AddVector(Intervention):
    """h += scale * vector at the given positions (activation steering)."""

    def __init__(
        self,
        vector: torch.Tensor,
        sites: Sequence[Site | str],
        scale: float = 1.0,
        positions: Sequence[int] | None = None,
    ):
        super().__init__(sites)
        if vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {tuple(vector.shape)}")
        self.vector, self.scale, self.positions = vector, scale, positions

    def __call__(self, h: torch.Tensor, site: Site) -> torch.Tensor:
        v = (self.scale * self.vector).to(h.device, h.dtype)
        mask = _select(h, self.positions)[None, :, None]
        return h + mask * v


class DirectionalAblation(Intervention):
    """Remove the component along `direction`: h -= (h·r̂) r̂.

    Use `DirectionalAblation.everywhere` to stop the model ever writing r̂ into the residual
    stream (embed + every attn/mlp output) — equivalent to weight orthogonalization.
    """

    def __init__(self, direction: torch.Tensor, sites: Sequence[Site | str]):
        super().__init__(sites)
        if direction.ndim != 1:
            raise ValueError(f"direction must be 1-D, got shape {tuple(direction.shape)}")
        norm = direction.norm()
        if norm == 0:
            raise ValueError("direction has zero norm")
        self.direction = direction / norm

    def __call__(self, h: torch.Tensor, site: Site) -> torch.Tensor:
        # Project in fp32: bf16 dot products over d~4k lose enough precision to leave residue.
        r = self.direction.to(h.device, torch.float32)
        hf = h.float()
        return (hf - (hf @ r)[..., None] * r).to(h.dtype)

    @classmethod
    def everywhere(cls, direction: torch.Tensor, num_layers: int) -> DirectionalAblation:
        sites = [Site("embed")]
        for i in range(num_layers):
            sites += [Site("attn_out", i), Site("mlp_out", i)]
        return cls(direction, sites)


class Lambda(Intervention):
    """Ad-hoc intervention from a function; handy in notebooks before writing a class."""

    def __init__(
        self, fn: Callable[[torch.Tensor, Site], torch.Tensor], sites: Sequence[Site | str]
    ):
        super().__init__(sites)
        self.fn = fn

    def __call__(self, h: torch.Tensor, site: Site) -> torch.Tensor:
        return self.fn(h, site)


def group_by_site(interventions: Sequence[Intervention]) -> dict[Site, list[Intervention]]:
    """Site -> interventions, preserving the given order within a site."""
    out: dict[Site, list[Intervention]] = {}
    for iv in interventions:
        for s in iv.sites:
            out.setdefault(s, []).append(iv)
    return out
