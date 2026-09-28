"""Named locations in the residual stream / sublayers.

A Site is backend-agnostic: backends map it to their own hook points. Sites carry an
execution-order key because nnsight (>=0.5) requires modules to be accessed in the order
they run during the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SiteKind = Literal["embed", "resid_pre", "attn_out", "mlp_out", "resid_post"]

# Rank of each kind within a layer's forward pass.
_KIND_RANK: dict[str, int] = {
    "embed": 0,
    "resid_pre": 0,
    "attn_out": 1,
    "mlp_out": 2,
    "resid_post": 3,
}


@dataclass(frozen=True, order=False)
class Site:
    kind: SiteKind
    layer: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KIND_RANK:
            raise ValueError(f"Unknown site kind {self.kind!r}; expected one of {list(_KIND_RANK)}")
        if self.kind == "embed" and self.layer is not None:
            raise ValueError("embed site takes no layer")
        if self.kind != "embed" and (self.layer is None or self.layer < 0):
            raise ValueError(f"{self.kind} site needs a non-negative layer, got {self.layer}")

    @property
    def order_key(self) -> tuple[int, int]:
        if self.kind == "embed":
            return (-1, 0)
        return (self.layer_index, _KIND_RANK[self.kind])

    @property
    def layer_index(self) -> int:
        if self.layer is None:
            raise ValueError(f"{self.kind} site has no layer")
        return self.layer

    def __str__(self) -> str:
        return self.kind if self.layer is None else f"{self.kind}.{self.layer}"

    @classmethod
    def parse(cls, s: str | Site) -> Site:
        """Parse 'resid_post.12' / 'embed'."""
        if isinstance(s, Site):
            return s
        kind, _, layer = s.partition(".")
        return cls(kind, int(layer) if layer else None)  # type: ignore[arg-type]


def sort_sites(sites: list[Site]) -> list[Site]:
    return sorted(set(sites), key=lambda s: s.order_key)


def resid_post(layers: list[int] | range) -> list[Site]:
    return [Site("resid_post", i) for i in layers]
