"""Direction utilities. Tensors are [..., d]; directions are 1-D [d]."""

from __future__ import annotations

import torch


def unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    if (n == 0).any():
        raise ValueError("cannot normalize a zero vector")
    return v / n


def mean_diff(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """mean(pos) - mean(neg) over the batch dim (dim 0), in fp32. Shapes [N, ..., d]."""
    if pos.shape[1:] != neg.shape[1:]:
        raise ValueError(f"shape mismatch {tuple(pos.shape)} vs {tuple(neg.shape)}")
    return pos.float().mean(0) - neg.float().mean(0)


def random_like(v: torch.Tensor, seed: int, match_norm: bool = True) -> torch.Tensor:
    """Isotropic random direction(s) shaped like v; norm-matched per last-dim vector.

    The standard control for direction interventions: same op, same norm, no semantics.
    """
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g, dtype=torch.float32)
    if match_norm:
        r = unit(r) * v.float().norm(dim=-1, keepdim=True)
    return r.to(v.dtype)


def project(h: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Scalar projection of h onto unit(direction): [..., d] -> [...]."""
    return h.float() @ unit(direction.float())


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
