import pytest
import torch

from interptemp.directions import cosine, mean_diff, random_like, unit


def test_mean_diff():
    pos = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    neg = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    assert torch.equal(mean_diff(pos, neg), torch.tensor([2.0, -2.0]))


def test_mean_diff_shape_mismatch():
    with pytest.raises(ValueError):
        mean_diff(torch.zeros(2, 3), torch.zeros(2, 4))


def test_random_like_norm_matched_and_deterministic():
    v = torch.randn(512) * 7
    r1, r2 = random_like(v, seed=1), random_like(v, seed=1)
    assert torch.equal(r1, r2)
    assert torch.isclose(r1.norm(), v.norm(), rtol=1e-5)
    assert not torch.equal(r1, random_like(v, seed=2))
    assert cosine(r1, v).abs() < 0.2  # random in high-d ~ orthogonal


def test_unit_zero_raises():
    with pytest.raises(ValueError):
        unit(torch.zeros(3))
