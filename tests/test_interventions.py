import pytest
import torch

from interptemp.directions import project
from interptemp.interventions import AddVector, DirectionalAblation, group_by_site
from interptemp.sites import Site

S = Site("resid_post", 0)


def test_add_vector_all_positions():
    h = torch.zeros(2, 3, 4)
    v = torch.arange(4.0)
    out = AddVector(v, [S], scale=2.0)(h, S)
    assert torch.equal(out, (2 * v).expand(2, 3, 4))


def test_add_vector_positions_and_out_of_range():
    h = torch.zeros(1, 3, 2)
    out = AddVector(torch.ones(2), [S], positions=[-1, 5])(h, S)  # 5 out of range -> ignored
    assert out[0, :2].abs().sum() == 0
    assert torch.equal(out[0, 2], torch.ones(2))


def test_add_vector_decode_step_seq1():
    # During generation after prefill seq == 1; positions=[-1] must still hit the new token.
    out = AddVector(torch.ones(2), [S], positions=[-1])(torch.zeros(1, 1, 2), S)
    assert torch.equal(out, torch.ones(1, 1, 2))


def test_add_vector_preserves_dtype():
    h = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    assert AddVector(torch.ones(4), [S])(h, S).dtype == torch.bfloat16


def test_ablation_removes_component_and_keeps_rest():
    torch.manual_seed(0)
    r = torch.randn(16)
    h = torch.randn(3, 5, 16)
    out = DirectionalAblation(r, [S])(h, S)
    assert project(out, r).abs().max() < 1e-5
    # orthogonal part untouched
    ru = r / r.norm()
    assert torch.allclose(out, h - (h @ ru)[..., None] * ru, atol=1e-6)


def test_ablation_idempotent():
    r, h = torch.randn(8), torch.randn(2, 3, 8)
    abl = DirectionalAblation(r, [S])
    once = abl(h, S)
    assert torch.allclose(abl(once, S), once, atol=1e-6)


def test_ablation_everywhere_sites():
    abl = DirectionalAblation.everywhere(torch.ones(4), num_layers=2)
    assert [str(s) for s in abl.sites] == [
        "embed",
        "attn_out.0",
        "mlp_out.0",
        "attn_out.1",
        "mlp_out.1",
    ]


def test_validation():
    with pytest.raises(ValueError):
        DirectionalAblation(torch.zeros(4), [S])
    with pytest.raises(ValueError):
        AddVector(torch.ones(2, 2), [S])
    with pytest.raises(ValueError):
        AddVector(torch.ones(2), [])


def test_group_by_site_keeps_order():
    a, b = AddVector(torch.ones(2), [S]), AddVector(torch.ones(2), [S, Site("embed")])
    g = group_by_site([a, b])
    assert g[S] == [a, b]
    assert g[Site("embed")] == [b]
