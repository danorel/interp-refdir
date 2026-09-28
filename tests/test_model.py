"""Integration tests on Qwen3-0.6B (same family/template as Qwen3-8B). `make test-model`."""

import pytest
import torch

from interptemp import sanity
from interptemp.config import GenerationConfig, ModelConfig
from interptemp.models.nnterp_model import NnterpModel
from interptemp.sites import Site

pytestmark = pytest.mark.model

PROMPTS = sanity.DEFAULT_PROMPTS


@pytest.fixture(scope="module")
def model():
    return NnterpModel(
        ModelConfig(
            name="Qwen/Qwen3-0.6B",
            dtype="float32",
            device_map="cpu",
            chat_template_kwargs={"enable_thinking": False},
        )
    )


def test_hf_parity(model):
    r = sanity.check_hf_parity(model, PROMPTS, atol=1e-4)
    assert r["passed"], r


def test_chat_template(model):
    r = sanity.check_chat_template(model)
    assert r["passed"], r
    assert r["n_bos"] == 0  # Qwen has no BOS
    assert "<think>\n\n</think>" in r["rendered"]  # enable_thinking=False from config
    assert "<think>\n\n</think>" not in model.format_chat(
        [{"role": "user", "content": "x"}], enable_thinking=True
    )


def test_batch_invariance(model):
    r = sanity.check_batch_invariance(model, PROMPTS, atol=1e-3)  # fp32: should be tight
    assert r["passed"], r


def test_zero_steer_identity(model):
    r = sanity.check_zero_steer_identity(model, PROMPTS)
    assert r["passed"], r


def test_steering_has_effect(model):
    r = sanity.check_steering_has_effect(model, PROMPTS)
    assert r["passed"], r


def test_ablation_projection(model):
    r = sanity.check_ablation_projection(model, PROMPTS, rel_tol=1e-4)
    assert r["passed"], r


def test_generation_and_zero_steer(model):
    r = sanity.check_generation(model, PROMPTS)
    assert r["passed"], r


def test_activation_shapes_and_positions(model):
    sites = [Site("resid_pre", 2), Site("resid_post", 1)]  # deliberately out of order
    acts = model.activations(PROMPTS, sites, positions=[-2, -1], batch_size=2)
    for s in sites:
        assert acts[s].shape == (len(PROMPTS), 2, model.hidden_size)
    # resid_post.1 is the input of layer 2
    assert torch.allclose(acts[Site("resid_post", 1)], acts[Site("resid_pre", 2)])


def test_full_sequence_across_batches_left_padded(model):
    acts = model.activations(PROMPTS, [Site("resid_post", 0)], positions=None, batch_size=1)
    t = model.encode(PROMPTS)["input_ids"].shape[1]
    assert acts[Site("resid_post", 0)].shape == (len(PROMPTS), t, model.hidden_size)


def test_thinking_generation_keeps_think_tags(model):
    prompt = model.format_chat([{"role": "user", "content": "What is 2+2?"}], enable_thinking=True)
    out = model.generate([prompt], GenerationConfig(max_new_tokens=16))[0]
    assert out.startswith("<think>")
