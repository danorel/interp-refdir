"""Sanity checks to run on every new model / pod before trusting any result.

Each check returns {"passed": bool, ...diagnostics}. Used by tests (small model, CPU) and by
experiments/sanity (real model, GPU).
"""

from __future__ import annotations

from typing import Any

import torch

from interptemp.config import GenerationConfig
from interptemp.directions import project, random_like
from interptemp.experiment import Experiment
from interptemp.interventions import AddVector, DirectionalAblation, Lambda
from interptemp.models.base import InterpModel
from interptemp.sites import Site, resid_post
from interptemp.utils import resolve_dtype

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "def fibonacci(n):",
]


def check_hf_parity(model: InterpModel, prompts: list[str], atol: float = 1e-3) -> dict[str, Any]:
    """Our backend's logits vs a plain HF forward on the same tokens.

    Loads a second copy of the weights: needs 2x model memory.
    """
    from transformers import AutoModelForCausalLM

    enc = model.encode(prompts)
    ours = model.logits(prompts, positions=None, batch_size=len(prompts))
    hf = AutoModelForCausalLM.from_pretrained(
        model.cfg.name,
        dtype=resolve_dtype(model.cfg.dtype),
        device_map=model.cfg.device_map,
        revision=model.cfg.revision,
    )
    with torch.inference_mode():
        ref = hf(**{k: v.to(hf.device) for k, v in enc.items()}).logits.float().cpu()
    del hf
    mask = enc["attention_mask"].bool()
    diff = (ours - ref).abs()[mask].max().item()
    argmax_agree = (ours.argmax(-1) == ref.argmax(-1))[mask].float().mean().item()
    return {"passed": diff <= atol, "max_abs_diff": diff, "argmax_agreement": argmax_agree}


def check_chat_template(model: InterpModel) -> dict[str, Any]:
    """Template renders, round-trips through the tokenizer, and has at most one BOS."""
    msgs = [{"role": "user", "content": "Hi there!"}]
    text = model.format_chat(msgs)
    ids = model.encode([text])["input_ids"][0].tolist()
    tok = model.tokenizer
    n_bos = ids.count(tok.bos_token_id) if tok.bos_token_id is not None else 0
    roundtrip = tok.decode(ids)
    expected = text if not tok.bos_token or text.startswith(tok.bos_token) else tok.bos_token + text
    return {
        "passed": n_bos <= 1 and roundtrip == expected and "Hi there!" in text,
        "n_bos": n_bos,
        "roundtrip_ok": roundtrip == expected,
        "rendered": text,
    }


def check_batch_invariance(model: InterpModel, prompts: list[str], atol: float) -> dict[str, Any]:
    """Last-token logits for each prompt alone vs inside a left-padded batch.

    Nonzero diffs are expected in bf16 (different kernels/shapes); argmax should agree.
    """
    batched = model.logits(prompts, batch_size=len(prompts))[:, -1]
    single = torch.cat([model.logits([p])[:, -1] for p in prompts])
    diff = (batched - single).abs().max().item()
    agree = (batched.argmax(-1) == single.argmax(-1)).float().mean().item()
    return {
        "passed": diff <= atol and agree == 1.0,
        "max_abs_diff": diff,
        "argmax_agreement": agree,
    }


def check_zero_steer_identity(model: InterpModel, prompts: list[str]) -> dict[str, Any]:
    layer = model.num_layers // 2
    base = model.logits(prompts)
    zero = AddVector(torch.zeros(model.hidden_size), [Site("resid_post", layer)])
    steered = model.logits(prompts, interventions=[zero])
    diff = (base - steered).abs().max().item()
    return {"passed": diff == 0.0, "max_abs_diff": diff}


def check_steering_has_effect(model: InterpModel, prompts: list[str]) -> dict[str, Any]:
    """Guards against interventions silently not being applied (e.g. wrong site mapping)."""
    layer = model.num_layers // 2
    resid = model.activations(prompts, [Site("resid_post", layer)])[Site("resid_post", layer)]
    v = random_like(resid[0, -1], seed=0) * 4  # 4x typical resid norm: must change something
    base = model.logits(prompts)
    steered = model.logits(prompts, interventions=[AddVector(v, [Site("resid_post", layer)])])
    diff = (base - steered).abs().max().item()
    return {"passed": diff > 1e-2, "max_abs_diff": diff}


def check_ablation_projection(
    model: InterpModel, prompts: list[str], rel_tol: float
) -> dict[str, Any]:
    """Ablating r̂ everywhere leaves ~0 projection on r̂ in every resid_post."""
    g = torch.Generator().manual_seed(0)
    r = torch.randn(model.hidden_size, generator=g)
    abl = DirectionalAblation.everywhere(r, model.num_layers)
    sites = resid_post(range(model.num_layers))
    acts = model.activations(prompts, sites, positions=None, interventions=[abl])
    worst = 0.0
    for s in sites:
        h = acts[s].float()
        norms = h.norm(dim=-1).clamp_min(1e-6)
        worst = max(worst, (project(h, r).abs() / norms).max().item())
    return {"passed": worst <= rel_tol, "max_rel_projection": worst}


def check_generation(model: InterpModel, prompts: list[str]) -> dict[str, Any]:
    """Generation works, zero-steer is a no-op, and interventions fire at every decode step."""
    gen = GenerationConfig(max_new_tokens=8, batch_size=len(prompts))
    chat = [model.format_chat([{"role": "user", "content": p}]) for p in prompts]
    site = Site("resid_post", model.num_layers // 2)
    outs = model.generate(chat, gen)
    outs_zero = model.generate(
        chat, gen, interventions=[AddVector(torch.zeros(model.hidden_size), [site])]
    )

    seq_lens: list[int] = []

    def spy(h: torch.Tensor, _: Site) -> torch.Tensor:
        seq_lens.append(h.shape[1])
        return h

    model.generate(chat, gen, interventions=[Lambda(spy, [site])])
    # prefill (full prompt) + one call per decoded token (seq == 1)
    fires_on_decode = len(seq_lens) > 1 and all(n == 1 for n in seq_lens[1:])
    return {
        "passed": len(outs) == len(prompts) and all(outs) and outs == outs_zero and fires_on_decode,
        "zero_steer_generation_identical": outs == outs_zero,
        "intervention_calls": len(seq_lens),
        "fires_on_decode": fires_on_decode,
        "samples": outs,
    }


def run_all(
    model: InterpModel,
    prompts: list[str] | None = None,
    hf_parity: bool = True,
    batch_atol: float = 0.5,
    ablation_rel_tol: float = 1e-2,
) -> dict[str, dict[str, Any]]:
    prompts = prompts or DEFAULT_PROMPTS
    results = {
        "chat_template": check_chat_template(model),
        "zero_steer_identity": check_zero_steer_identity(model, prompts),
        "steering_has_effect": check_steering_has_effect(model, prompts),
        "ablation_projection": check_ablation_projection(model, prompts, ablation_rel_tol),
        "batch_invariance": check_batch_invariance(model, prompts, batch_atol),
        "generation": check_generation(model, prompts),
    }
    if hf_parity:
        results["hf_parity"] = check_hf_parity(model, prompts)
    return results


class SanityExperiment(Experiment):
    """`uv run interp-run experiments/sanity/config.yaml` on every new pod/model."""

    def run(self) -> dict[str, Any]:
        results = run_all(self.imodel, **self.params)
        for name, r in results.items():
            status = "PASS" if r["passed"] else "FAIL"
            diag = {k: v for k, v in r.items() if k not in ("passed", "samples", "rendered")}
            self.log.info(f"[{status}] {name} {diag}")
        return {"all_passed": all(r["passed"] for r in results.values()), **results}
