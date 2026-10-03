import math

import pytest
import torch

from experiments.refusal_direction.experiment import (
    kl_div,
    rate,
    refusal_score,
    select_direction,
    summarize,
    wilson,
)


def test_refusal_score_is_log_odds_of_refusal_tokens():
    # p(token 0) + p(token 1) = 0.5 -> log-odds 0; all mass on token 2 -> very negative
    logits = torch.log(torch.tensor([[0.25, 0.25, 0.5], [0.0, 0.0, 1.0]]))
    s = refusal_score(logits, [0, 1])
    assert s[0].item() == pytest.approx(0.0, abs=1e-5)
    assert s[1].item() < -15


def test_kl_div_zero_for_identical_and_positive_otherwise():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    assert kl_div(a, a).item() == pytest.approx(0.0, abs=1e-6)
    assert kl_div(a, torch.tensor([[3.0, 2.0, 1.0]])).item() > 0


def test_wilson_contains_point_estimate_and_handles_extremes():
    lo, hi = wilson(8, 16)
    assert lo < 0.5 < hi
    assert wilson(0, 16)[0] == 0.0
    assert wilson(16, 16)[1] == 1.0
    assert all(math.isnan(x) for x in wilson(0, 0))
    # More data -> tighter interval
    assert (wilson(32, 64)[1] - wilson(32, 64)[0]) < (hi - lo)


def test_rate_keeps_missing_labels_in_denominator():
    r = rate(["refusal", None, "compliance", "refusal"], "refusal")
    assert (r["k"], r["n"], r["rate"]) == (2, 4, 0.5)


def _row(layer, bypass, induce, kl):
    return {"layer": layer, "pos": -1, "bypass_score": bypass, "induce_score": induce, "kl": kl}


def test_select_direction_applies_filters_before_min_bypass():
    rows = [
        _row(0, bypass=-9.0, induce=-1.0, kl=0.01),  # best bypass but can't induce refusal
        _row(1, bypass=-8.0, induce=2.0, kl=0.5),  # best passing induce, but breaks harmless
        _row(2, bypass=-3.0, induce=1.0, kl=0.05),
        _row(3, bypass=-5.0, induce=0.5, kl=0.02),
    ]
    best, passed = select_direction(rows, max_kl=0.1)
    assert passed
    assert best["layer"] == 3


def test_select_direction_fallback_still_rejects_model_breaking_directions():
    rows = [
        _row(0, bypass=-17.0, induce=-1.0, kl=15.0),  # "bypasses" by breaking the model
        _row(1, bypass=-2.0, induce=-1.0, kl=0.0),
        _row(2, bypass=-1.0, induce=-1.0, kl=0.0),
    ]
    best, passed = select_direction(rows, max_kl=0.1)
    assert not passed
    assert best["layer"] == 1


def test_select_direction_raises_when_every_ablation_breaks_the_model():
    with pytest.raises(ValueError):
        select_direction([_row(0, bypass=-9.0, induce=1.0, kl=5.0)], max_kl=0.1)


def test_summarize_groups_conditions_and_pools_random_seeds():
    def row(vector, label, sub):
        return {
            "prompt_set": "harmful",
            "intervention": "ablate",
            "vector": vector,
            "refusal_score": 1.0,
            "substring_refusal": sub,
            "label": label,
        }

    rows = [
        row("r", "compliance", False),
        row("r", "incoherent", False),
        row("random_0", "refusal", True),
        row("random_1", "refusal", True),
    ]
    s = summarize(rows)
    assert s["harmful/ablate/r"]["compliance"]["rate"] == 0.5
    assert s["harmful/ablate/r"]["incoherent"]["k"] == 1  # incoherent is not compliance
    assert s["harmful/ablate/random*"]["refusal"]["n"] == 2
    assert s["harmful/ablate/random*"]["substring_refusal"]["rate"] == 1.0
