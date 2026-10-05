import json
import math
from pathlib import Path

import pytest
import torch

from experiments.refusal_direction.metrics import (
    kl_div,
    rate,
    refusal_score,
    select_direction,
    summarize,
    wilson,
)
from experiments.refusal_direction.schema import (
    GenerationRow,
    Stage,
    SweepCandidate,
    condition_key,
    is_pooled,
)

COMMITTED_SUMMARY = Path("experiments/refusal_direction/results/qwen3-4b/summary.json")


# ---- scoring ----------------------------------------------------------------------------


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
    assert (wilson(32, 64)[1] - wilson(32, 64)[0]) < (hi - lo)  # more data -> tighter


def test_rate_keeps_missing_labels_in_denominator():
    r = rate(["refusal", None, "compliance", "refusal"], "refusal")
    assert (r.k, r.n, r.rate) == (2, 4, 0.5)


# ---- direction selection ----------------------------------------------------------------


def _candidate(layer, bypass, induce, kl):
    return SweepCandidate(
        layer=layer,
        pos=-1,
        layer_frac=0.5,
        norm=1.0,
        bypass_score=bypass,
        induce_score=induce,
        kl=kl,
    )


def test_select_direction_applies_filters_before_min_bypass():
    candidates = [
        _candidate(0, bypass=-9.0, induce=-1.0, kl=0.01),  # best bypass, can't induce refusal
        _candidate(1, bypass=-8.0, induce=2.0, kl=0.5),  # induces, but breaks harmless
        _candidate(2, bypass=-3.0, induce=1.0, kl=0.05),
        _candidate(3, bypass=-5.0, induce=0.5, kl=0.02),
    ]
    best, passed = select_direction(candidates, max_kl=0.1)
    assert passed
    assert best.layer == 3


def test_select_direction_fallback_still_rejects_model_breaking_directions():
    candidates = [
        _candidate(0, bypass=-17.0, induce=-1.0, kl=15.0),  # "bypasses" by breaking the model
        _candidate(1, bypass=-2.0, induce=-1.0, kl=0.0),
        _candidate(2, bypass=-1.0, induce=-1.0, kl=0.0),
    ]
    best, passed = select_direction(candidates, max_kl=0.1)
    assert not passed
    assert best.layer == 1


def test_select_direction_raises_when_every_ablation_breaks_the_model():
    with pytest.raises(ValueError):
        select_direction([_candidate(0, bypass=-9.0, induce=1.0, kl=5.0)], max_kl=0.1)


# ---- rows and aggregation ---------------------------------------------------------------


def _row(vector, label=None, judged=False, substring=False, prompt_set="harmful", **kw):
    return GenerationRow(
        id=kw.pop("id", "x"),
        prompt_set=prompt_set,
        intervention=kw.pop("intervention", "ablate"),
        vector=vector,
        prompt="p",
        completion="c",
        refusal_score=1.0,
        substring_refusal=substring,
        judged=judged,
        label=label,
        **kw,
    )


def test_summarize_groups_conditions_and_pools_random_seeds():
    rows = [
        _row("r", "compliance", judged=True),
        _row("r", "incoherent", judged=True),
        _row("random_0", "refusal", judged=True, substring=True),
        _row("random_1", "refusal", judged=True, substring=True),
    ]
    s = summarize(rows)
    assert s["harmful/ablate/r"]["compliance"]["rate"] == 0.5
    assert s["harmful/ablate/r"]["incoherent"]["k"] == 1  # incoherent is not compliance
    assert s["harmful/ablate/random*"]["refusal"]["n"] == 2
    assert s["harmful/ablate/random*"]["substring_refusal"]["rate"] == 1.0


def test_summarize_emits_judge_keys_whenever_a_judge_ran_even_if_all_failed():
    rows = [_row("r", None, judged=True), _row("r", None, judged=True)]
    s = summarize(rows)["harmful/ablate/r"]
    assert s["judge_missing"] == 2
    assert s["refusal"]["rate"] == 0.0  # failures are not passes
    assert "refusal" not in summarize([_row("r")])["harmful/ablate/r"]  # no judge -> no keys


def test_generation_row_omits_fields_that_never_applied():
    unjudged = _row("r").to_dict()
    assert "label" not in unjudged and "alpha" not in unjudged
    judged = _row("r", "refusal", judged=True, alpha=1.0).to_dict()
    assert judged["label"] == "refusal" and judged["alpha"] == 1.0
    assert "judge_error" not in judged and "judged" not in judged


def test_row_key_matches_interp_label_key_fields():
    assert _row("r", id="harmful_test:3").key == "harmful_test:3|ablate|r"


def test_is_pooled_only_matches_the_pooled_random_view():
    assert is_pooled(condition_key("harmful", "ablate", "random*"))
    assert not is_pooled(condition_key("harmful", "ablate", "random_0"))


# ---- stage configuration ----------------------------------------------------------------


def test_stage_parse_is_order_independent_and_rejects_typos():
    assert Stage.parse_all(["eval", "baseline"]) == [Stage.BASELINE, Stage.EVAL]
    with pytest.raises(ValueError, match="unknown stages"):
        Stage.parse_all(["baseline", "sweepp"])


# ---- schema stability -------------------------------------------------------------------


@pytest.mark.skipif(not COMMITTED_SUMMARY.exists(), reason="no committed results")
def test_summarize_still_matches_the_committed_summary_schema():
    """The published results and plot_layers.py read these keys; a rename breaks them."""
    committed = json.loads(COMMITTED_SUMMARY.read_text())["eval"]["conditions"]
    rows = [
        _row("r", "refusal", judged=True, prompt_set="harmful", intervention="ablate"),
        _row("random_0", "refusal", judged=True, prompt_set="harmful", intervention="ablate"),
    ]
    produced = summarize(rows)
    assert set(produced) <= set(committed), "produced a condition key the results don't have"
    for key in produced:
        assert set(produced[key]) == set(committed[key]), f"field set changed for {key}"
        assert set(produced[key]["refusal"]) == set(committed[key]["refusal"])
