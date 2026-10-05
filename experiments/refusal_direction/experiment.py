"""Refusal direction (Arditi et al. 2024): mean-diff direction, ablation + addition, controls.

This module only orchestrates. The scoring lives in `metrics`, the records in `schema`, the
judge wiring in `judging`, and the direction/control construction in `vectors`.

Stages (`params.stages`), run in order into one run dir:
  baseline  val, no intervention: does the model refuse harmful / comply with harmless?
  sweep     score every (layer, position) candidate on val with the logit proxy — no
            generation — and pick one; see `metrics.select_direction`
  alpha     choose the addition scale on harmless val (generation + judge)
  eval      test: r vs norm-matched random vs contrast control, ablation on harmful
            (necessity) and addition on harmless (sufficiency)

Selection happens on val, reported numbers come from test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import torch

from experiments.refusal_direction import vectors as vec
from experiments.refusal_direction.judging import apply_judge, load_judge_rubric
from experiments.refusal_direction.metrics import (
    kl_div,
    refusal_score,
    select_direction,
    summarize,
)
from experiments.refusal_direction.schema import (
    LABELS,
    AlphaResult,
    GenerationRow,
    Selection,
    Stage,
    SweepCandidate,
    SweepResult,
    condition_key,
)
from interptemp.experiment import Experiment
from interptemp.interventions import AddVector, DirectionalAblation, Intervention
from interptemp.judges import JudgeInput
from interptemp.judges.substring import SubstringJudge
from interptemp.sites import Site
from interptemp.tasks import Example, JsonlTask

# Every split prepare_data.py writes. Contrast sets back the contrast control.
TRAIN_SETS = ("harmful_train", "harmless_train")
VAL_SETS = ("harmful_val", "harmless_val")
TEST_SETS = ("harmful_test", "harmless_test")
CONTRAST_SETS = ("contrast_question", "contrast_imperative")
ALL_SETS = (*TRAIN_SETS, *VAL_SETS, *TEST_SETS, *CONTRAST_SETS)


class RefusalDirectionExperiment(Experiment):
    # ---- configuration-derived state (built once, on first use) -------------------------

    def judge_kwargs(self) -> dict[str, Any]:
        return {"rubric": load_judge_rubric(), "labels": LABELS}

    @cached_property
    def data(self) -> dict[str, list[Example]]:
        data_dir = Path(self.params["data_dir"])
        missing = [name for name in ALL_SETS if not (data_dir / f"{name}.jsonl").exists()]
        if missing:
            raise FileNotFoundError(
                f"missing splits in {data_dir}: {missing} — run prepare_data.py first"
            )
        return {name: JsonlTask(str(data_dir / f"{name}.jsonl")).load() for name in ALL_SETS}

    @cached_property
    def prompts(self) -> dict[str, list[str]]:
        """Chat-formatted prompt strings per split — exactly what the model is fed."""
        return {
            name: [self.imodel.format_chat(ex.messages) for ex in examples]
            for name, examples in self.data.items()
        }

    @cached_property
    def positions(self) -> list[int]:
        return list(self.params["positions"])

    @cached_property
    def refusal_token_ids(self) -> list[int]:
        """Single-token refusal openers; the logit proxy is their summed probability."""
        ids = []
        for token in self.params["refusal_tokens"]:
            encoded = self.imodel.tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(encoded) != 1:
                raise ValueError(f"refusal token {token!r} is not a single token: {encoded}")
            ids.append(encoded[0])
        return ids

    @cached_property
    def direction_grid(self) -> torch.Tensor:
        """mean-diff direction per (layer, position): [num_layers, n_positions, d]."""
        sites = [self.site(layer) for layer in range(self.imodel.num_layers)]
        grid = vec.build_direction_grid(
            self.imodel,
            self.prompts["harmful_train"],
            self.prompts["harmless_train"],
            sites,
            self.positions,
            self.params["batch_size"],
        )
        self.save_tensor("directions.pt", grid)
        return grid

    # ---- intervention and model plumbing -------------------------------------------------

    def site(self, layer: int) -> Site:
        return Site(self.params["site_kind"], layer)

    def ablation(self, direction: torch.Tensor) -> DirectionalAblation:
        """Remove the direction from embed and every attn/mlp output, so no component can
        write it back into the residual stream at a later layer."""
        return DirectionalAblation.everywhere(direction, self.imodel.num_layers)

    def addition(self, direction: torch.Tensor, layer: int, alpha: float) -> AddVector:
        """Add at one layer, every position — including generated tokens, since interventions
        re-run at each decode step. Steering only the prompt washes out during generation."""
        return AddVector(direction, [self.site(layer)], scale=alpha)

    def next_token_logits(
        self, prompts: Sequence[str], interventions: Sequence[Intervention] = ()
    ) -> torch.Tensor:
        """Logits for the first generated token: [n_prompts, vocab]."""
        logits = self.imodel.logits(
            prompts, interventions, positions=[-1], batch_size=self.params["batch_size"]
        )
        return logits[:, 0]

    def log_candidate_positions(self) -> None:
        """Candidate positions are chat-template specific; log what they actually are here."""
        tokenizer = self.imodel.tokenizer
        ids = tokenizer(self.prompts["harmful_train"][0], add_special_tokens=False)["input_ids"]
        decoded = {pos: tokenizer.decode([ids[pos]]) for pos in self.positions}
        self.log.info(f"candidate positions: {decoded}")

    # ---- one condition -------------------------------------------------------------------

    def run_condition(
        self,
        split: str,
        interventions: Sequence[Intervention],
        *,
        intervention_name: str,
        vector_name: str,
        alpha: float | None = None,
    ) -> list[GenerationRow]:
        """Generate under one condition and score it with the two cheap metrics."""
        prompts, examples = self.prompts[split], self.data[split]
        completions = self.imodel.generate(
            prompts, self.cfg.generation, interventions=list(interventions)
        )
        scores = refusal_score(
            self.next_token_logits(prompts, interventions), self.refusal_token_ids
        )
        substring = SubstringJudge().judge(
            [JudgeInput(ex.user_text, c) for ex, c in zip(examples, completions, strict=True)]
        )
        return [
            GenerationRow(
                id=example.id,
                prompt_set=split.split("_")[0],
                intervention=intervention_name,
                vector=vector_name,
                alpha=alpha,
                prompt=example.user_text,
                completion=completion,
                refusal_score=score.item(),
                substring_refusal=marker.label == "yes",
            )
            for example, completion, score, marker in zip(
                examples, completions, scores, substring, strict=True
            )
        ]

    def judge_rows(self, rows: Sequence[GenerationRow]) -> None:
        """Label rows in place, if a judge is configured; warn about partial failures."""
        if self.cfg.judge is None:
            return
        errors = apply_judge(self.judge, rows)
        if errors:
            self.log.warning(
                f"judge missing {sum(errors.values())}/{len(rows)} labels: {errors.most_common(3)}"
            )

    def save_rows(self, name: str, rows: Sequence[GenerationRow]) -> None:
        self.save_jsonl(name, [row.to_dict() for row in rows])

    # ---- stages --------------------------------------------------------------------------

    @cached_property
    def stages(self) -> list[Stage]:
        return Stage.parse_all(self.params["stages"])

    def pinned_site(self) -> tuple[int, int] | None:
        """(layer, position) fixed in the config, letting a run skip the sweep."""
        layer, pos = self.params["layer"], self.params["pos"]
        return None if layer is None or pos is None else (layer, pos)

    def run(self) -> dict[str, Any]:
        """baseline and eval measure; sweep and alpha decide. `select` holds the decisions."""
        self.log_candidate_positions()
        report: dict[str, Any] = {
            "model": self.cfg.model.name,
            "num_layers": self.imodel.num_layers,
        }
        if Stage.BASELINE in self.stages:
            report["baseline"] = self.stage_baseline()

        selection = self.select(report)
        self.save_tensor("direction.pt", selection.direction)
        report["direction"] = selection.to_dict()

        if Stage.EVAL in self.stages:
            report["eval"] = self.stage_eval(selection)
        return report

    def select(self, report: dict[str, Any]) -> Selection:
        """Resolve (layer, position, alpha) by running the selection stages that are enabled.

        A stage that does not run falls back to the value pinned in the config, which is how
        a later run reuses an earlier sweep instead of repeating it. Each stage writes its own
        evidence into `report`.
        """
        site = self.pinned_site()
        if Stage.SWEEP in self.stages:
            sweep = self.stage_sweep()
            report["sweep"], site = sweep.to_dict(), (sweep.best.layer, sweep.best.pos)
        if site is None:
            raise ValueError("set params.layer and params.pos, or run the `sweep` stage")
        layer, pos = site
        direction = self.direction_grid[layer, self.positions.index(pos)]

        alpha = self.params["alpha"]
        if Stage.ALPHA in self.stages:
            chosen = self.stage_alpha(direction, layer)
            report["alpha"], alpha = chosen.to_dict(), chosen.chosen
        return Selection(layer=layer, pos=pos, alpha=alpha, direction=direction)

    def stage_baseline(self) -> dict[str, Any]:
        """No intervention on val: is there enough refusal for ablation to have a target?"""
        rows = [row for split in VAL_SETS for row in self.condition_none(split)]
        self.judge_rows(rows)
        self.save_rows("baseline_val.jsonl", rows)

        report = summarize(rows)
        metric = "refusal" if self.cfg.judge is not None else "substring_refusal"
        observed = report[condition_key("harmful", "none", "-")][metric]["rate"]
        floor = self.params["min_baseline_refusal"]
        if observed < floor:
            self.log.warning(
                f"baseline harmful refusal {observed:.2f} < {floor}: "
                "too little refusal for ablation results to mean much"
            )
        return report

    def condition_none(self, split: str) -> list[GenerationRow]:
        return self.run_condition(split, [], intervention_name="none", vector_name="-")

    def stage_sweep(self) -> SweepResult:
        """Score every (layer, position) candidate on val using the logit proxy only.

        Generating and judging all candidates would cost ~190k generations; this is one
        forward pass per prompt per candidate.
        """
        harmful, harmless = self.prompts["harmful_val"], self.prompts["harmless_val"]
        clean_harmless_logits = self.next_token_logits(harmless)
        clean = {
            "harmful": refusal_score(
                self.next_token_logits(harmful), self.refusal_token_ids
            ).mean().item(),
            "harmless": refusal_score(clean_harmless_logits, self.refusal_token_ids).mean().item(),
        }  # fmt: skip
        self.log.info(f"clean refusal score: {clean}")

        candidates = self.sweep_candidates(harmful, harmless, clean_harmless_logits)
        self.save_jsonl("sweep.jsonl", [c.to_dict() for c in candidates])

        best, induce_passed = select_direction(candidates, self.params["sweep"]["max_kl"])
        if not induce_passed:
            self.log.warning("no candidate induces refusal; using min bypass among low-KL ones")
        return SweepResult(
            best=best,
            induce_filter_passed=induce_passed,
            clean_refusal_score=clean,
            candidates=candidates,
        )

    def sweep_candidates(
        self,
        harmful: Sequence[str],
        harmless: Sequence[str],
        clean_harmless_logits: torch.Tensor,
    ) -> list[SweepCandidate]:
        """Score the (layer, position) grid. Late layers are skipped: they mostly write
        straight to the unembedding, where "removing refusal" says little about mediation."""
        num_layers = self.imodel.num_layers
        max_layer = math.floor(self.params["sweep"]["max_layer_frac"] * num_layers)
        candidates: list[SweepCandidate] = []
        for layer in range(max_layer):
            for index, pos in enumerate(self.positions):
                direction = self.direction_grid[layer, index]
                if vec.is_degenerate(direction):
                    self.log.info(f"skip layer {layer} pos {pos}: zero-norm direction")
                    continue
                candidates.append(
                    self.score_candidate(
                        direction, layer, pos, harmful, harmless, clean_harmless_logits
                    )
                )
            if candidates:
                best = min(candidates, key=lambda c: c.bypass_score)
                self.log.info(
                    f"sweep layer {layer}/{max_layer - 1} done; best so far "
                    f"layer {best.layer} pos {best.pos} bypass {best.bypass_score:.2f}"
                )
        return candidates

    def score_candidate(
        self,
        direction: torch.Tensor,
        layer: int,
        pos: int,
        harmful: Sequence[str],
        harmless: Sequence[str],
        clean_harmless_logits: torch.Tensor,
    ) -> SweepCandidate:
        # One batched pass for both splits, then split the rows back apart.
        ablated = self.next_token_logits([*harmful, *harmless], [self.ablation(direction)])
        ablated_harmful, ablated_harmless = ablated[: len(harmful)], ablated[len(harmful) :]
        added = self.next_token_logits(harmless, [self.addition(direction, layer, 1.0)])
        return SweepCandidate(
            layer=layer,
            pos=pos,
            layer_frac=layer / self.imodel.num_layers,
            norm=direction.norm().item(),
            bypass_score=refusal_score(ablated_harmful, self.refusal_token_ids).mean().item(),
            induce_score=refusal_score(added, self.refusal_token_ids).mean().item(),
            kl=kl_div(clean_harmless_logits, ablated_harmless).mean().item(),
        )

    def stage_alpha(self, direction: torch.Tensor, layer: int) -> AlphaResult:
        """Pick the addition scale on harmless val: most refusal, subject to staying coherent."""
        rows = [
            row
            for alpha in self.params["alphas"]
            for row in self.run_condition(
                "harmless_val",
                [self.addition(direction, layer, alpha)],
                intervention_name="add",
                vector_name="r",
                alpha=alpha,
            )
        ]
        self.judge_rows(rows)
        self.save_rows("alpha_val.jsonl", rows)

        key = condition_key("harmless", "add", "r")
        per_alpha = {
            alpha: summarize([r for r in rows if r.alpha == alpha])[key]
            for alpha in self.params["alphas"]
        }
        return AlphaResult(chosen=self.choose_alpha(per_alpha), per_alpha=per_alpha)

    def choose_alpha(self, per_alpha: dict[float, dict[str, Any]]) -> float:
        """Highest refusal among coherent scales; ties go to the smallest (least off-distribution).

        Without a judge there is no incoherence signal, and a large scale that destroys the
        model looks like a win — so fall back to the configured alpha instead of guessing.
        """
        if self.cfg.judge is None:
            self.log.warning("alpha stage without a judge: keeping params.alpha")
            return self.params["alpha"]
        coherent = [
            alpha
            for alpha, report in per_alpha.items()
            if report["incoherent"]["rate"] <= self.params["max_incoherent"]
        ]
        if not coherent:
            self.log.warning("every alpha exceeds max_incoherent; keeping params.alpha")
            return self.params["alpha"]
        return max(coherent, key=lambda a: (per_alpha[a]["refusal"]["rate"], -a))

    def stage_eval(self, selection: Selection) -> dict[str, Any]:
        """Test set: both causal tests for `r` and for every control, plus side effects."""
        direction, layer, pos = selection.direction, selection.layer, selection.pos
        site = self.site(layer)
        all_vectors = vec.control_vectors(direction, self.params["n_random"]) | {
            "contrast": vec.contrast_direction(
                self.imodel,
                self.prompts["contrast_question"],
                self.prompts["contrast_imperative"],
                site,
                pos,
                direction.norm().item(),
            )
        }
        self.save_tensor("eval_vectors.pt", torch.stack(list(all_vectors.values())))

        diagnostics = vec.ablation_diagnostics(
            self.imodel, all_vectors, {s: self.prompts[s] for s in TEST_SETS}, site, pos
        )
        self.log.info(f"diagnostics: {diagnostics}")

        rows = [row for split in TEST_SETS for row in self.condition_none(split)]
        for name, vector in all_vectors.items():
            # Necessity on harmful, sufficiency on harmless.
            rows += self.run_condition(
                "harmful_test",
                [self.ablation(vector)],
                intervention_name="ablate",
                vector_name=name,
            )
            rows += self.run_condition(
                "harmless_test",
                [self.addition(vector, layer, selection.alpha)],
                intervention_name="add",
                vector_name=name,
            )
        # Side effects: ablating r must leave harmless behaviour intact.
        rows += self.run_condition(
            "harmless_test", [self.ablation(direction)], intervention_name="ablate", vector_name="r"
        )
        self.judge_rows(rows)
        self.save_rows("eval_test.jsonl", rows)
        return {"diagnostics": diagnostics, "conditions": summarize(rows)}
