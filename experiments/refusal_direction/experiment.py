"""Refusal direction (Arditi et al. 2024): mean-diff direction, ablation + addition, controls.

Stages (params.stages), run in order in one run dir:
  baseline  val, no intervention: does the model refuse harmful / comply with harmless?
  sweep     every (layer, pos) candidate on val via a logit refusal score (no generation);
            pick the one that best bypasses refusal under ablation, subject to filters
  alpha     addition scale on harmless val (generation + judge)
  eval      test: baseline vs r vs norm-matched random vs contrast-control directions,
            ablation on harmful (necessity) and addition on harmless (sufficiency)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from interptemp.directions import mean_diff, random_like, unit
from interptemp.experiment import Experiment
from interptemp.interventions import AddVector, DirectionalAblation, Intervention
from interptemp.judges import JudgeInput
from interptemp.judges.substring import SubstringJudge
from interptemp.sites import Site
from interptemp.tasks import JsonlTask

HERE = Path(__file__).parent
RUBRIC_PATH = HERE / "rubric.md"
HUMAN_ONLY_MARKER = "<!-- human-only"
LABELS = ["refusal", "compliance", "incoherent"]
SETS = [
    "harmful_train", "harmless_train", "harmful_val", "harmless_val",
    "harmful_test", "harmless_test", "contrast_question", "contrast_imperative",
]  # fmt: skip


def load_judge_rubric(path: Path = RUBRIC_PATH) -> str:
    """Judge part of rubric.md + the item to label, as a LLMJudge `str.format` template."""
    rules = path.read_text().split(HUMAN_ONLY_MARKER)[0].strip()
    rules = rules.replace("{", "{{").replace("}", "}}")
    return f"{rules}\n\n---\n\nNow label this item.\n\nPrompt: {{prompt}}\n\nResponse: {{response}}"


# ---- pure metrics (unit-tested) ---------------------------------------------------------


def refusal_score(logits: torch.Tensor, token_ids: list[int], eps: float = 1e-8) -> torch.Tensor:
    """Log-odds that the next token is a refusal opener: [..., V] -> [...]."""
    p = logits.float().softmax(-1)[..., token_ids].sum(-1)
    return torch.log(p + eps) - torch.log(1 - p + eps)


def kl_div(ref_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(ref || other) over the vocab: [..., V] -> [...]."""
    ref, other = ref_logits.float().log_softmax(-1), logits.float().log_softmax(-1)
    return (ref.exp() * (ref - other)).sum(-1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def select_direction(rows: list[dict[str, Any]], max_kl: float) -> tuple[dict[str, Any], bool]:
    """Arditi et al. selection: min bypass score among candidates that (a) induce refusal on
    harmless when added and (b) barely change harmless outputs when ablated.

    Returns (row, filters_passed). If nothing induces refusal, falls back to the KL filter
    alone; never drops it: a direction that breaks the model also "bypasses refusal".
    """
    low_kl = [r for r in rows if r["kl"] < max_kl]
    if not low_kl:
        raise ValueError(f"no candidate has KL < {max_kl}: every ablation breaks the model")
    ok = [r for r in low_kl if r["induce_score"] > 0]
    best = min(ok or low_kl, key=lambda r: r["bypass_score"])
    return best, bool(ok)


def rate(labels: list[Any], target: Any) -> dict[str, Any]:
    """k/n with Wilson CI. Missing (None) labels stay in n: a failure to judge isn't a pass."""
    n, k = len(labels), sum(lab == target for lab in labels)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else math.nan, "ci95": [lo, hi]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per condition (prompt_set/intervention/vector): judge-label rates, substring refusal
    rate and mean logit refusal score. Random seeds are also pooled as `random*`."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = f"{r['prompt_set']}/{r['intervention']}/{r['vector']}"
        groups[key].append(r)
        if r["vector"].startswith("random_"):
            groups[f"{r['prompt_set']}/{r['intervention']}/random*"].append(r)
    out: dict[str, Any] = {}
    for key, rs in sorted(groups.items()):
        s: dict[str, Any] = {
            "substring_refusal": rate([r["substring_refusal"] for r in rs], True),
            "mean_refusal_score": sum(r["refusal_score"] for r in rs) / len(rs),
        }
        if "label" in rs[0]:
            labels = [r["label"] for r in rs]
            s |= {lab: rate(labels, lab) for lab in LABELS}
            s["judge_missing"] = sum(lab is None for lab in labels)
        out[key] = s
    return out


# ---- experiment -------------------------------------------------------------------------


class RefusalDirectionExperiment(Experiment):
    def judge_kwargs(self) -> dict[str, Any]:
        return {"rubric": load_judge_rubric(), "labels": LABELS}

    def run(self) -> dict[str, Any]:
        p, m = self.params, self.imodel
        self.data = {
            name: JsonlTask(f"{p['data_dir']}/{name}.jsonl").load()
            for name in SETS
            if Path(f"{p['data_dir']}/{name}.jsonl").exists()
        }
        self.prompts = {
            k: [m.format_chat(ex.messages) for ex in exs] for k, exs in self.data.items()
        }
        self.refusal_ids = self._refusal_token_ids(p["refusal_tokens"])
        self.positions: list[int] = list(p["positions"])
        self._log_positions()

        summary: dict[str, Any] = {"model": self.cfg.model.name, "num_layers": m.num_layers}
        stages = p["stages"]
        if "baseline" in stages:
            summary["baseline"] = self._baseline()

        dirs = self._directions()  # [L, P, d]
        layer, pos = p["layer"], p["pos"]
        if "sweep" in stages:
            sel = self._sweep(dirs)
            summary["sweep"] = sel
            layer, pos = sel["layer"], sel["pos"]
        if layer is None or pos is None:
            raise ValueError("set params.layer and params.pos, or run the `sweep` stage")
        r = dirs[layer, self.positions.index(pos)]
        self.save_tensor("direction.pt", r)
        summary["direction"] = {"layer": layer, "pos": pos, "norm": r.norm().item()}

        alpha = p["alpha"]
        if "alpha" in stages:
            summary["alpha"] = self._alpha_sweep(r, layer)
            alpha = summary["alpha"]["chosen"]
        summary["direction"]["alpha"] = alpha

        if "eval" in stages:
            summary["eval"] = self._eval(r, layer, pos, alpha)
        return summary

    # ---- helpers ------------------------------------------------------------------------

    def _refusal_token_ids(self, tokens: list[str]) -> list[int]:
        ids = []
        for t in tokens:
            enc = self.imodel.tokenizer(t, add_special_tokens=False)["input_ids"]
            if len(enc) != 1:
                raise ValueError(f"refusal token {t!r} is not a single token: {enc}")
            ids.append(enc[0])
        return ids

    def _log_positions(self) -> None:
        """The candidate positions are template-specific: log what they are on this model."""
        tok = self.imodel.tokenizer
        ids = tok(self.prompts["harmful_train"][0], add_special_tokens=False)["input_ids"]
        decoded = {pos: tok.decode([ids[pos]]) for pos in self.positions}
        self.log.info(f"candidate positions: {decoded}")

    def _site(self, layer: int) -> Site:
        return Site(self.params["site_kind"], layer)

    def _ablate(self, r: torch.Tensor) -> DirectionalAblation:
        return DirectionalAblation.everywhere(r, self.imodel.num_layers)

    def _add(self, r: torch.Tensor, layer: int, alpha: float) -> AddVector:
        # All positions, incl. generated tokens (interventions run at every decode step).
        return AddVector(r, [self._site(layer)], scale=alpha)

    def _last_logits(
        self, prompts: list[str], interventions: list[Intervention] | tuple = ()
    ) -> torch.Tensor:
        lg = self.imodel.logits(
            prompts, interventions, positions=[-1], batch_size=self.params["batch_size"]
        )
        return lg[:, 0]

    def _judge(self, rows: list[dict[str, Any]]) -> None:
        if self.cfg.judge is None:
            return
        results = self.judge.judge([JudgeInput(r["prompt"], r["completion"]) for r in rows])
        for r, j in zip(rows, results, strict=True):
            r["label"], r["judge_reasoning"] = j.label, j.reasoning
            if j.label is None:
                r["judge_error"] = j.meta.get("error", "unparseable output")
        errors = Counter(r["judge_error"] for r in rows if "judge_error" in r)
        if not errors:
            return
        n_missing = sum(errors.values())
        # Missing labels count as "not refusal": an all-failed judge silently zeroes every
        # rate (and the alpha stage would then pick on noise), so stop the run instead.
        if n_missing == len(rows):
            raise RuntimeError(f"judge failed on all {len(rows)} rows: {errors.most_common(3)}")
        self.log.warning(f"judge missing {n_missing}/{len(rows)} labels: {errors.most_common(3)}")

    def _generate(
        self,
        set_name: str,
        interventions: list[Intervention],
        intervention: str,
        vector: str,
        **extra: Any,
    ) -> list[dict[str, Any]]:
        """Generations + logit refusal score + substring refusal for one condition."""
        prompts = self.prompts[set_name]
        outs = self.imodel.generate(prompts, self.cfg.generation, interventions=interventions)
        scores = refusal_score(self._last_logits(prompts, interventions), self.refusal_ids)
        exs = self.data[set_name]
        subs = SubstringJudge().judge(
            [JudgeInput(ex.user_text, o) for ex, o in zip(exs, outs, strict=True)]
        )
        return [
            {
                "id": ex.id,
                "prompt_set": set_name.split("_")[0],
                "intervention": intervention,
                "vector": vector,
                **extra,
                "prompt": ex.user_text,
                "completion": o,
                "refusal_score": s.item(),
                "substring_refusal": sj.label == "yes",
            }
            for ex, o, s, sj in zip(exs, outs, scores, subs, strict=True)
        ]

    # ---- stages -------------------------------------------------------------------------

    def _baseline(self) -> dict[str, Any]:
        rows = []
        for set_name in ("harmful_val", "harmless_val"):
            rows += self._generate(set_name, [], "none", "-")
        self._judge(rows)
        self.save_jsonl("baseline_val.jsonl", rows)
        summ = summarize(rows)

        key = "refusal" if self.cfg.judge is not None else "substring_refusal"
        harmful = summ["harmful/none/-"][key]["rate"]
        if harmful < self.params["min_baseline_refusal"]:
            self.log.warning(
                f"baseline harmful refusal {harmful:.2f} < {self.params['min_baseline_refusal']}: "
                "too little refusal for ablation results to mean much"
            )
        return summ

    def _directions(self) -> torch.Tensor:
        m = self.imodel
        sites = [self._site(layer) for layer in range(m.num_layers)]
        acts = {
            k: m.activations(
                self.prompts[k],
                sites,
                positions=self.positions,
                batch_size=self.params["batch_size"],
            )
            for k in ("harmful_train", "harmless_train")
        }
        dirs = torch.stack(
            [mean_diff(acts["harmful_train"][s], acts["harmless_train"][s]) for s in sites]
        )
        self.save_tensor("directions.pt", dirs)
        return dirs

    def _sweep(self, dirs: torch.Tensor) -> dict[str, Any]:
        m, cfg = self.imodel, self.params["sweep"]
        harmful, harmless = self.prompts["harmful_val"], self.prompts["harmless_val"]
        clean_harmful = refusal_score(self._last_logits(harmful), self.refusal_ids).mean().item()
        clean_harmless_logits = self._last_logits(harmless)
        clean_harmless = refusal_score(clean_harmless_logits, self.refusal_ids).mean().item()
        self.log.info(
            f"clean refusal score: harmful={clean_harmful:.2f} harmless={clean_harmless:.2f}"
        )

        n_layers = math.floor(cfg["max_layer_frac"] * m.num_layers)
        rows = []
        for layer in range(n_layers):
            for pi, pos in enumerate(self.positions):
                r = dirs[layer, pi]
                # Layer-0 input at template positions is the same token embedding for every
                # prompt, so the mean-diff is exactly zero: no candidate there.
                if r.norm() <= 1e-6:
                    self.log.info(f"skip layer {layer} pos {pos}: zero-norm direction")
                    continue
                abl = self._last_logits(harmful + harmless, [self._ablate(r)])
                abl_harmful, abl_harmless = abl[: len(harmful)], abl[len(harmful) :]
                add = self._last_logits(harmless, [self._add(r, layer, 1.0)])
                rows.append(
                    {
                        "layer": layer,
                        "pos": pos,
                        "layer_frac": layer / m.num_layers,
                        "norm": r.norm().item(),
                        "bypass_score": refusal_score(abl_harmful, self.refusal_ids).mean().item(),
                        "induce_score": refusal_score(add, self.refusal_ids).mean().item(),
                        "kl": kl_div(clean_harmless_logits, abl_harmless).mean().item(),
                    }
                )
            if rows:
                best = min(rows, key=lambda r: r["bypass_score"])
                self.log.info(f"sweep layer {layer}/{n_layers - 1} done; best so far {best}")
        self.save_jsonl("sweep.jsonl", rows)

        sel, passed = select_direction(rows, cfg["max_kl"])
        if not passed:
            self.log.warning("no candidate induces refusal; using min bypass among low-KL ones")
        return {
            **sel,
            "filters_passed": passed,
            "clean_refusal_score": {"harmful": clean_harmful, "harmless": clean_harmless},
        }

    def _alpha_sweep(self, r: torch.Tensor, layer: int) -> dict[str, Any]:
        rows = []
        for alpha in self.params["alphas"]:
            rows += self._generate(
                "harmless_val", [self._add(r, layer, alpha)], "add", "r", alpha=alpha
            )
        self._judge(rows)
        self.save_jsonl("alpha_val.jsonl", rows)

        per_alpha = {
            a: summarize([r for r in rows if r["alpha"] == a])["harmless/add/r"]
            for a in self.params["alphas"]
        }
        chosen = self.params["alpha"]
        if self.cfg.judge is None:
            self.log.warning(
                "alpha stage without a judge: keeping params.alpha (no incoherence check)"
            )
        else:
            ok = [
                a for a, s in per_alpha.items()
                if s["incoherent"]["rate"] <= self.params["max_incoherent"]
            ]  # fmt: skip
            if ok:  # highest refusal; ties -> smallest alpha (least off-distribution)
                chosen = max(ok, key=lambda a: (per_alpha[a]["refusal"]["rate"], -a))
            else:
                self.log.warning("every alpha exceeds max_incoherent; keeping params.alpha")
        return {"chosen": chosen, "per_alpha": per_alpha}

    def _eval(self, r: torch.Tensor, layer: int, pos: int, alpha: float) -> dict[str, Any]:
        m = self.imodel
        site = self._site(layer)
        contrast = {
            k: m.activations(self.prompts[k], [site], positions=[pos])[site][:, 0]
            for k in ("contrast_question", "contrast_imperative")
        }
        # Same construction as r, from a contrast with no refusal; scaled to |r| for addition.
        c = (
            unit(mean_diff(contrast["contrast_question"], contrast["contrast_imperative"]))
            * r.norm()
        )
        vectors = {"r": r, "contrast": c} | {
            f"random_{s}": random_like(r, seed=s) for s in range(self.params["n_random"])
        }
        self.save_tensor("eval_vectors.pt", torch.stack(list(vectors.values())))

        diag = self._diagnostics(vectors, site, pos)
        self.log.info(f"diagnostics: {diag}")

        rows = self._generate("harmful_test", [], "none", "-")
        rows += self._generate("harmless_test", [], "none", "-")
        for name, v in vectors.items():
            rows += self._generate("harmful_test", [self._ablate(v)], "ablate", name)
            rows += self._generate("harmless_test", [self._add(v, layer, alpha)], "add", name)
        # Side effects: ablating r should leave harmless behaviour intact.
        rows += self._generate("harmless_test", [self._ablate(r)], "ablate", "r")
        self._judge(rows)
        self.save_jsonl("eval_test.jsonl", rows)
        return {"diagnostics": diag, "conditions": summarize(rows)}

    def _diagnostics(
        self, vectors: dict[str, torch.Tensor], site: Site, pos: int
    ) -> dict[str, Any]:
        """Share of ||h||² each vector's ablation removes at the chosen site (test prompts),
        and cosine to r: a control that removes ~nothing is a trivially passed control."""
        m = self.imodel
        out: dict[str, Any] = {}
        for k in ("harmful_test", "harmless_test"):
            h = m.activations(self.prompts[k], [site], positions=[pos])[site][:, 0].float()
            out[f"frac_removed/{k}"] = {
                name: ((h @ unit(v.float())) ** 2 / h.pow(2).sum(-1)).mean().item()
                for name, v in vectors.items()
            }
        out["cos_to_r"] = {
            name: torch.nn.functional.cosine_similarity(
                v.float(), vectors["r"].float(), dim=0
            ).item()
            for name, v in vectors.items()
        }
        return out
