"""Example: contrastive (mean-diff) direction -> steering, vs norm-matched random controls.

Copy this directory to start a new "find direction / intervene / measure" project.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from interptemp.directions import mean_diff, project, random_like
from interptemp.experiment import Experiment
from interptemp.interventions import AddVector
from interptemp.judges import JudgeInput
from interptemp.sites import Site
from interptemp.tasks import build_task

RUBRIC_PATH = Path(__file__).with_name("rubric.md")
# Everything after this marker in rubric.md is for human labelers only (real calibration
# examples from the data the judge will label — showing them would inflate agreement).
HUMAN_ONLY_MARKER = "<!-- human-only"


def load_judge_rubric(path: Path = RUBRIC_PATH) -> str:
    """Judge part of rubric.md + the item to label, as a LLMJudge `str.format` template."""
    rules = path.read_text().split(HUMAN_ONLY_MARKER)[0].strip()
    rules = rules.replace("{", "{{").replace("}", "}}")  # rubric text is literal, not a template
    return f"{rules}\n\n---\n\nNow label this item.\n\nPrompt: {{prompt}}\n\nResponse: {{response}}"


class SteeringExperiment(Experiment):
    def judge_kwargs(self) -> dict[str, Any]:
        return {"rubric": load_judge_rubric(), "labels": ["yes", "no"]}

    def run(self) -> dict[str, Any]:
        p, m = self.params, self.imodel
        data = {name: build_task(spec).load() for name, spec in p["tasks"].items()}
        prompts = {name: [m.format_chat(ex.messages) for ex in exs] for name, exs in data.items()}

        site = Site("resid_post", round(p["layer_frac"] * (m.num_layers - 1)))
        acts = {
            name: m.activations(prompts[name], [site], positions=p["positions"])[site]
            for name in ("positive", "negative")
        }
        direction = mean_diff(acts["positive"], acts["negative"]).mean(0)  # avg over positions
        self.save_tensor("direction.pt", direction)

        # Diagnostic: does the direction separate the (train) sets at all?
        sep = {k: project(v, direction).mean().item() for k, v in acts.items()}
        self.log.info(f"{site}: |dir|={direction.norm():.2f} mean proj {sep}")

        vectors = {"direction": direction} | {
            f"random_{s}": random_like(direction, seed=s) for s in range(p["n_random_controls"])
        }
        rows = []
        for vec_name, vec in vectors.items():
            for scale in p["scales"]:
                if scale == 0 and vec_name != "direction":
                    continue  # baseline is identical for every vector
                iv = AddVector(vec, [site], scale=scale)
                outs = m.generate(prompts["eval"], self.cfg.generation, interventions=[iv])
                rows += [
                    {
                        "id": ex.id,
                        "vector": vec_name,
                        "scale": scale,
                        "prompt": ex.user_text,
                        "completion": o,
                    }
                    for ex, o in zip(data["eval"], outs, strict=True)
                ]

        if self.cfg.judge is not None:
            results = self.judge.judge([JudgeInput(r["prompt"], r["completion"]) for r in rows])
            for r, j in zip(rows, results, strict=True):
                r["label"], r["judge_reasoning"] = j.label, j.reasoning
        self.save_jsonl("generations.jsonl", rows)
        return {
            "site": str(site),
            "direction_norm": direction.norm().item(),
            "proj": sep,
            **_rates(rows),
        }


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "label" not in rows[0]:
        return {}
    by: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by[f"{r['vector']}@{r['scale']}"].append(r["label"])
    return {
        "positive_rate": {k: sum(v == "yes" for v in vs) / len(vs) for k, vs in by.items()},
        "judge_missing": sum(r["label"] is None for r in rows),
    }
