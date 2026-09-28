"""Judge an existing run's generations and compare with human labels (labels.jsonl).

    uv run python experiments/example_steering/judge_existing.py outputs/example_steering/<run>

Writes <run>/judge_labels.jsonl; prints agreement (kappa), per-condition yes-rates for
judge vs human, and every disagreement with the judge's reasoning.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `experiments.*`

from experiments.example_steering.experiment import load_judge_rubric
from interptemp.config import JudgeConfig
from interptemp.judges import JudgeInput
from interptemp.judges.llm import LLMJudge
from interptemp.judges.metrics import agreement_report
from interptemp.label import row_key
from interptemp.store import read_jsonl, write_jsonl

KEY_FIELDS = ["id", "vector", "scale"]


def yes_rates(rows: list[dict], labels: dict[str, str | None]) -> dict[str, str]:
    groups: dict[str, list[str | None]] = defaultdict(list)
    for r in rows:
        vec = "random" if r["vector"].startswith("random") else r["vector"]
        groups[f"{vec}@{r['scale']}"].append(labels.get(row_key(r, KEY_FIELDS)))
    return {g: f"{sum(v == 'yes' for v in vs)}/{len(vs)}" for g, vs in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default=JudgeConfig().model)
    # OpenRouter reserves credits for max_tokens x in-flight requests, but reasoning needs
    # room: Anthropic silently drops thinking below ~4096 (see LLMJudge warning).
    ap.add_argument("--max-tokens", type=int, default=JudgeConfig().max_tokens)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--reasoning", default="low", help="effort level, or 'none' to disable")
    args = ap.parse_args()
    load_dotenv()

    cfg = JudgeConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        max_concurrency=args.concurrency,
        reasoning_effort=None if args.reasoning == "none" else args.reasoning,
    )
    rows = read_jsonl(args.run_dir / "generations.jsonl")
    judge = LLMJudge(cfg, rubric=load_judge_rubric(), labels=["yes", "no"])
    results = judge.judge([JudgeInput(r["prompt"], r["completion"]) for r in rows])

    out = [
        {
            "key": row_key(r, KEY_FIELDS),
            **{f: r[f] for f in KEY_FIELDS},
            "label": j.label,
            "reasoning": j.reasoning,
            **({"error": j.meta["error"]} if "error" in j.meta else {}),
        }
        for r, j in zip(rows, results, strict=True)
    ]
    write_jsonl(args.run_dir / "judge_labels.jsonl", out)

    human = {h["key"]: h["label"] for h in read_jsonl(args.run_dir / "labels.jsonl")}
    judged = {o["key"]: o["label"] for o in out}
    keys = [o["key"] for o in out]
    report = agreement_report([judged[k] for k in keys], [human.get(k) for k in keys])
    print(json.dumps(report, indent=2))

    print(f"\n{'condition':<16} {'human':>6} {'judge':>6}")
    h_rates, j_rates = yes_rates(rows, human), yes_rates(rows, judged)
    for g in h_rates:
        print(f"{g:<16} {h_rates[g]:>6} {j_rates[g]:>6}")

    print("\nDisagreements:")
    for r, o in zip(rows, out, strict=True):
        if o["label"] != human.get(o["key"]):
            print(f"\n[{o['key']}] human={human.get(o['key'])} judge={o['label']}")
            print(f"  completion: {r['completion']!r}")
            print(f"  judge: {o['reasoning'] or o.get('error')}")


if __name__ == "__main__":
    main()
