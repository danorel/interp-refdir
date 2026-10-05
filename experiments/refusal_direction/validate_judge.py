"""Judge vs human agreement on a run's generations.

    # 1. label blind (condition fields hidden, rows shuffled); ~60 rows is enough for kappa
    uv run interp-label label outputs/refusal_direction/<run>/eval_test.jsonl \
        --question "Refusal, compliance, or incoherent?" \
        --choices r=refusal,c=compliance,i=incoherent --key id,intervention,vector
    # 2. compare
    uv run python experiments/refusal_direction/validate_judge.py \
        outputs/refusal_direction/<run>/eval_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.refusal_direction.schema import KEY_FIELDS
from interptemp.judges.metrics import agreement_report
from interptemp.label import load_labels, row_key
from interptemp.store import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("generations", type=Path)
    ap.add_argument("--labels", type=Path, help="default: <generations dir>/labels.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.generations)
    human = load_labels(args.labels or args.generations.with_name("labels.jsonl"))
    pairs = [(r, human[k]["label"]) for r in rows if (k := row_key(r, KEY_FIELDS)) in human]
    if not pairs:
        raise SystemExit("no labeled rows match; check --key fields used when labeling")
    if "label" not in pairs[0][0]:
        raise SystemExit("generations have no judge labels (run with a judge configured)")

    report = agreement_report([r["label"] for r, _ in pairs], [h for _, h in pairs])
    print(json.dumps(report, indent=2, default=str))
    for r, h in pairs:
        if r["label"] != h:
            print(f"\n--- judge={r['label']} human={h} [{r['intervention']}/{r['vector']}]")
            print(f"prompt: {r['prompt'][:200]}\nresponse: {r['completion'][:300]}")
            print(f"judge reasoning: {r.get('judge_reasoning', '')}")


if __name__ == "__main__":
    main()
