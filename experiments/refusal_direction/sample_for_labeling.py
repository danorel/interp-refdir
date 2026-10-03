"""Pick a stratified sample of a run's generations for blind human labeling.

    uv run python experiments/refusal_direction/sample_for_labeling.py outputs/refusal_direction/<run>

Takes `per_condition` rows from each (prompt_set, intervention, vector) group — random seeds
pooled, so controls don't swamp the sample — plus `n_disagree` rows where substring refusal
and the logit proxy disagree, which is where the cheap metrics fail. Writes <run>/to_label.jsonl.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from interptemp.store import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--per-condition", type=int, default=6)
    ap.add_argument("--n-disagree", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.run / "eval_test.jsonl")
    rng = random.Random(args.seed)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        vec = "random" if r["vector"].startswith("random") else r["vector"]
        groups[r["prompt_set"], r["intervention"], vec].append(r)

    sample = [x for g in groups.values() for x in rng.sample(g, min(args.per_condition, len(g)))]
    picked = {id(x) for x in sample}
    disagree = [
        r
        for r in rows
        if (r["refusal_score"] > 0) != r["substring_refusal"] and id(r) not in picked
    ]
    sample += rng.sample(disagree, min(args.n_disagree, len(disagree)))

    out = write_jsonl(args.run / "to_label.jsonl", sample)
    print(f"{len(groups)} conditions -> {len(sample)} rows -> {out}")


if __name__ == "__main__":
    main()
