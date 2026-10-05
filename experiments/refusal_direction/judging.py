"""Turning `rubric.md` into a judge prompt, and applying a judge to generation rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from experiments.refusal_direction.schema import GenerationRow
from interptemp.judges import JudgeInput
from interptemp.judges.base import Judge

RUBRIC_PATH = Path(__file__).with_name("rubric.md")
# Everything after this marker is calibration material for human labelers only: showing the
# judge real examples from the data it will label would inflate agreement.
HUMAN_ONLY_MARKER = "<!-- human-only"


def load_judge_rubric(path: Path = RUBRIC_PATH) -> str:
    """The judge-visible part of the rubric, as an `LLMJudge` `str.format` template."""
    rules = path.read_text().split(HUMAN_ONLY_MARKER)[0].strip()
    rules = rules.replace("{", "{{").replace("}", "}}")  # rubric text is literal, not a template
    return f"{rules}\n\n---\n\nNow label this item.\n\nPrompt: {{prompt}}\n\nResponse: {{response}}"


def apply_judge(judge: Judge, rows: Sequence[GenerationRow]) -> Counter[str]:
    """Label `rows` in place; return a counter of per-row failures (empty if all succeeded).

    Raises if every row failed: missing labels count as "not refusal", so a judge that is
    down end-to-end would silently zero every rate and let later stages tune on noise.
    """
    results = judge.judge([JudgeInput(r.prompt, r.completion) for r in rows])
    for row, result in zip(rows, results, strict=True):
        row.judged = True
        row.label, row.judge_reasoning = result.label, result.reasoning
        if result.label is None:
            row.judge_error = result.meta.get("error", "unparseable output")

    errors = Counter(r.judge_error for r in rows if r.judge_error is not None)
    if sum(errors.values()) == len(rows):
        raise RuntimeError(f"judge failed on all {len(rows)} rows: {errors.most_common(3)}")
    return errors
