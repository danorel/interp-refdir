"""Blind manual labeling of JSONL rows in the terminal.

    uv run interp-label label outputs/<run>/generations.jsonl --question "Positive tone?"
    uv run interp-label summary outputs/<run>/labels.jsonl --by vector,scale

- Blind: only `--show` fields are displayed (condition fields like vector/scale are hidden)
  and rows are shuffled, so expectations can't leak into labels.
- Source data is never modified: labels go to a separate labels.jsonl, rewritten atomically
  after every keypress, so quitting (or crashing) loses nothing and re-running resumes.
- Each label row stores the row's key fields (e.g. id/vector/scale), so labels are
  self-contained for summaries and joinable with judge outputs (kappa).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from interptemp.store import read_jsonl

RESERVED = {"s": "skip", "u": "undo", "q": "quit"}
# Fields never used as key: model outputs / judge results rather than row identity.
NON_KEY_FIELDS = {"label", "judge_reasoning", "completion", "response"}


def default_key_fields(rows: Sequence[dict[str, Any]], show: Sequence[str]) -> list[str]:
    """Scalar fields that aren't displayed or outputs, e.g. id, vector, scale."""
    first = rows[0]
    return [
        k
        for k, v in first.items()
        if k not in show and k not in NON_KEY_FIELDS and isinstance(v, str | int | float | bool)
    ]


def row_key(row: dict[str, Any], fields: Sequence[str]) -> str:
    return "|".join(str(row[f]) for f in fields)


def parse_choices(spec: str) -> dict[str, str]:
    """'y=yes,n=no' -> {'y': 'yes', 'n': 'no'}."""
    choices = dict(item.split("=", 1) for item in spec.split(","))
    if clash := set(choices) & set(RESERVED):
        raise ValueError(f"keys {sorted(clash)} are reserved for {RESERVED}")
    if any(len(k) != 1 for k in choices):
        raise ValueError(f"choice keys must be single characters, got {list(choices)}")
    return choices


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}


def save_labels(path: Path, labels: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in labels.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: never leaves a half-written labels file


def read_key() -> str:
    """Single keypress without Enter on a TTY; falls back to a line of stdin (pipes, tests)."""
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()[:1]
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":  # Ctrl-C in raw mode
        raise KeyboardInterrupt
    return ch


def _render(row: dict[str, Any], show: Sequence[str]) -> str:
    width = min(shutil.get_terminal_size().columns, 110) - 4
    parts = []
    for f in show:
        body = "\n".join(
            textwrap.fill(line, width, initial_indent="  ", subsequent_indent="  ") or "  "
            for line in str(row.get(f, "")).splitlines()
        )
        parts.append(f"{f.upper()}:\n{body}")
    return "\n\n".join(parts)


def label_loop(
    rows: Sequence[dict[str, Any]],
    out: Path,
    question: str,
    choices: dict[str, str],
    show: Sequence[str],
    key_fields: Sequence[str],
    seed: int = 0,
    get_key: Callable[[], str] = read_key,
    echo: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    labels = load_labels(out)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    queue = [i for i in order if row_key(rows[i], key_fields) not in labels]
    history: list[int] = []  # row indices labeled this session, for undo
    prompt = "  ".join(f"[{k}] {v}" for k, v in {**choices, **RESERVED}.items())

    while queue:
        i = queue[0]
        key = row_key(rows[i], key_fields)
        echo("\n" + "=" * 60)
        echo(f"[{len(labels) + 1}/{len(rows)}]\n")
        echo(_render(rows[i], show))
        echo(f"\n{question}  {prompt}")
        ch = get_key()
        if ch in choices:
            labels[key] = {
                "key": key,
                "label": choices[ch],
                **{f: rows[i][f] for f in key_fields},
            }
            save_labels(out, labels)
            history.append(queue.pop(0))
        elif ch == "s":
            queue.append(queue.pop(0))  # revisit at the end
        elif ch == "u":
            if not history:
                echo("(nothing to undo)")
                continue
            prev = history.pop()
            del labels[row_key(rows[prev], key_fields)]
            save_labels(out, labels)
            queue.insert(0, prev)
        elif ch == "q" or ch == "":
            break
        else:
            echo(f"(unknown key {ch!r})")

    echo(f"\n{len(labels)}/{len(rows)} labeled -> {out}")
    return labels


def summarize(labels: Sequence[dict[str, Any]], by: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-group label counts and rates, e.g. by=['vector', 'scale']."""
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for r in labels:
        groups["@".join(str(r[f]) for f in by) if by else "all"][r["label"]] += 1
    return {
        g: {"n": sum(c.values()), **{lab: round(n / sum(c.values()), 3) for lab, n in c.items()}}
        for g, c in sorted(groups.items())
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Blind manual labeling of JSONL rows.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lab = sub.add_parser("label", help="label rows interactively")
    lab.add_argument("data", type=Path, help="JSONL with rows to label")
    lab.add_argument("--out", type=Path, help="labels file (default: <data dir>/labels.jsonl)")
    lab.add_argument("--question", default="Label?")
    lab.add_argument("--choices", default="y=yes,n=no", help="key=label pairs")
    lab.add_argument("--show", default="prompt,completion", help="fields to display")
    lab.add_argument("--key", help="fields identifying a row (default: non-shown scalars)")
    lab.add_argument("--seed", type=int, default=0, help="shuffle seed")

    summ = sub.add_parser("summary", help="label rates per group")
    summ.add_argument("labels", type=Path)
    summ.add_argument("--by", default="", help="comma-separated fields to group by")

    args = ap.parse_args(argv)
    if args.cmd == "summary":
        by = [f for f in args.by.split(",") if f]
        for group, stats in summarize(read_jsonl(args.labels), by).items():
            print(f"{group:<30} {stats}")
        return

    rows = read_jsonl(args.data)
    if not rows:
        sys.exit(f"{args.data} is empty")
    show = args.show.split(",")
    key_fields = args.key.split(",") if args.key else default_key_fields(rows, show)
    keys = [row_key(r, key_fields) for r in rows]
    if len(set(keys)) != len(keys):
        sys.exit(f"key fields {key_fields} don't uniquely identify rows; pass --key")
    print(f"key fields: {key_fields} (hidden: labeling is blind)")
    label_loop(
        rows,
        out=args.out or args.data.parent / "labels.jsonl",
        question=args.question,
        choices=parse_choices(args.choices),
        show=show,
        key_fields=key_fields,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
