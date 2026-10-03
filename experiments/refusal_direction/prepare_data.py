"""Build the train/val/test JSONL splits from Arditi et al.'s published splits.

    uv run python experiments/refusal_direction/prepare_data.py

Sources: github.com/andyrdt/refusal_direction/tree/main/dataset/splits
  harmful: AdvBench/TDC/HarmBench (train, val), JailbreakBench & co. (test)
  harmless: Alpaca

Also writes a *contrast control* pair (harmless questions vs harmless imperatives): a
mean-diff direction built by the same pipeline from a contrast unrelated to refusal.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/{}.json"
OUT = Path(__file__).with_name("data")
SEED = 0
SIZES = {"train": 128, "val": 32, "test": 64}
N_CONTRAST = 128


def fetch(name: str) -> list[dict]:
    with urllib.request.urlopen(URL.format(name)) as r:
        return json.load(r)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def write(name: str, rows: list[dict], source: str) -> None:
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for i, r in enumerate(rows):
            row = {"id": f"{name}:{i}", "prompt": r["instruction"], "source": source}
            if r.get("category"):
                row["category"] = r["category"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    used: set[str] = set()

    def sample(pool: list[dict], n: int) -> list[dict]:
        # Dedup across all splits: the published splits overlap in a few items.
        fresh = [r for r in pool if norm(r["instruction"]) not in used]
        if len(fresh) < n:
            raise ValueError(f"need {n}, only {len(fresh)} unused items")
        out = rng.sample(fresh, n)
        used.update(norm(r["instruction"]) for r in out)
        return out

    for kind in ("harmful", "harmless"):
        for split, n in SIZES.items():
            src = f"{kind}_{split}"
            write(src, sample(fetch(src), n), src)

    # Contrast control from the harmless pool, disjoint from everything above.
    pool = fetch("harmless_train")
    questions = [r for r in pool if r["instruction"].strip().endswith("?")]
    imperatives = [r for r in pool if not r["instruction"].strip().endswith("?")]
    write("contrast_question", sample(questions, N_CONTRAST), "harmless_train")
    write("contrast_imperative", sample(imperatives, N_CONTRAST), "harmless_train")


if __name__ == "__main__":
    main()
