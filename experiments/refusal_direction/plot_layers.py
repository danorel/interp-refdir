"""Per-layer refusal before/after ablation (+ test-set conditions) for a finished run.

    uv run python experiments/refusal_direction/plot_layers.py outputs/refusal_direction/<run> \
        [--device mps]

For every swept layer, takes the position the sweep chose on *val* (min bypass among
low-KL candidates), ablates that direction everywhere and measures refusal on the held-out
*test* harmful prompts. Refusal here = logit proxy: P(first token in refusal_tokens) > 0.5,
whose agreement with substring refusal on the run's generations is printed and plotted.

Writes <run>/layer_curve.jsonl (cache; delete to recompute), layers.png, conditions.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `experiments.*`

from experiments.refusal_direction.metrics import refusal_score, select_direction, wilson
from experiments.refusal_direction.schema import SweepCandidate, is_pooled
from interptemp.config import ModelConfig
from interptemp.interventions import DirectionalAblation
from interptemp.models.base import InterpModel, build_model
from interptemp.store import read_jsonl, write_jsonl
from interptemp.tasks import JsonlTask

# Reference palette (dataviz skill), first three categorical slots: validated all-pairs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e0"


def pick_positions(sweep: list[dict], max_kl: float) -> dict[int, tuple[SweepCandidate, bool]]:
    """Per layer: the sweep's own selection rule restricted to that layer (val data)."""
    by_layer: dict[int, list[SweepCandidate]] = defaultdict(list)
    for row in sweep:
        by_layer[row["layer"]].append(SweepCandidate(**row))
    out = {}
    for layer, candidates in sorted(by_layer.items()):
        try:
            out[layer] = (select_direction(candidates, max_kl)[0], True)
        except ValueError:  # every position breaks harmless outputs at this layer
            out[layer] = (min(candidates, key=lambda c: c.kl), False)
    return out


def compute_curve(run: Path, device: str | None) -> list[dict]:
    cfg_all = yaml.safe_load((run / "config.yaml").read_text())
    p = cfg_all["params"]
    mcfg = ModelConfig(**cfg_all["model"])
    if device:
        mcfg.device_map = device
    m = build_model(mcfg)
    assert isinstance(m, InterpModel)

    exs = JsonlTask(f"{p['data_dir']}/harmful_test.jsonl").load()
    prompts = [m.format_chat(ex.messages) for ex in exs]
    ids = [m.tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in p["refusal_tokens"]]
    dirs = torch.load(run / "directions.pt")
    positions = list(p["positions"])
    tok_ids = m.tokenizer(prompts[0], add_special_tokens=False)["input_ids"]

    def scores(interventions=()) -> list[float]:
        lg = m.logits(prompts, interventions, positions=[-1], batch_size=p["batch_size"])[:, 0]
        return refusal_score(lg, ids).tolist()

    clean = scores()
    rows = [{"layer": -1, "scores": clean}]  # layer -1 = no intervention
    picks = pick_positions(read_jsonl(run / "sweep.jsonl"), p["sweep"]["max_kl"])
    for layer, (sel, kl_ok) in picks.items():
        r = dirs[layer, positions.index(sel.pos)]
        s = scores([DirectionalAblation.everywhere(r, m.num_layers)])
        rows.append(
            {
                "layer": layer,
                "pos": sel.pos,
                "pos_token": m.tokenizer.decode([tok_ids[sel.pos]]),
                "kl_val": sel.kl,
                "kl_ok": kl_ok,
                "induce_val": sel.induce_score,
                "scores": s,
            }
        )
        print(f"layer {layer}: pos {sel.pos} refusal {sum(x > 0 for x in s)}/{len(s)}")
    return rows


def refusal_rate(scores: list[float]) -> tuple[float, float, float]:
    k, n = sum(s > 0 for s in scores), len(scores)
    lo, hi = wilson(k, n)
    return k / n, lo, hi


def proxy_agreement(run: Path) -> float | None:
    """How often the logit proxy (score > 0) matches substring refusal on real generations.

    None when the generations are absent — published result dirs keep only the aggregates,
    so they can still redraw their own plots.
    """
    path = run / "eval_test.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return sum((r["refusal_score"] > 0) == r["substring_refusal"] for r in rows) / len(rows)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_layers(run: Path, curve: list[dict], summary: dict, agree: float | None) -> Path:
    import matplotlib.pyplot as plt

    base = next(r for r in curve if r["layer"] == -1)
    layers = [r for r in curve if r["layer"] >= 0]
    b, blo, bhi = refusal_rate(base["scores"])
    chosen = summary["direction"]["layer"]
    num_layers = summary["num_layers"]

    agreement = (
        f", agrees with substring refusal on {agree:.0%} of generations"
        if agree is not None
        else ""
    )
    fig, (ax, ax_kl) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    xs = [r["layer"] for r in layers]
    x0, x1 = min(xs) - 0.5, max(xs) + 0.5

    # Baseline: band = 95% CI, line = rate (constant across layers: no intervention).
    ax.axhspan(blo * 100, bhi * 100, color=GRAY, alpha=0.15, linewidth=0)
    ax.axhline(b * 100, color=GRAY, linewidth=2)
    ax.text(x1, b * 100 + 2, f"no intervention: {b:.0%}", ha="right", va="bottom",
            color=MUTED, fontsize=9)  # fmt: skip

    for r in layers:
        v, lo, hi = refusal_rate(r["scores"])
        ok = r["kl_ok"]
        ax.errorbar(r["layer"], v * 100, yerr=[[(v - lo) * 100], [(hi - v) * 100]],
                    fmt="o", ms=8, color=BLUE, mfc=BLUE if ok else "white", mew=2,
                    elinewidth=1.5, capsize=3)  # fmt: skip
    ax.plot(xs, [refusal_rate(r["scores"])[0] * 100 for r in layers], color=BLUE,
            linewidth=1, alpha=0.4)  # fmt: skip

    # Best layer = the one the experiment selected on val (not re-picked on test).
    sel = next(r for r in layers if r["layer"] == chosen)
    sv, _, _ = refusal_rate(sel["scores"])
    ax.scatter([chosen], [sv * 100], s=380, marker="*", color=ORANGE, zorder=5,
               edgecolor="white", linewidth=1.5)  # fmt: skip
    ax.annotate(
        f"selected on val: layer {chosen} ({chosen / num_layers:.0%} depth), pos {sel['pos_token']!r}\n"
        f"refusal {b:.0%} → {sv:.0%} on test (overcome {b - sv:+.0%} pts)",
        xy=(chosen, sv * 100), xytext=(chosen - 9, 38), color=INK, fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
    )  # fmt: skip

    ax.set_ylim(-3, 103)
    ax.set_ylabel("refusal rate on harmful test, %", color=MUTED, fontsize=10)
    fig.suptitle(
        f"{summary['model']}: ablating the per-layer refusal direction everywhere",
        x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold",
    )  # fmt: skip
    ax.set_title(
        f"n={len(base['scores'])} harmful test prompts · bars = Wilson 95% CI · hollow = no "
        "position passes KL<0.1 at this layer\nrefusal = logit proxy P(first token ∈ refusal "
        f"tokens) > 0.5{agreement}",
        loc="left", color=MUTED, fontsize=8.5,
    )  # fmt: skip
    style(ax)

    kl = [r["kl_val"] for r in layers]
    ax_kl.bar(xs, kl, width=0.6, color=[BLUE if r["kl_ok"] else GRAY for r in layers])
    ax_kl.axhline(0.1, color=INK, linestyle="--", linewidth=1)
    ax_kl.text(x1, 0.1, " KL limit 0.1", color=INK, fontsize=8, va="bottom", ha="right")
    ax_kl.set_yscale("log")
    ax_kl.set_ylabel("KL on harmless val\n(side effects)", color=MUTED, fontsize=9)
    ax_kl.set_xlabel("layer", color=MUTED, fontsize=10)
    ax_kl.set_xlim(x0, x1)
    style(ax_kl)

    fig.tight_layout()
    out = run / "layers.png"
    fig.savefig(out, dpi=150)
    return out


def plot_conditions(run: Path, summary: dict) -> Path:
    import matplotlib.pyplot as plt

    cond = summary["eval"]["conditions"]
    alpha = summary["direction"]["alpha"]
    # Judge labels once the judge is validated; substring refusal is the fallback.
    metric = "refusal" if "refusal" in next(iter(cond.values())) else "substring_refusal"
    # `random*` is a pooled view of random_0..4; skip it so rows aren't counted twice.
    missing = sum(c.get("judge_missing", 0) for k, c in cond.items() if not is_pooled(k))
    n_rows = sum(c[metric]["n"] for k, c in cond.items() if not is_pooled(k))
    panels = [
        ("harmful", "ablate", "Necessity: ablate on harmful (lower = refusal removed)"),
        ("harmless", "add", f"Sufficiency: add α={alpha}·vector on harmless (higher = induced)"),
    ]
    bars = [("none", "-", "no intervention", GRAY), (None, "r", "refusal dir r", BLUE),
            (None, "random*", "random ×5 (pooled)", ORANGE), (None, "contrast", "contrast ctrl", AQUA)]  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pset, iv, title) in zip(axes, panels, strict=True):
        for i, (iv_override, vec, _label, color) in enumerate(bars):
            s = cond[f"{pset}/{iv_override or iv}/{vec}"][metric]
            v, (lo, hi) = s["rate"] * 100, (x * 100 for x in s["ci95"])
            ax.bar(i, v, width=0.6, color=color)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, elinewidth=1.2)
            ax.text(i, hi + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
        ax.set_xticks(range(len(bars)), [b[2] for b in bars], fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    label = "judge" if metric == "refusal" else "substring"
    axes[0].set_ylabel(f"{label} refusal rate on test, %", color=MUTED, fontsize=10)
    axes[0].set_ylim(0, 110)
    fig.suptitle(
        f"{summary['model']} · layer {summary['direction']['layer']} · n=64 per bar "
        f"(random: 5×64) · Wilson 95% CI\n{missing}/{n_rows} rows blocked by the judge's "
        "provider, counted as non-refusal",
        x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.tight_layout()
    out = run / "conditions.png"
    fig.savefig(out, dpi=150)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--device", help="override model.device_map, e.g. mps")
    args = ap.parse_args()

    cache = args.run / "layer_curve.jsonl"
    if cache.exists():
        curve = read_jsonl(cache)
    else:
        curve = compute_curve(args.run, args.device)
        write_jsonl(cache, curve)
    summary = json.loads((args.run / "summary.json").read_text())
    agree = proxy_agreement(args.run)
    print(
        "logit proxy vs substring agreement: "
        + (f"{agree:.1%}" if agree is not None else "n/a (no generations in this dir)")
    )
    print(plot_layers(args.run, curve, summary, agree))
    print(plot_conditions(args.run, summary))


if __name__ == "__main__":
    main()
