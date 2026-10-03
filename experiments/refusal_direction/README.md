# Refusal direction (Arditi et al. 2024) with random-vector and contrast controls

**Status: paused, partially complete.** Ablation (necessity) is replicated on Qwen3-4B with
controls; addition (sufficiency) is under-measured and the LLM judge is not yet validated.
See [What is missing](#what-is-missing) before reusing any number.

## Question

Is refusal in a chat model mediated by a single direction `r` in the residual stream?
If so, two causal tests should pass, and fail for control vectors:

- **Necessity:** ablating `r` everywhere makes the model comply with harmful requests.
- **Sufficiency:** adding `r` at one layer makes the model refuse harmless requests.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-4B` (bf16, Apple MPS; template sanity checks pass) |
| Data | Arditi et al.'s published splits ([prepare_data.py](prepare_data.py)), deduplicated across splits |
| train | 128 harmful (AdvBench/TDC/HarmBench) + 128 harmless (Alpaca): direction only |
| val | 32 + 32: all selection (layer, position, α) |
| test | 64 + 64 (harmful mostly JailbreakBench): reported numbers only |
| Direction | `mean(harmful) − mean(harmless)` at `resid_pre` of layer *l*, position *p* |
| Candidates | 9 post-instruction positions (`<|im_end|>` … `</think>\n\n`) × layers < 0.8·L |
| Ablation | `h − (h·r̂)r̂` on embed + every attn/mlp output (≡ weight orthogonalization) |
| Addition | `h + α·r` at `resid_pre` of layer *l*, all positions incl. generated tokens |
| Controls | 5 norm-matched random vectors; a *contrast* direction (harmless questions vs imperatives, same construction, scaled to ‖r‖) |
| Metrics | substring refusal; logit proxy `log-odds P(first token ∈ {"I","As"})`; LLM judge `refusal / compliance / incoherent` (incoherent and judge failures stay in the denominator); Wilson 95% CIs |

**Selection rule** (Arditi et al., [experiment.py](experiment.py) `select_direction`): among
candidates whose ablation keeps harmless next-token KL < 0.1, and whose addition induces
refusal (proxy > 0), take the strongest refusal bypass. The KL filter is never dropped: a
direction that breaks the model also "bypasses refusal".

## Results (Qwen3-4B, run `20261002-211058`)

The judge failed on every call in this run (environment issue, since fixed to fail fast), so
**all numbers below are substring refusal / logit proxy, not judge labels**. The proxy agrees
with substring refusal on 96% of the 1088 test generations.

![per-layer ablation](results/qwen3-4b/layers.png)

- Selected on val: **layer 21 (58% depth), position `<|im_end|>`**, ‖r‖ = 34.1, val KL 0.075.
  It is also the best layer on test, so selection did not overfit val.
- The effect is confined to **layers 20–23**: ablation drops harmful-test refusal to 2–3%.
  Every other swept layer stays at 53–77% (72% without intervention; logit proxy).

![test conditions](results/qwen3-4b/conditions.png)

| Test condition (n=64; random pooled 5×64) | Ablate on harmful | Add on harmless (α=0.5) |
|---|---|---|
| no intervention | 69% [57–79] | 2% [0–8] |
| **r** | **2% [0–8]** | **19% [11–30]** |
| random ×5 | 70% [64–74] | 2% [1–4] |
| contrast | 72% [60–81] | 2% [0–8] |

- **Necessity: replicated.** Non-overlapping CIs between `r` and both controls. Ablating `r`
  on harmless prompts leaves them unchanged (2% refusal).
- **Sufficiency: under-measured, but specific.** With the judge down, the α stage picked the
  smallest α on noise (0.5). On val, α=1 gives 91% and α=2 gives 100% substring refusal;
  α=4 drops to 69%, likely degeneration (unverified without judge labels).
- Ablating `r` removes 15.3% of ‖h‖² on harmful vs 0.25% on harmless prompts (60×).
  Random directions remove ~0.01%, contrast 0.8%; `cos(contrast, r)` = 0.07.

Smaller models: Qwen3-0.6B refuses ~0/32 and Qwen3-1.7B 7/32 harmful val prompts at
baseline, so neither is usable for the ablation test. On 0.6B no candidate induces refusal.

## What is missing

To finish the project, in order:

1. **Re-run α and eval with a working judge** (~15 min on MPS, sweep skipped):
   `uv run interp-run experiments/refusal_direction/config.yaml model=configs/models/qwen3-4b.yaml model.device_map=mps 'params.stages=[baseline,alpha,eval]' params.layer=21 params.pos=-9`
2. **Validate the judge on hand labels** (~60 rows, blind) with `interp-label` +
   [validate_judge.py](validate_judge.py); require κ ≥ 0.7 before reporting judge rates.
   Known metric failure modes to cover: "I'm sorry to hear…" (substring false positive),
   "As of 2024…" (proxy false positive on `As`).
3. **Replicate on Qwen3-8B** (`configs/models/qwen3-8b.yaml`): needs a 48 GB GPU (e.g. 1× L40S);
   run `make sanity` first. Compare at relative depth `l/L`.
4. **Stronger ablation control.** The contrast direction removes ~20× less ‖h‖² than `r`, so it
   is not a norm-removal-matched control. Build one that removes a comparable share.
5. Optional: heatmap of the sweep over (layer × position); more data seeds; α sweep with
   incoherence rate from the judge.

Known limitations: val has only 32 harmful prompts; one data seed; one model; KL threshold and
0.8·L cutoff taken from the paper, not calibrated here. Generations are not committed (the
ablated model's harmful completions should not be published); regenerate them locally.

## Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | all parameters; stages `baseline → sweep → alpha → eval` |
| [experiment.py](experiment.py) | experiment + pure metric functions (unit-tested in `tests/test_refusal_direction.py`) |
| [prepare_data.py](prepare_data.py) | builds `data/*.jsonl` from Arditi et al.'s splits |
| [rubric.md](rubric.md) | judge rubric (`refusal / compliance / incoherent`) |
| [validate_judge.py](validate_judge.py) | judge vs blind human labels (κ, confusions) |
| [plot_layers.py](plot_layers.py) | per-layer test curve + condition bars from a run dir |
| `results/qwen3-4b/` | aggregate outputs of the run above (no generations) |
