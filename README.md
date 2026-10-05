# interptemp

Template for fast mech-interp experiments: generation, activations, interventions
(steering / ablation), and LLM judges — backend- and model-agnostic, with sanity checks
built in. Default stack: [nnterp](https://github.com/Butanium/nnterp) (nnsight) + Qwen3 +
OpenRouter, managed with `uv`.

## Results

### Refusal is mediated by a single direction (Qwen3-4B)

A replication of [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) — full write-up,
method and caveats in [`experiments/refusal_direction/`](experiments/refusal_direction/).

A single mean-diff direction `r`, read off layer 21 at the post-instruction token, controls
refusal in both directions on held-out test prompts. Judge-labeled, n=64 per condition,
Wilson 95% CIs:

| Test condition | Ablate `r` on harmful | Add `r` on harmless |
|---|---|---|
| no intervention | 86% [75–92] refusal | 0% [0–6] refusal |
| **`r`** | **5% [2–13]** | **98% [92–100]** |
| 5 norm-matched random vectors | 88% [84–91] | 0% [0–1] |
| contrast-direction control | 88% [77–94] | 2% [0–8] |

Removing `r` strips 15.3% of ‖h‖² on harmful prompts but 0.25% on harmless ones, and leaves
harmless behaviour untouched (64/64 compliance). Neither control family moves either number.

![test conditions](experiments/refusal_direction/results/qwen3-4b/conditions.png)

The effect is confined to layers 20–23; every other layer stays within the no-intervention
interval. The lower panel is the side-effect guard (KL on harmless prompts) that keeps the
sweep from "removing refusal" by simply breaking the model:

![layer sweep](experiments/refusal_direction/results/qwen3-4b/layers.png)

The judge was validated at κ = 0.93 against blind hand labels before any of these rates were
reported. **One model is not a replication** — reproducing this on Qwen3-8B is the main open
item, along with disentangling "harmful" from "topic" in the training contrast.

## Status

Early and lightly tested — please open an issue if something breaks.

| Component | Tested |
|---|---|
| nnterp backend, interventions, sanity checks | ✅ end-to-end on Qwen3-0.6B (CPU); unit + integration tests |
| LLM judge via OpenRouter (Gemini, Claude Haiku) | ✅ real API runs |
| Blind labeling CLI | ✅ manual use |
| GPU / Qwen3-8B, `make sanity` on a pod | ❌ not yet |
| vLLM backend (`envs/vllm`) | ❌ never run |
| Gemma config, `infra/setup_pod.sh` | ❌ not yet |

## Quickstart

```bash
make install                 # uv sync + pre-commit hooks
cp .env.example .env         # OPENROUTER_API_KEY, HF_TOKEN
make test                    # unit tests, no model (seconds)
make test-model              # integration tests on Qwen3-0.6B, CPU ok (~1-2 min)

# Any experiment = one config. Swap model / override anything from the CLI:
uv run interp-run experiments/refusal_direction/config.yaml \
    model=configs/models/qwen3-0.6b.yaml judge=null generation.max_new_tokens=16
```

Each run writes `outputs/<name>/<timestamp>/` with the resolved `config.yaml`, `meta.json`
(git sha + dirty flag), results, and `summary.json`.

## GPU box (Sesterce / Runpod / Vast)

1. Rent **1× H100 80GB** (or A100 80GB). Qwen3-8B bf16 ≈ 16 GB weights; the rest is KV cache
   for long CoT, cached activations, and the 2nd weight copy used by the HF-parity check.
   Attach a persistent volume if available (mount at `/workspace`).
2. `git clone <repo> /workspace/<proj> && cd /workspace/<proj> && bash infra/setup_pod.sh`
   (`WITH_VLLM=1` to also build the vLLM env, `MODEL=...` to prefetch another model).
3. Connect from Cursor / VS Code via Remote-SSH.
4. `make sanity` — **all checks must pass before any experiment on a new model/pod.**
5. Stop the instance when idle. Code lives in git; pull results with
   `rsync -avz <host>:/workspace/<proj>/outputs/ outputs/`.

## Layout

```
src/interptemp/
  sites.py          Site("resid_post", 12): backend-agnostic hook points, execution-ordered
  interventions.py  Intervention ABC; AddVector, DirectionalAblation(.everywhere), Lambda
  directions.py     mean_diff, random_like (norm-matched control), project, cosine
  models/           Generator ABC (text only) -> InterpModel ABC (+ internals)
                    NnterpModel (default), VLLMGenerator (bulk sampling, separate env)
  tasks/            Task ABC + Example; JsonlTask, HFDatasetTask, split()
  judges/           Judge ABC; LLMJudge (OpenRouter, cached), SubstringJudge; metrics (kappa)
  experiment.py     Experiment ABC: lazy model/judge, run dir, save helpers
  sanity.py         reusable checks + SanityExperiment
  label.py          blind labeling CLI (`interp-label label|summary`) for judge validation
  config.py         typed YAML configs + dotted CLI overrides
  registry.py       short names -> classes; or any "module:Class"
configs/models/     per-model YAMLs (qwen3-8b, qwen3-0.6b, gemma-3-4b-it, qwen3-8b-vllm)
experiments/        one dir per experiment: config.yaml + experiment.py (+ data/)
envs/vllm/          isolated vLLM env (its torch pin conflicts with nnterp)
infra/              pod bootstrap
```

## Extending

| Want to...                  | Do                                                                               |
|-----------------------------|----------------------------------------------------------------------------------|
| New experiment              | copy `experiments/refusal_direction/`, subclass `Experiment`, set `target:`      |
| New model (same backend)    | add `configs/models/<m>.yaml`; run `make sanity` with `model=<that yaml>`        |
| New backend (TL, NDIF, ...) | subclass `InterpModel`, reference as `model.backend: pkg.mod:Class`              |
| New intervention            | subclass `Intervention` (pure `h -> h'` at declared sites); unit-test w/o model  |
| New dataset                 | `JsonlTask`/`HFDatasetTask` in config, or subclass `Task` (+ `score` if possible)|
| New judge rubric            | override `Experiment.judge_kwargs()` → `{"rubric": ..., "labels": [...]}`        |
| New judge type              | subclass `Judge`, reference as `judge.backend: pkg.mod:Class`                    |
| Anything nnsight-specific   | `self.imodel.model` is the raw `StandardizedTransformer`                         |

Method-specific knobs go in `params:` (free-form); promote to typed config only when shared.

## Conventions / gotchas

- **Prompts are fully formatted strings.** Call `model.format_chat(messages)` first; its
  defaults come from `model.chat_template_kwargs` (e.g. `enable_thinking`). BOS is added only
  if missing — no double BOS on Llama/Gemma.
- **Left padding everywhere**, so `positions=[-1]` = last prompt token for every row.
- Interventions in `generate` apply at prefill *and* every decode step; at decode `seq == 1`.
- `DirectionalAblation.everywhere` ablates embed + every attn/mlp output ⇒ the residual stream
  never contains r̂ (equivalent to weight orthogonalization).
- nnsight ≥ 0.5 requires accessing modules in forward order inside a trace — `NnterpModel`
  sorts sites; do the same in custom trace code. Containers must be created *outside* the
  `with trace` block, values leave it only via `.save()`.
- Always include a control (random norm-matched direction, other layer, shuffled labels).
- Validate every LLM judge on ~50 hand labels (`judges.metrics.agreement_report`) before
  trusting it. Judge failures are `label=None` — report the rate, never drop silently.
- Judge responses are cached in `.cache/judge.sqlite` (keyed on model + prompt + params).

## Tooling

`ruff` (format + lint), `pyright` (basic), `pytest`, `pre-commit` — all via `uv run`, versions
locked in `uv.lock`. `make check` = lint + typecheck + unit tests.
