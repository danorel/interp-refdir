# Example run: full loop on real data

One run of this example, kept so you can see (and re-run) the whole
generate → blind-label → judge → agreement loop without a GPU.

| File | What |
|---|---|
| `config.yaml` | resolved run config: Qwen3-0.6B (CPU), `resid_post.14`, scales 0/1/2/4, 3 random controls, `max_new_tokens=40` |
| `generations.jsonl` | 78 completions (6 eval prompts × direction/random × scale) |
| `labels.jsonl` | blind human labels via `interp-label`; `revised_from` marks 6 labels changed after the rubric was finalised |
| `judge_labels.jsonl` | `anthropic/claude-haiku-4.5`, `reasoning=low`, final `../rubric.md` |

## Results (human labels)

| | direction | random (3 vectors) |
|---|---|---|
| scale 1 | 0/6 | 0/18 |
| scale 2 | 0/6 | 0/18 |
| scale 4 | 3/6 | 0/18 |

Judge vs human: accuracy 0.97, **Cohen's kappa 0.49** (3 human positives; judge 1/6 at
scale 4). The judge is conservative: it under-counts the effect but never flags a random
control.

## Caveats

- Toy data: 6 prompts, greedy decoding, one seed, a 0.6B model.
- The kappa is measured on the same texts the rubric was tuned on (a *dev* set), so it is
  optimistic; with 3 positives, one item moves it by ~0.25. A real judge validation needs a
  held-out, blindly labeled set with more positives.

## Reproduce

```bash
uv run interp-label summary experiments/example_steering/results/labels.jsonl --by vector,scale
uv run python experiments/example_steering/judge_existing.py experiments/example_steering/results \
    --model anthropic/claude-haiku-4.5 --reasoning low   # needs OPENROUTER_API_KEY, ~$0.10
```
