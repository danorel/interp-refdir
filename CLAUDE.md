# Project notes for Claude Code

Mech-interp experiment template. See README.md for layout and extension points.

- Env: `uv` only (`uv run ...`, `uv add ...`). Never pip-install into the project env.
- vLLM lives in `envs/vllm` (separate env); don't add it to the main pyproject.
- Before finishing a change: `make check`; if models/ or sanity.py changed, also `make test-model`.
- New experiments go in `experiments/<name>/` (config.yaml + experiment.py subclassing `Experiment`);
  keep `src/interptemp` generic — no project-specific code there.
- nnsight rules: access modules in forward order inside a trace; create containers outside
  the `with` block; values escape only via `.save()`.
- Every intervention result needs a control; every LLM judge needs validation vs hand labels.
