.PHONY: install fmt lint typecheck test test-model check sanity

install:            ## env + git hooks
	uv sync
	uv run pre-commit install

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run pyright

test:               ## fast unit tests (no model, no API)
	uv run pytest

test-model:         ## integration tests on Qwen3-0.6B (CPU ok, ~1-2 min)
	uv run pytest -m model

check: lint typecheck test

sanity:             ## full sanity suite on the configured (big) model — run on the pod
	uv run interp-run experiments/sanity/config.yaml
