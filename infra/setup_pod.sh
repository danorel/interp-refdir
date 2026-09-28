#!/usr/bin/env bash
# Bootstrap a fresh GPU box (Sesterce / Runpod / Vast). Idempotent; re-run after restarts.
#   git clone <repo> /workspace/<proj> && cd /workspace/<proj> && bash infra/setup_pod.sh
set -euo pipefail

PERSIST="${PERSIST:-/workspace}"   # mount point of the persistent volume, if any
export HF_HOME="${HF_HOME:-$PERSIST/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST/uv-cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Persist env vars for new shells.
grep -q "HF_HOME=" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<RC
export HF_HOME=$HF_HOME
export UV_CACHE_DIR=$UV_CACHE_DIR
export PATH="\$HOME/.local/bin:\$PATH"
RC

uv sync
[ -f .env ] || { cp .env.example .env; echo ">> fill in .env (OPENROUTER_API_KEY, HF_TOKEN)"; }
set -a; source .env; set +a

uv run python - <<PY
import torch
assert torch.cuda.is_available(), "CUDA not available: wrong image / driver?"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY

uv run hf download "$MODEL" --exclude "*.pth" "original/*" >/dev/null
echo ">> weights cached in $HF_HOME"

if [ "${WITH_VLLM:-0}" = "1" ]; then
  uv sync --project envs/vllm
fi

echo ">> next: uv run pytest && make sanity"
