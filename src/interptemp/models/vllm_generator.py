"""Generation-only backend running vLLM in its own env (envs/vllm) via subprocess.

Use for bulk sampling (e.g. long CoT); use NnterpModel whenever you need internals.
Prompts must already be chat-formatted (`format_chat`), same as for NnterpModel.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from interptemp.config import GenerationConfig, ModelConfig
from interptemp.models.base import Generator
from interptemp.store import read_jsonl, write_jsonl

VLLM_PROJECT = Path(__file__).resolve().parents[3] / "envs" / "vllm"


class VLLMGenerator(Generator):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # dtype/revision map onto vLLM LLM(...) kwargs; backend_kwargs pass through verbatim.
        self.llm_kwargs = {"dtype": cfg.dtype, **cfg.backend_kwargs}
        if cfg.revision:
            self.llm_kwargs["revision"] = cfg.revision

    def generate(self, prompts: Sequence[str], gen: GenerationConfig) -> list[str]:
        params = {
            "max_tokens": gen.max_new_tokens,
            "temperature": gen.temperature if gen.do_sample else 0.0,
            "top_p": gen.top_p if gen.do_sample else 1.0,
            "skip_special_tokens": gen.skip_special_tokens,
        }
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / "in.jsonl", Path(d) / "out.jsonl"
            write_jsonl(inp, ({"prompt": p} for p in prompts))
            cmd = [
                "uv", "run", "--project", str(VLLM_PROJECT), "python", str(VLLM_PROJECT / "generate.py"),
                "--model", self.cfg.name, "--input", str(inp), "--output", str(out),
                "--params", json.dumps(params), "--llm-kwargs", json.dumps(self.llm_kwargs),
            ]  # fmt: skip
            if not self.cfg.add_bos:
                cmd.append("--no-add-bos")
            subprocess.run(cmd, check=True)
            rows = read_jsonl(out)
        if len(rows) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(rows)} rows for {len(prompts)} prompts")
        return [r["completion"] for r in rows]
