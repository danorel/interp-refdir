"""LLM judge via OpenRouter (OpenAI-compatible API).

- Raw responses are cached (keyed by model + messages + params), parsing happens after the
  cache, so fixing a parser never re-pays for calls.
- Failed calls / unparseable outputs yield label=None instead of raising: report the
  failure rate, don't silently drop rows.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from interptemp.cache import DiskCache, stable_hash
from interptemp.config import JudgeConfig
from interptemp.judges.base import Judge, JudgeInput, JudgeResult
from interptemp.utils import get_logger

log = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Appended verbatim (not .format-ed), so rubrics can't break it and it can contain braces.
_OUTPUT_SPEC = """

Respond with a single JSON object and nothing else:
{"reasoning": "<brief justification>", "label": "<one of: %s>"}"""


class LLMJudge(Judge):
    def __init__(
        self,
        cfg: JudgeConfig,
        rubric: str,
        labels: Sequence[str] = ("yes", "no"),
        system: str | None = None,
    ):
        """`rubric` is a str.format template over {prompt}, {response} and JudgeInput.extra."""
        self.cfg, self.rubric, self.labels, self.system = cfg, rubric, list(labels), system
        self.cache = DiskCache(cfg.cache_path)
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set (see .env.example)")
            self._client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=5)
        return self._client

    def build_messages(self, item: JudgeInput) -> list[dict[str, str]]:
        body = self.rubric.format(prompt=item.prompt, response=item.response, **item.extra)
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return [*msgs, {"role": "user", "content": body + _OUTPUT_SPEC % ", ".join(self.labels)}]

    def parse(self, raw: str) -> tuple[str | None, str]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None, ""
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, ""
        label = str(obj.get("label", "")).strip().lower()
        allowed = {lab.lower(): lab for lab in self.labels}
        return allowed.get(label), str(obj.get("reasoning", ""))

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.cfg.reasoning_effort:
            req["extra_body"] = {"reasoning": {"effort": self.cfg.reasoning_effort}}
        return req

    async def _one(self, item: JudgeInput, sem: asyncio.Semaphore) -> JudgeResult:
        req = self._request(self.build_messages(item))
        key = stable_hash(req)
        raw = self.cache.get(key)
        meta: dict[str, Any] = {"cached": raw is not None}
        if raw is None:
            async with sem:
                try:
                    resp = await self.client.chat.completions.create(**req)
                    raw = resp.choices[0].message.content or ""
                except Exception as e:  # after client retries; keep the row, flag it
                    return JudgeResult(label=None, meta={"error": repr(e)})
            meta["reasoning_tokens"] = _reasoning_tokens(resp)
            self.cache.set(key, raw)
        label, reasoning = self.parse(raw)
        return JudgeResult(label=label, reasoning=reasoning, raw=raw, meta=meta)

    async def ajudge(self, items: Sequence[JudgeInput]) -> list[JudgeResult]:
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        results = await tqdm_asyncio.gather(*(self._one(it, sem) for it in items), desc="judge")
        self._warn_if_reasoning_dropped(results)
        return results

    def _warn_if_reasoning_dropped(self, results: Sequence[JudgeResult]) -> None:
        """Providers may silently skip reasoning, e.g. Anthropic via OpenRouter when max_tokens
        leaves less than its 1024-token minimum thinking budget. Surface that instead."""
        if not self.cfg.reasoning_effort:
            return
        counts = [r.meta["reasoning_tokens"] for r in results if "reasoning_tokens" in r.meta]
        if counts and all(c == 0 for c in counts):
            log.warning(
                f"reasoning_effort={self.cfg.reasoning_effort!r} requested but all {len(counts)} "
                f"fresh responses from {self.cfg.model} used 0 reasoning tokens; reasoning was "
                f"likely dropped. Try a larger max_tokens (now {self.cfg.max_tokens})."
            )

    def judge(self, items: Sequence[JudgeInput]) -> list[JudgeResult]:
        # Fresh client per call: AsyncOpenAI's connection pool binds to the event loop that
        # first used it, and each asyncio.run creates a new loop.
        self._client = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ajudge(items))
        # Already inside an event loop (Jupyter / `# %%` cells): run in a worker thread.
        with ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, self.ajudge(items)).result()


def _reasoning_tokens(resp: Any) -> int | None:
    """Reasoning tokens from an OpenAI-style usage block; None if the provider doesn't report."""
    details = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", None)
