"""Backend interfaces.

`Generator` is the minimum (text in -> text out); `InterpModel` adds access to internals.
Prompts are always fully-formatted strings: call `format_chat` first for chat models, so the
exact text fed to the model is visible and loggable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from interptemp.config import GenerationConfig, ModelConfig
from interptemp.interventions import Intervention
from interptemp.registry import resolve
from interptemp.sites import Site

Message = dict[str, str]


class Generator(ABC):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._tokenizer: Any = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.name, revision=self.cfg.revision
            )
        return self._tokenizer

    def format_chat(self, messages: Sequence[Message], **template_kwargs: Any) -> str:
        """Render messages with the chat template, ending in an open assistant turn.

        `cfg.chat_template_kwargs` (e.g. enable_thinking=False) are defaults; call kwargs win.
        """
        kwargs = {**self.cfg.chat_template_kwargs, **template_kwargs}
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **kwargs
        )

    @abstractmethod
    def generate(self, prompts: Sequence[str], gen: GenerationConfig) -> list[str]:
        """Return only the newly generated text for each prompt."""


class InterpModel(Generator):
    """Generation + internals. Implementations must apply interventions in site order."""

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @abstractmethod
    def encode(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Exact tokenization used by every method (left-padded, BOS handled once)."""

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        gen: GenerationConfig,
        interventions: Sequence[Intervention] = (),
    ) -> list[str]:
        """Generate with interventions applied at every decoding step."""

    @abstractmethod
    def logits(
        self,
        prompts: Sequence[str],
        interventions: Sequence[Intervention] = (),
        positions: Sequence[int] | None = (-1,),
        batch_size: int = 16,
    ) -> torch.Tensor:
        """[B, len(positions), vocab] (or [B, T, vocab] if positions is None), on CPU."""

    @abstractmethod
    def activations(
        self,
        prompts: Sequence[str],
        sites: Sequence[Site],
        positions: Sequence[int] | None = (-1,),
        interventions: Sequence[Intervention] = (),
        batch_size: int = 16,
    ) -> dict[Site, torch.Tensor]:
        """Post-intervention activations per site: [B, len(positions), d] on CPU.

        Positions index from the end (left padding), so -1 is the last prompt token for
        every row. positions=None returns the full left-padded sequence [B, T, d].
        """


def build_model(cfg: ModelConfig) -> Generator:
    cls = resolve("model", cfg.backend)
    return cls(cfg)
