"""nnterp (nnsight) backend: exact HF forward pass + standardized module names across families.

nnsight gotchas encoded here (verified on nnsight 0.7 / nnterp 1.3):
- Modules must be accessed in forward-execution order inside a trace -> we sort Sites.
- Variables created inside a `with trace` block don't escape unless `.save()`d; containers
  must be created outside the block.
- `generator.output` contains prompt + new tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from interptemp.config import GenerationConfig, ModelConfig
from interptemp.interventions import Intervention, group_by_site
from interptemp.models.base import InterpModel
from interptemp.sites import Site, sort_sites
from interptemp.utils import batched, resolve_dtype


class NnterpModel(InterpModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        from nnterp import StandardizedTransformer

        kwargs: dict[str, Any] = {"dtype": resolve_dtype(cfg.dtype), "device_map": cfg.device_map}
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        # Escape hatch: the raw nnterp model, for anything the interface doesn't cover.
        self.model = StandardizedTransformer(cfg.name, **{**kwargs, **cfg.backend_kwargs})
        tok = self.model.tokenizer
        # Left padding makes negative positions (-1 = last prompt token) valid for every row.
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    # ---- tokenization -------------------------------------------------------------------

    def encode(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize without auto special tokens; add BOS only if missing (no double BOS)."""
        tok = self.tokenizer
        texts = list(prompts)
        if self.cfg.add_bos and tok.bos_token:
            texts = [t if t.startswith(tok.bos_token) else tok.bos_token + t for t in texts]
        return dict(tok(texts, return_tensors="pt", padding=True, add_special_tokens=False))

    # ---- site access --------------------------------------------------------------------

    def _layer_accessor(self, site: Site) -> Any:
        m = self.model
        return {
            "resid_pre": m.layers_input,
            "attn_out": m.attentions_output,
            "mlp_out": m.mlps_output,
            "resid_post": m.layers_output,
        }[site.kind]

    def _get(self, site: Site) -> Any:
        if site.kind == "embed":
            return self.model.token_embeddings
        return self._layer_accessor(site)[site.layer_index]

    def _set(self, site: Site, value: Any) -> None:
        if site.kind == "embed":
            self.model.token_embeddings = value
        else:
            self._layer_accessor(site)[site.layer_index] = value

    def _hook(
        self,
        interventions: Sequence[Intervention],
        record: Sequence[Site] = (),
        positions: Sequence[int] | None = None,
        saved: dict[Site, Any] | None = None,
    ) -> None:
        """Must be called inside a trace. Applies interventions and records post-edit values."""
        by_site = group_by_site(interventions)
        record_set = set(record)
        for site in sort_sites([*by_site, *record_set]):
            h = self._get(site)
            if site in by_site:
                for iv in by_site[site]:
                    h = iv(h, site)
                self._set(site, h)
            if site in record_set:
                assert saved is not None
                sel = h if positions is None else h[:, list(positions)]
                saved[site] = sel.cpu().save()

    # ---- public API ---------------------------------------------------------------------

    @torch.inference_mode()
    def activations(
        self,
        prompts: Sequence[str],
        sites: Sequence[Site],
        positions: Sequence[int] | None = (-1,),
        interventions: Sequence[Intervention] = (),
        batch_size: int = 16,
    ) -> dict[Site, torch.Tensor]:
        sites = [Site.parse(s) for s in sites]
        chunks: dict[Site, list[torch.Tensor]] = {s: [] for s in sites}
        for batch in batched(prompts, batch_size):
            saved: dict[Site, Any] = {}
            with self.model.trace(self.encode(batch)):
                self._hook(interventions, sites, positions, saved)
            for s in sites:
                chunks[s].append(saved[s])
        return {s: _cat_left_padded(v) for s, v in chunks.items()}

    @torch.inference_mode()
    def logits(
        self,
        prompts: Sequence[str],
        interventions: Sequence[Intervention] = (),
        positions: Sequence[int] | None = (-1,),
        batch_size: int = 16,
    ) -> torch.Tensor:
        chunks = []
        for batch in batched(prompts, batch_size):
            with self.model.trace(self.encode(batch)):
                self._hook(interventions)
                lg = self.model.logits
                sel = lg if positions is None else lg[:, list(positions)]
                out = sel.float().cpu().save()
            chunks.append(out)
        return _cat_left_padded(chunks)

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[str],
        gen: GenerationConfig,
        interventions: Sequence[Intervention] = (),
    ) -> list[str]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": gen.max_new_tokens,
            "do_sample": gen.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if gen.do_sample:
            kwargs |= {"temperature": gen.temperature, "top_p": gen.top_p}
        texts: list[str] = []
        for batch in batched(prompts, gen.batch_size):
            enc = self.encode(batch)
            with self.model.generate(enc, **kwargs) as tracer:
                if interventions:
                    with tracer.all():  # apply at prefill and every decode step
                        self._hook(interventions)
                out = self.model.generator.output.save()
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts += self.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=gen.skip_special_tokens
            )
        return texts


def _cat_left_padded(chunks: list[torch.Tensor]) -> torch.Tensor:
    """Concat along batch; left-pad dim 1 with zeros if batches differ in length (positions=None)."""
    t = max(c.shape[1] for c in chunks)
    return torch.cat([F.pad(c, (0, 0, t - c.shape[1], 0)) for c in chunks])
