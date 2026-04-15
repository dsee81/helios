from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ConditionedEmbeddings:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor]


def ensure_batched_prompt_embeds(prompt_embeds: torch.Tensor) -> torch.Tensor:
    if prompt_embeds.ndim == 2:
        return prompt_embeds.unsqueeze(0)
    if prompt_embeds.ndim != 3:
        raise ValueError(f"Expected prompt embeds with ndim 2 or 3, got shape {tuple(prompt_embeds.shape)}")
    return prompt_embeds


def _slice_or_pad_sequence(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[0] >= target_length:
        return sequence[:target_length]
    if sequence.shape[0] == 0:
        raise ValueError("Cannot initialize turn-bin embeddings from an empty prompt embedding sequence.")
    pad = sequence[-1:].expand(target_length - sequence.shape[0], -1)
    return torch.cat([sequence, pad], dim=0)


class TurnBinEmbeddingBank(nn.Module):
    def __init__(
        self,
        init_bank: torch.Tensor,
        init_scale: float = 0.05,
        learnable_delta: bool = True,
    ):
        super().__init__()
        if init_bank.ndim != 3:
            raise ValueError(f"Expected init_bank with shape [num_bins, num_vectors, hidden], got {tuple(init_bank.shape)}")
        self.num_bins, self.num_vectors, self.hidden_size = init_bank.shape
        self.learnable_delta = learnable_delta
        self.register_buffer("init_bank", init_bank.detach().clone(), persistent=True)
        if learnable_delta:
            self.delta = nn.Parameter(torch.zeros_like(init_bank))
            if init_scale > 0:
                nn.init.normal_(self.delta, mean=0.0, std=init_scale)
        else:
            self.delta = nn.Parameter(init_bank.detach().clone())

    @classmethod
    def from_phrases(
        cls,
        generator,
        init_phrases: list[str],
        num_vectors: int,
        init_scale: float = 0.05,
        learnable_delta: bool = True,
    ) -> "TurnBinEmbeddingBank":
        with torch.no_grad():
            phrase_embeds = generator.encode_prompt_text(init_phrases)
            init_slices = []
            for phrase_embed in phrase_embeds:
                init_slices.append(_slice_or_pad_sequence(phrase_embed, num_vectors))
            init_bank = torch.stack(init_slices, dim=0)
        return cls(init_bank=init_bank, init_scale=init_scale, learnable_delta=learnable_delta)

    def effective_bank(self) -> torch.Tensor:
        if self.learnable_delta:
            return self.init_bank + self.delta
        return self.delta

    def forward(self, bin_ids: torch.Tensor) -> torch.Tensor:
        if bin_ids.ndim != 1:
            bin_ids = bin_ids.view(-1)
        return self.effective_bank()[bin_ids]


def concat_turn_bin_prefix(prompt_embeds: torch.Tensor, turn_bin_prefix: torch.Tensor) -> torch.Tensor:
    prompt_embeds = ensure_batched_prompt_embeds(prompt_embeds)
    if turn_bin_prefix.ndim == 2:
        turn_bin_prefix = turn_bin_prefix.unsqueeze(0)
    if turn_bin_prefix.shape[0] != prompt_embeds.shape[0]:
        raise ValueError(
            f"Batch mismatch for prompt embeds {tuple(prompt_embeds.shape)} and bin prefix {tuple(turn_bin_prefix.shape)}"
        )
    return torch.cat([turn_bin_prefix.to(prompt_embeds.dtype), prompt_embeds], dim=1)


def build_conditioned_prompt_embeds(
    prompt_embeds: torch.Tensor,
    bin_ids: torch.Tensor,
    embedding_bank: TurnBinEmbeddingBank,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
) -> ConditionedEmbeddings:
    prompt_embeds = ensure_batched_prompt_embeds(prompt_embeds)
    turn_bin_prefix = embedding_bank(bin_ids.to(prompt_embeds.device))
    conditioned = concat_turn_bin_prefix(prompt_embeds, turn_bin_prefix)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = ensure_batched_prompt_embeds(negative_prompt_embeds)
    return ConditionedEmbeddings(prompt_embeds=conditioned, negative_prompt_embeds=negative_prompt_embeds)
