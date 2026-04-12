from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .actions import NUM_ACTIONS


class ActionPrefixBank(nn.Module):
    def __init__(self, prefix_length: int, hidden_size: int = 4096, init_std: float = 0.02):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.prefix = nn.Parameter(torch.randn(NUM_ACTIONS, prefix_length, hidden_size) * init_std)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        if action_ids.ndim != 1:
            action_ids = action_ids.view(-1)
        return self.prefix[action_ids]


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


def concat_action_prefix(prompt_embeds: torch.Tensor, action_prefix: torch.Tensor) -> torch.Tensor:
    prompt_embeds = ensure_batched_prompt_embeds(prompt_embeds)
    if action_prefix.ndim == 2:
        action_prefix = action_prefix.unsqueeze(0)
    if action_prefix.shape[0] != prompt_embeds.shape[0]:
        raise ValueError(
            f"Batch mismatch for prompt embeds {tuple(prompt_embeds.shape)} and prefix {tuple(action_prefix.shape)}"
        )
    return torch.cat([action_prefix.to(prompt_embeds.dtype), prompt_embeds], dim=1)


def build_conditioned_prompt_embeds(
    prompt_embeds: torch.Tensor,
    action_ids: torch.Tensor,
    prefix_bank: ActionPrefixBank,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
) -> ConditionedEmbeddings:
    prompt_embeds = ensure_batched_prompt_embeds(prompt_embeds)
    action_prefix = prefix_bank(action_ids.to(prompt_embeds.device))
    conditioned = concat_action_prefix(prompt_embeds, action_prefix)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = ensure_batched_prompt_embeds(negative_prompt_embeds)
    return ConditionedEmbeddings(prompt_embeds=conditioned, negative_prompt_embeds=negative_prompt_embeds)
