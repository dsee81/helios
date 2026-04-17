from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from prefix_opt.motion import temporal_smoothness_loss


def anchor_loss(effective_bank: torch.Tensor, init_bank: torch.Tensor) -> torch.Tensor:
    effective_flat = effective_bank.reshape(effective_bank.shape[0], -1)
    init_flat = init_bank.reshape(init_bank.shape[0], -1)
    cosine = 1.0 - F.cosine_similarity(effective_flat, init_flat, dim=-1)
    l2 = (effective_bank - init_bank).pow(2).mean(dim=(1, 2))
    return (cosine + l2).mean()


def neighbor_smoothness_loss(effective_bank: torch.Tensor) -> torch.Tensor:
    if effective_bank.shape[0] < 2:
        return effective_bank.new_tensor(0.0)
    return (effective_bank[1:] - effective_bank[:-1]).pow(2).mean()


def compute_turn_ti_losses(
    generated_latents: torch.Tensor,
    target_latents: torch.Tensor,
    embedding_bank,
    loss_cfg,
) -> tuple[torch.Tensor, Dict[str, float]]:
    common_frames = min(generated_latents.shape[2], target_latents.shape[2])
    generated_latents = generated_latents[:, :, :common_frames]
    target_latents = target_latents[:, :, :common_frames]

    reconstruction = F.l1_loss(generated_latents, target_latents)
    bank = embedding_bank.effective_bank()
    anchor = anchor_loss(bank, embedding_bank.init_bank)
    neighbor = neighbor_smoothness_loss(bank)
    smoothness = temporal_smoothness_loss(generated_latents)

    total = (
        reconstruction * loss_cfg.reconstruction_weight
        + anchor * loss_cfg.anchor_weight
        + neighbor * loss_cfg.neighbor_smoothness_weight
        + smoothness * loss_cfg.temporal_smoothness_weight
    )
    logs = {
        "loss": float(total.detach().item()),
        "reconstruction": float(reconstruction.detach().item()),
        "anchor": float(anchor.detach().item()),
        "neighbor_smoothness": float(neighbor.detach().item()),
        "temporal_smoothness": float(smoothness.detach().item()),
    }
    return total, logs
