from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .motion import MotionEstimates, drift_penalty, nondiff_motion_metrics, soft_motion_statistics, temporal_smoothness_loss


def pad_sequence_mean(values: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack([value.to(device=device).float().mean() for value in values], dim=0)


def sign_targets(values: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    pos = (values > threshold).float()
    neg = (values < -threshold).float()
    return pos - neg


def compute_prefix_losses(
    generated_video: torch.Tensor,
    source_video: torch.Tensor,
    velocity_targets: List[torch.Tensor],
    yaw_targets: List[torch.Tensor],
    action_ids: torch.Tensor,
    loss_cfg,
    global_step: int = 0,
) -> tuple[torch.Tensor, Dict[str, float], MotionEstimates]:
    common_frames = min(generated_video.shape[2], source_video.shape[2])
    generated_video = generated_video[:, :, :common_frames]
    source_video = source_video[:, :, :common_frames]

    estimates = soft_motion_statistics(generated_video, loss_cfg.velocity_scale, loss_cfg.yaw_scale)
    source_estimates = soft_motion_statistics(source_video, loss_cfg.velocity_scale, loss_cfg.yaw_scale)

    target_velocity = pad_sequence_mean(velocity_targets, generated_video.device) / max(loss_cfg.velocity_scale, 1e-6)
    target_yaw = pad_sequence_mean(yaw_targets, generated_video.device) / max(loss_cfg.yaw_scale, 1e-6)

    velocity_loss = F.smooth_l1_loss(estimates.signed_velocity, target_velocity)
    yaw_loss = F.smooth_l1_loss(estimates.yaw_rate, target_yaw)
    direction_loss = F.mse_loss(torch.tanh(estimates.signed_velocity), sign_targets(target_velocity))
    action_loss = F.cross_entropy(estimates.action_logits, action_ids)
    source_consistency = F.l1_loss(generated_video, source_video)
    source_motion = F.smooth_l1_loss(estimates.signed_velocity, source_estimates.signed_velocity) + F.smooth_l1_loss(
        estimates.yaw_rate, source_estimates.yaw_rate
    )
    smoothness = temporal_smoothness_loss(generated_video)
    drift = drift_penalty(generated_video)

    total = (
        velocity_loss * loss_cfg.velocity_weight
        + yaw_loss * loss_cfg.yaw_weight
        + direction_loss * loss_cfg.direction_weight
        + action_loss * loss_cfg.action_ce_weight
        + source_consistency * loss_cfg.source_consistency_weight
        + source_motion * loss_cfg.source_motion_weight
        + smoothness * loss_cfg.temporal_smoothness_weight
        + drift * loss_cfg.drift_weight
    )

    logs = {
        "loss": float(total.detach().item()),
        "velocity_loss": float(velocity_loss.detach().item()),
        "yaw_loss": float(yaw_loss.detach().item()),
        "direction_loss": float(direction_loss.detach().item()),
        "action_ce": float(action_loss.detach().item()),
        "source_consistency": float(source_consistency.detach().item()),
        "source_motion": float(source_motion.detach().item()),
        "temporal_smoothness": float(smoothness.detach().item()),
        "drift_penalty": float(drift.detach().item()),
        "pred_velocity": float(estimates.signed_velocity.mean().detach().item()),
        "pred_yaw": float(estimates.yaw_rate.mean().detach().item()),
        "target_velocity": float(target_velocity.mean().detach().item()),
        "target_yaw": float(target_yaw.mean().detach().item()),
    }

    if loss_cfg.use_non_diff_metrics and loss_cfg.non_diff_metrics_every > 0 and global_step % loss_cfg.non_diff_metrics_every == 0:
        logs.update(nondiff_motion_metrics(generated_video))

    return total, logs, estimates
