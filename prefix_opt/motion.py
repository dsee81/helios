from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
import torch

from .utils import tensor_to_numpy_video


@dataclass
class MotionEstimates:
    signed_velocity: torch.Tensor
    yaw_rate: torch.Tensor
    temporal_difference: torch.Tensor
    framewise_velocity: torch.Tensor
    framewise_yaw: torch.Tensor
    action_logits: torch.Tensor


def rgb_to_gray(video: torch.Tensor) -> torch.Tensor:
    r, g, b = video[:, 0:1], video[:, 1:2], video[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def soft_motion_statistics(video: torch.Tensor, velocity_scale: float, yaw_scale: float) -> MotionEstimates:
    gray = rgb_to_gray(video)
    energy = gray.abs() + 1e-6
    _, _, _, height, width = energy.shape

    x_coords = torch.linspace(-1.0, 1.0, width, device=video.device, dtype=video.dtype).view(1, 1, 1, 1, width)
    y_coords = torch.linspace(-1.0, 1.0, height, device=video.device, dtype=video.dtype).view(1, 1, 1, height, 1)
    radial = torch.sqrt(x_coords.square() + y_coords.square())

    norm = energy.sum(dim=(-1, -2), keepdim=True) + 1e-6
    radial_mean = (energy * radial).sum(dim=(-1, -2), keepdim=False) / norm.squeeze(-1).squeeze(-1)
    x_mean = (energy * x_coords).sum(dim=(-1, -2), keepdim=False) / norm.squeeze(-1).squeeze(-1)
    temporal_difference = (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean(dim=(1, 2, 3, 4))

    framewise_velocity = (radial_mean[:, 1:] - radial_mean[:, :-1]).mean(dim=1) / max(velocity_scale, 1e-6)
    framewise_yaw = (x_mean[:, 1:] - x_mean[:, :-1]).mean(dim=1) / max(yaw_scale, 1e-6)

    signed_velocity = framewise_velocity
    yaw_rate = framewise_yaw

    stop_logit = -(signed_velocity.abs() + yaw_rate.abs())
    forward_logit = signed_velocity - yaw_rate.abs()
    left_logit = yaw_rate - signed_velocity.abs() * 0.25
    backward_logit = -signed_velocity - yaw_rate.abs()
    right_logit = -yaw_rate - signed_velocity.abs() * 0.25
    action_logits = torch.stack([forward_logit, left_logit, backward_logit, right_logit, stop_logit], dim=-1)

    return MotionEstimates(
        signed_velocity=signed_velocity,
        yaw_rate=yaw_rate,
        temporal_difference=temporal_difference,
        framewise_velocity=framewise_velocity,
        framewise_yaw=framewise_yaw,
        action_logits=action_logits,
    )


def temporal_smoothness_loss(video: torch.Tensor) -> torch.Tensor:
    if video.shape[2] < 3:
        return video.new_tensor(0.0)
    second_diff = video[:, :, 2:] - 2.0 * video[:, :, 1:-1] + video[:, :, :-2]
    return second_diff.abs().mean()


def drift_penalty(video: torch.Tensor) -> torch.Tensor:
    frame_mean = video.mean(dim=(1, 3, 4))
    centered = frame_mean - frame_mean.mean(dim=-1, keepdim=True)
    return centered.square().mean()


def _farneback_metrics_single(video_np: np.ndarray) -> Dict[str, float]:
    prev_gray = cv2.cvtColor(video_np[0], cv2.COLOR_RGB2GRAY)
    magnitudes = []
    mean_dx = []
    for frame in video_np[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitudes.append(float(np.linalg.norm(flow, axis=-1).mean()))
        mean_dx.append(float(flow[..., 0].mean()))
        prev_gray = gray
    return {
        "farneback_motion": float(np.mean(magnitudes)) if magnitudes else 0.0,
        "farneback_dx": float(np.mean(mean_dx)) if mean_dx else 0.0,
    }


def _visual_odometry_single(video_np: np.ndarray) -> Dict[str, float]:
    prev_gray = cv2.cvtColor(video_np[0], cv2.COLOR_RGB2GRAY)
    rotations = []
    translations = []
    for frame in video_np[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
        if pts_prev is None:
            prev_gray = gray
            continue
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)
        valid_prev = pts_prev[status.flatten() == 1]
        valid_next = pts_next[status.flatten() == 1]
        if len(valid_prev) < 6:
            prev_gray = gray
            continue
        matrix, _ = cv2.estimateAffinePartial2D(valid_prev, valid_next)
        if matrix is not None:
            dx = float(matrix[0, 2])
            dy = float(matrix[1, 2])
            theta = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
            translations.append(float(np.sqrt(dx * dx + dy * dy)))
            rotations.append(theta)
        prev_gray = gray
    return {
        "vo_translation": float(np.mean(translations)) if translations else 0.0,
        "vo_rotation": float(np.mean(rotations)) if rotations else 0.0,
    }


def nondiff_motion_metrics(video: torch.Tensor) -> Dict[str, float]:
    video_np = tensor_to_numpy_video(video)[0]
    metrics = {}
    metrics.update(_farneback_metrics_single(video_np))
    metrics.update(_visual_odometry_single(video_np))
    return metrics

