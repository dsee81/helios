from __future__ import annotations

import os
import random
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonical_sample_key(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 4 and all(p.isdigit() for p in parts[-3:]):
        return "_".join(parts[:-3])
    return stem


def find_preferred_root(root: str, action_name: str) -> str:
    candidate = os.path.join(root, action_name)
    return candidate if os.path.isdir(candidate) else root


def tensor_to_numpy_video(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got {tuple(video.shape)}")
    video = video.detach().clamp(-1, 1)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    return video.permute(0, 2, 3, 4, 1).cpu().numpy()


def maybe_to_pil(image_obj) -> Optional[Image.Image]:
    if image_obj is None:
        return None
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if torch.is_tensor(image_obj):
        array = image_obj.detach().cpu()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.permute(1, 2, 0)
        return Image.fromarray(array.numpy().astype(np.uint8)).convert("RGB")
    if isinstance(image_obj, np.ndarray):
        return Image.fromarray(image_obj.astype(np.uint8)).convert("RGB")
    return None


def cycle_items(items: Iterable[str], fallback: str) -> str:
    for item in items:
        return item
    return fallback
