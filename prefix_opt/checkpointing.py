from __future__ import annotations

import os
from typing import Optional

import torch

from .utils import ensure_dir


def save_prefix_checkpoint(output_dir: str, step: int, prefix_bank, optimizer=None, metadata: Optional[dict] = None) -> str:
    ensure_dir(output_dir)
    checkpoint_path = os.path.join(output_dir, f"prefix_checkpoint_{step:08d}.pt")
    payload = {
        "step": step,
        "prefix_bank": prefix_bank.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_prefix_checkpoint(checkpoint_path: str, prefix_bank, optimizer=None, map_location: str = "cpu") -> dict:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    prefix_bank.load_state_dict(payload["prefix_bank"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

