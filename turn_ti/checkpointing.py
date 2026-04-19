from __future__ import annotations

import os
from typing import Optional

import torch

from prefix_opt.utils import ensure_dir


def save_turn_ti_checkpoint(
    output_dir: str,
    step: int,
    embedding_bank,
    optimizer=None,
    metadata: Optional[dict] = None,
    filename: Optional[str] = None,
) -> str:
    ensure_dir(output_dir)
    checkpoint_path = os.path.join(output_dir, filename or f"turn_ti_checkpoint_{step:08d}.pt")
    payload = {
        "step": step,
        "embedding_bank": embedding_bank.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_turn_ti_checkpoint(checkpoint_path: str, embedding_bank, optimizer=None, map_location: str = "cpu") -> dict:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    embedding_bank.load_state_dict(payload["embedding_bank"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
