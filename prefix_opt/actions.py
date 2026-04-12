from __future__ import annotations

from typing import Dict


ACTION_TO_ID: Dict[str, int] = {
    "w": 0,
    "a": 1,
    "s": 2,
    "d": 3,
    "stop": 4,
}

ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}
NUM_ACTIONS = len(ACTION_TO_ID)
DEFAULT_ACTION_ORDER = ["w", "a", "s", "d"]


def normalize_action_name(name: str) -> str:
    action = name.strip().lower()
    if action not in ACTION_TO_ID:
        raise ValueError(f"Unsupported action '{name}'. Expected one of {sorted(ACTION_TO_ID)}")
    return action

