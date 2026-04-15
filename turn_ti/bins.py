from __future__ import annotations

from typing import Dict, List


DEFAULT_TURN_BIN_ORDER: List[str] = [
    "right_gentle",
    "right_strong",
    "straight_stable",
    "straight_wobbly",
    "left_gentle",
    "left_strong",
]

TURN_BIN_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(DEFAULT_TURN_BIN_ORDER)}
ID_TO_TURN_BIN: Dict[int, str] = {idx: name for name, idx in TURN_BIN_TO_ID.items()}
NUM_TURN_BINS = len(DEFAULT_TURN_BIN_ORDER)


def normalize_turn_bin_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in TURN_BIN_TO_ID:
        raise ValueError(f"Unsupported turn bin '{name}'. Expected one of {DEFAULT_TURN_BIN_ORDER}")
    return normalized
