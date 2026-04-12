from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_text(json.dumps(data, sort_keys=True, ensure_ascii=False))
