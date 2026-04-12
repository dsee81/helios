from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_with_jsonschema(instance: Any, schema_path: str | Path) -> None:
    """
    Validate `instance` against a JSON schema file.

    Uses `jsonschema` if installed; otherwise performs no-op validation.
    (Downstream modules still do strict structural checks.)
    """
    schema_path = Path(schema_path)
    _ = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore
    except Exception:
        return
    jsonschema.validate(instance=instance, schema=_)
