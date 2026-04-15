from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json
from .schema_validate import validate_with_jsonschema


COMPONENT_KEYS = [
    "role_instruction",
    "action_description",
    "loop_completion_requirement",
    "temporal_consistency_constraints",
    "scene_preservation_constraints",
    "negative_constraints",
]


@dataclass(frozen=True)
class PromptTemplate:
    raw: dict[str, Any]

    @property
    def components(self) -> dict[str, str]:
        return self.raw["components"]

    @property
    def render_cfg(self) -> dict[str, Any]:
        return self.raw["render"]


def validate_template_data(repo_root: str | Path, data: dict[str, Any]) -> PromptTemplate:
    repo_root = Path(repo_root).resolve()
    schema_path = repo_root / "gepa_prompt_opt" / "schemas" / "loop_prompt_template.schema.json"
    validate_with_jsonschema(data, schema_path)

    if data.get("task") != "loop":
        raise ValueError(f"Template task must be 'loop'. Got: {data.get('task')!r}")
    _validate_structure(data)
    return PromptTemplate(raw=data)


def load_template(repo_root: str | Path, template_path: str | Path) -> PromptTemplate:
    repo_root = Path(repo_root).resolve()
    template_path = Path(template_path)
    if not template_path.is_absolute():
        template_path = (repo_root / template_path).resolve()

    data = read_json(template_path)
    return validate_template_data(repo_root, data)


def _validate_structure(data: dict[str, Any]) -> None:
    comps = data.get("components", {})
    for k in COMPONENT_KEYS:
        if k not in comps or not isinstance(comps[k], str) or not comps[k].strip():
            raise ValueError(f"Template.components must contain a non-empty string for key: {k}")
    order = data.get("render", {}).get("order")
    if order != COMPONENT_KEYS:
        raise ValueError(f"Template.render.order must be exactly {COMPONENT_KEYS}. Got: {order!r}")


def render_prompt(template: PromptTemplate, variables: dict[str, Any]) -> str:
    """
    Render the prompt by concatenating fixed-structure components with placeholder substitution.
    Placeholders are Python `str.format` style, e.g. '{initial_prompt}'.
    """
    separator = template.render_cfg.get("separator", "\n\n")
    labels = bool(template.render_cfg.get("labels", False))
    parts: list[str] = []
    for key in template.render_cfg["order"]:
        text = template.components[key]
        try:
            text = text.format(**variables)
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Missing template variable: {missing}") from e
        if labels:
            parts.append(f"{key.replace('_', ' ').title()}: {text}")
        else:
            parts.append(text)
    return separator.join(parts).strip()
