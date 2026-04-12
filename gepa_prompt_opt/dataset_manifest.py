from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json
from .schema_validate import validate_with_jsonschema


@dataclass(frozen=True)
class DatasetVideo:
    id: int
    input_video_path: Path
    initial_prompt: str
    duration_seconds: float
    task_type: str
    split: str
    variables: dict[str, Any]


@dataclass(frozen=True)
class LoopDataset:
    manifest_path: Path
    path_base: Path
    default_task_type: str
    videos: list[DatasetVideo]


def _resolve_path(path_base: Path, p: str) -> Path:
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate
    return (path_base / candidate).resolve()


def load_loop_dataset(repo_root: str | Path, manifest_path: str | Path) -> LoopDataset:
    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()

    data = read_json(manifest_path)
    schema_path = repo_root / "gepa_prompt_opt" / "schemas" / "loop_dataset_manifest.schema.json"
    validate_with_jsonschema(data, schema_path)

    if data.get("task") != "loop":
        raise ValueError(f"Dataset manifest task must be 'loop'. Got: {data.get('task')!r}")

    path_base = data.get("path_base")
    base_dir = (manifest_path.parent / Path(path_base)).resolve() if path_base else manifest_path.parent.resolve()
    default_task_type = str(data.get("default_task_type") or "loop")

    videos: list[DatasetVideo] = []
    seen_ids: set[int] = set()
    for entry in data["videos"]:
        vid = int(entry["id"])
        if vid in seen_ids:
            raise ValueError(f"Duplicate id in dataset manifest: {vid}")
        seen_ids.add(vid)

        input_video_path = _resolve_path(base_dir, entry["input_video_path"])
        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found for id={vid}: {input_video_path}")

        initial_prompt = str(entry["initial_prompt"])
        duration_seconds = float(entry["duration_seconds"])
        task_type = str(entry.get("task_type") or default_task_type)
        split = str(entry.get("split") or "train")

        variables = dict(entry.get("variables") or {})
        variables.setdefault("initial_prompt", initial_prompt)
        variables.setdefault("duration_seconds", duration_seconds)

        videos.append(
            DatasetVideo(
                id=vid,
                input_video_path=input_video_path,
                initial_prompt=initial_prompt,
                duration_seconds=duration_seconds,
                task_type=task_type,
                split=split,
                variables=variables,
            )
        )

    videos.sort(key=lambda v: v.id)
    return LoopDataset(
        manifest_path=manifest_path,
        path_base=base_dir,
        default_task_type=default_task_type,
        videos=videos,
    )
