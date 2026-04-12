import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


_DEFAULT_FILENAME_GLOB = "*_*_ori*.mp4"


@dataclass(frozen=True)
class VideoExample:
    id: int
    video_path: str
    video_name: str
    duration: Optional[float] = None
    prompt: Optional[str] = None


def _resolve_video_path(raw_path: str, csv_path: str) -> str:
    if raw_path is None:
        return raw_path
    raw_path = str(raw_path).strip()
    if not raw_path:
        return raw_path
    path = Path(raw_path)
    if not path.is_absolute():
        path = (Path(csv_path).resolve().parent / path).resolve()
    return str(path)


def _parse_id_from_filename(filename: str) -> int:
    # Expected format: {id}_{target-duration}_{true-duration}.mp4 (per eval/README.md)
    # We only need the leading integer id.
    m = re.match(r"^(\d+)_", filename)
    if not m:
        raise ValueError(f"Cannot parse leading numeric id from filename: {filename}")
    return int(m.group(1))


def load_video_examples(
    input_csv: str,
    *,
    id_column: str = "id",
    duration_column: str = "duration",
    prompt_column: str = "prompt",
    video_path_column: Optional[str] = None,
    video_dir: Optional[str] = None,
    filename_glob: str = _DEFAULT_FILENAME_GLOB,
    require_prompt: bool = False,
) -> list[VideoExample]:
    """
    Load a list of VideoExample either:
      - from explicit video paths in the CSV (video_path_column), or
      - by scanning a directory for files and matching leading {id}_... against CSV ids (video_dir).

    CSV-relative paths are resolved relative to the CSV file's directory.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if id_column not in df.columns:
        raise ValueError(f"CSV must contain '{id_column}' column. Found columns: {df.columns.tolist()}")
    if duration_column and duration_column not in df.columns:
        raise ValueError(f"CSV must contain '{duration_column}' column. Found columns: {df.columns.tolist()}")
    if require_prompt and prompt_column not in df.columns:
        raise ValueError(f"CSV must contain '{prompt_column}' column. Found columns: {df.columns.tolist()}")

    df = df.copy()
    df[id_column] = df[id_column].astype(int)

    examples: list[VideoExample] = []

    if video_path_column:
        if video_path_column not in df.columns:
            raise ValueError(
                f"CSV must contain '{video_path_column}' column when --video_path_column is set. "
                f"Found columns: {df.columns.tolist()}"
            )

        for _, row in df.iterrows():
            vid = int(row[id_column])
            raw_path = row[video_path_column]
            video_path = _resolve_video_path(raw_path, input_csv)
            if not video_path or not os.path.exists(video_path):
                raise FileNotFoundError(f"Video path for id={vid} not found: {video_path!r}")

            prompt = str(row[prompt_column]) if prompt_column in df.columns else None
            duration = float(row[duration_column]) if duration_column in df.columns else None
            examples.append(
                VideoExample(
                    id=vid,
                    video_path=video_path,
                    video_name=os.path.basename(video_path),
                    duration=duration,
                    prompt=prompt,
                )
            )
        return examples

    if not video_dir:
        raise ValueError("Either video_path_column or video_dir must be provided")

    df_dict = df.set_index(id_column).to_dict("index")
    video_files = glob.glob(os.path.join(video_dir, filename_glob))
    video_files.sort(key=lambda x: _parse_id_from_filename(os.path.basename(x)))

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        try:
            vid = _parse_id_from_filename(video_name)
        except Exception:
            continue

        if vid not in df_dict:
            continue

        row = df_dict[vid]
        prompt = str(row.get(prompt_column)) if prompt_column in df.columns else None
        duration = float(row.get(duration_column)) if duration_column in df.columns else None
        examples.append(
            VideoExample(
                id=int(vid),
                video_path=video_path,
                video_name=video_name,
                duration=duration,
                prompt=prompt,
            )
        )

    return examples

