from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
import csv


@dataclass(frozen=True)
class EvalPipelineConfig:
    repo_root: Path
    eval_script_path: Path  # eval/run_metrics.sh
    video_path_column: str = "video_path"
    task_type: str = "loop"
    disable_vlm: bool = False
    run_naturalness: bool = True


def run_eval_pipeline(
    config: EvalPipelineConfig,
    *,
    input_csv: Path,
    base_output_dir: Path,
    experiment_name: str,
    dry_run: bool = False,
) -> Path:
    """
    Runs the existing evaluation pipeline as a black box and returns the combined report JSON path.
    """
    resolved_video_dir = "."
    if config.video_path_column:
        try:
            with open(input_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
            if first and first.get(config.video_path_column):
                first_video = Path(str(first[config.video_path_column])).expanduser()
                if not first_video.is_absolute():
                    first_video = (input_csv.parent / first_video).resolve()
                resolved_video_dir = str(first_video.parent)
        except Exception:
            resolved_video_dir = "."

    env = dict(os.environ)
    env["INPUT_CSV"] = str(input_csv)
    env["BASE_OUTPUT_DIR"] = str(base_output_dir)
    env["EXPERIMENT_NAME"] = experiment_name
    env["TASK_TYPE"] = config.task_type
    env["VIDEO_PATH_COLUMN"] = config.video_path_column
    env["VIDEO_DIR"] = resolved_video_dir
    env["DRY_RUN"] = "1" if dry_run else "0"
    env["DISABLE_VLM"] = "1" if config.disable_vlm else "0"
    env["RUN_NATURALNESS"] = "1" if config.run_naturalness else "0"
    env["PYTHON_BIN"] = sys.executable

    subprocess.run(["bash", str(config.eval_script_path)], env=env, cwd=str(config.eval_script_path.parent), check=True)

    combined = base_output_dir / experiment_name / Path(resolved_video_dir).name / "combined_video_report.json"
    if not dry_run and not combined.exists():
        raise FileNotFoundError(f"Expected combined report not found: {combined}")
    return combined
