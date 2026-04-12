from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalPipelineConfig:
    repo_root: Path
    eval_script_path: Path  # eval/run_metrics.sh
    video_path_column: str = "video_path"
    task_type: str = "loop"


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
    env = dict(os.environ)
    env["INPUT_CSV"] = str(input_csv)
    env["BASE_OUTPUT_DIR"] = str(base_output_dir)
    env["EXPERIMENT_NAME"] = experiment_name
    env["TASK_TYPE"] = config.task_type
    env["VIDEO_PATH_COLUMN"] = config.video_path_column
    env["VIDEO_DIR"] = "."  # unused when VIDEO_PATH_COLUMN is set
    env["DRY_RUN"] = "1" if dry_run else "0"

    subprocess.run(["bash", str(config.eval_script_path)], env=env, cwd=str(config.eval_script_path.parent), check=True)

    combined = base_output_dir / experiment_name / "combined_video_report.json"
    if not dry_run and not combined.exists():
        raise FileNotFoundError(f"Expected combined report not found: {combined}")
    return combined
