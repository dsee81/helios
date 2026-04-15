#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import pandas as pd


PER_VIDEO_SCORE_FIELD = {
    "aesthetic": "aesthetic_score",
    "motion_amplitude": "motion_fb",
    "motion_smoothness": "motion_smoothness_score",
    "semantic": "semantic_score",
    "naturalness": "naturalness_score",
    "drifting_aesthetic": "drift_aesthetic_score",
    "drifting_motion_smoothness": "drift_motion_smoothness_score",
    "drifting_semantic": "drift_semantic_score",
    "drifting_naturalness": "drift_naturalness_score",
}


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _resolve_video_path(raw_path: str, csv_path: str) -> str:
    if raw_path is None:
        return None
    raw_path = str(raw_path).strip()
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = (Path(csv_path).resolve().parent / p).resolve()
    return str(p)

def _infer_video_path_for_id(vid: int, video_path_per_id: dict, per_video_meta: dict) -> str | None:
    path = video_path_per_id.get(vid)
    if path:
        return path
    meta = per_video_meta.get(vid, {})
    for metric_info in meta.values():
        if isinstance(metric_info, dict) and metric_info.get("video_path"):
            return metric_info["video_path"]
    return None


def main(args: argparse.Namespace) -> int:
    experiment_output_dir = Path(args.experiment_output_dir)
    if not experiment_output_dir.exists():
        raise FileNotFoundError(f"experiment_output_dir not found: {experiment_output_dir}")

    df = pd.read_csv(args.input_csv)
    if args.id_column not in df.columns:
        raise ValueError(f"CSV must contain '{args.id_column}'. Found columns: {df.columns.tolist()}")
    if args.prompt_column and args.prompt_column not in df.columns:
        raise ValueError(f"CSV must contain '{args.prompt_column}'. Found columns: {df.columns.tolist()}")

    df = df.copy()
    df[args.id_column] = df[args.id_column].astype(int)

    task_type_per_id = {}
    if args.task_type_column and args.task_type_column in df.columns:
        task_type_per_id = dict(zip(df[args.id_column].tolist(), df[args.task_type_column].tolist()))

    video_path_per_id = {}
    if args.video_path_column and args.video_path_column in df.columns:
        for _, row in df.iterrows():
            vid = int(row[args.id_column])
            video_path_per_id[vid] = _resolve_video_path(row[args.video_path_column], args.input_csv)

    prompt_per_id = dict(zip(df[args.id_column].tolist(), df[args.prompt_column].tolist()))

    # Load metric result files
    results_files = sorted([p for p in experiment_output_dir.iterdir() if p.name.endswith("_results.json")])
    if not results_files:
        raise FileNotFoundError(f"No *_results.json files found in {experiment_output_dir}")

    per_video_metrics: dict[int, dict] = {}
    per_video_meta: dict[int, dict] = {}

    for rf in results_files:
        metric_key = rf.name.replace("_results.json", "")
        data = _load_json(rf)
        score_field = PER_VIDEO_SCORE_FIELD.get(metric_key)
        if not score_field:
            continue

        for item in data.get("per_video_results", []):
            vid = int(item["id"])
            per_video_metrics.setdefault(vid, {})
            per_video_meta.setdefault(vid, {})

            if score_field in item:
                per_video_metrics[vid][metric_key] = item[score_field]

            # Keep extra details (e.g., start/end drift components)
            per_video_meta[vid].setdefault(metric_key, {})
            for k, v in item.items():
                if k == "id":
                    continue
                per_video_meta[vid][metric_key][k] = v

    # Import failure explanation (keep sys.path local to eval/)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from failure_explanation import explain_failures

    combined = {
        "experiment_name": experiment_output_dir.name,
        "task_type_default": args.task_type,
        "input_csv": str(Path(args.input_csv).resolve()),
        "experiment_output_dir": str(experiment_output_dir.resolve()),
        "videos": [],
    }

    merged_results_path = experiment_output_dir / "merged_results.json"
    if merged_results_path.exists():
        combined["stage1_merged_results"] = _load_json(merged_results_path)

    all_ids = sorted(set(df[args.id_column].tolist()) | set(per_video_metrics.keys()))

    for vid in all_ids:
        raw_metrics = per_video_metrics.get(vid, {})

        # Map Helios metric names -> failure_explanation expected raw_metrics keys
        failure_raw_metrics = {}
        for k in [
            "semantic",
            "naturalness",
            "motion_amplitude",
            "motion_smoothness",
            "aesthetic",
            "drifting_aesthetic",
            "drifting_semantic",
            "drifting_naturalness",
            "drifting_motion_smoothness",
        ]:
            if k in raw_metrics:
                failure_raw_metrics[k] = raw_metrics[k]

        prompt_text = prompt_per_id.get(vid, "")
        task_type = task_type_per_id.get(vid, args.task_type)
        video_path = None if args.disable_vlm else _infer_video_path_for_id(vid, video_path_per_id, per_video_meta)

        report = None
        if failure_raw_metrics:
            report = explain_failures(failure_raw_metrics, task_type, prompt_text=prompt_text, video_path=video_path)
            if args.write_individual_reports:
                out_path = experiment_output_dir / f"failure_report_{vid}.json"
                with open(out_path, "w") as f:
                    json.dump(report, f, indent=2)

        combined["videos"].append(
            {
                "id": vid,
                "video_path": video_path,
                "prompt": prompt_text,
                "task_type": task_type,
                "raw_metrics": raw_metrics,
                "metric_details": per_video_meta.get(vid, {}),
                "failure_report": report,
            }
        )

    combined["num_videos"] = len(combined["videos"])

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = (experiment_output_dir / output_path).resolve()
    else:
        output_path = (experiment_output_dir / "combined_video_report.json").resolve()
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Wrote combined report: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a combined per-video report and run failure explanation.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--experiment_output_dir", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="reconstruct_original")

    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--video_path_column", type=str, default=None)
    parser.add_argument("--task_type_column", type=str, default=None)

    parser.add_argument("--write_individual_reports", action="store_true")
    parser.add_argument("--disable_vlm", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    raise SystemExit(main(args))
