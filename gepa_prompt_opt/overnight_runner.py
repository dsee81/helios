from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .gepa_driver import OptimizeConfig, optimize_with_gepa
from .io_utils import read_json, write_json


DEFAULT_INITIAL_PROMPTS = {
    "clip_4.mp4": (
        "A first-person walking video across a sunlit brick plaza on a campus walkway with pedestrians nearby. "
        "Move forward smoothly through the open walkway, make a gentle looping path, and return to the starting "
        "viewpoint by the end while keeping the same plaza layout, lighting, and people consistent."
    ),
    "clip_82.mp4": (
        "A first-person walking video through a tree-lined campus courtyard with orange benches, paving stones, "
        "and a few people crossing the scene. Move forward naturally, follow a smooth loop around the courtyard, "
        "and return to the original viewpoint by the end without changing the environment or lighting."
    ),
    "clip_98.mp4": (
        "A first-person walking video beside a campus service area with an event truck, nearby people, and "
        "buildings in the background. Move forward steadily through the open space, complete a loop, and end at "
        "the starting viewpoint while preserving the truck, crowd, and surrounding structures."
    ),
    "clip_112.mp4": (
        "A first-person walking video through a quiet gravel courtyard bordered by campus buildings, low stone "
        "planters, and trees. Move forward smoothly, trace a looping route through the courtyard, and return to "
        "the original viewpoint by the end while keeping the scene composition and lighting stable."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_manifest(video_dir: Path) -> dict:
    videos = []
    for idx, video_path in enumerate(sorted(video_dir.glob("*.mp4")), start=1):
        prompt = DEFAULT_INITIAL_PROMPTS.get(video_path.name)
        if not prompt:
            raise KeyError(f"No default prompt configured for {video_path.name}")
        videos.append(
            {
                "id": idx,
                "input_video_path": str(video_path.resolve()),
                "initial_prompt": prompt,
                "duration_seconds": 8.0,
                "split": "train",
                "variables": {
                    "scene": video_path.stem.replace("_", " "),
                    "goal": "return to the starting viewpoint at the end of the clip",
                    "route_hint": "move forward, trace a smooth loop, and end where the clip began",
                },
            }
        )
    if len(videos) != 4:
        raise ValueError(f"Expected exactly 4 videos in {video_dir}, found {len(videos)}")
    return {
        "version": "1.0",
        "task": "loop",
        "path_base": ".",
        "default_task_type": "loop",
        "videos": videos,
    }


def _latest_candidate_dir(work_dir: Path) -> str | None:
    candidates = sorted(
        [
            p
            for pattern in ("eval_*_cand_*", "cand_*")
            for p in work_dir.glob(pattern)
            if p.is_dir()
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    return str(candidates[-1].resolve())


def _write_status(status_path: Path, *, state: str, exit_code: int | None, work_dir: Path, error: str | None = None) -> None:
    payload = {
        "state": state,
        "exit_code": exit_code,
        "updated_at": _utc_now(),
        "last_completed_candidate": _latest_candidate_dir(work_dir),
    }
    if error:
        payload["error"] = error
    if status_path.exists():
        existing = read_json(status_path)
        if isinstance(existing, dict):
            existing.update(payload)
            payload = existing
    write_json(status_path, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and run the overnight GEPA loop optimization job.")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default="/mnt/shared_storage/dsee/Helios/gepa_inf_samples/gepa_samples")
    parser.add_argument("--seed_template", type=str, default="gepa_prompt_opt/examples/seed_loop_template.json")
    parser.add_argument("--reflection_lm", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--candidates_per_iteration", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(Path(args.video_dir).resolve())
    manifest_path = run_root / "loop_dataset.json"
    write_json(manifest_path, manifest)

    seed_template_src = Path(args.seed_template)
    if not seed_template_src.is_absolute():
        seed_template_src = (repo_root / seed_template_src).resolve()
    seed_template_data = read_json(seed_template_src)
    seed_template_path = run_root / "seed_template.json"
    write_json(seed_template_path, seed_template_data)

    run_config = {
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "video_dir": str(Path(args.video_dir).resolve()),
        "dataset_manifest": str(manifest_path),
        "seed_template": str(seed_template_path),
        "reflection_lm": args.reflection_lm,
        "cuda_visible_devices": args.cuda_visible_devices,
        "num_iterations": args.num_iterations,
        "candidates_per_iteration": args.candidates_per_iteration,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "disable_vlm": False,
        "run_naturalness": False,
        "logs": {
            "stdout": str((logs_dir / "runner.log").resolve()),
            "stderr": str((logs_dir / "runner.err").resolve()),
        },
    }
    write_json(run_root / "run_config.json", run_config)

    status_path = run_root / "status.json"
    write_json(
        status_path,
        {
            "state": "running",
            "exit_code": None,
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "last_completed_candidate": None,
        },
    )

    cfg = OptimizeConfig(
        repo_root=repo_root,
        work_dir=run_root,
        dataset_manifest=manifest_path,
        seed_template=seed_template_path,
        num_iterations=args.num_iterations,
        candidates_per_iteration=args.candidates_per_iteration,
        reflection_lm=args.reflection_lm,
        inference_cuda_visible_devices=args.cuda_visible_devices,
        num_frames=args.num_frames,
        fps=args.fps,
        eval_split="train",
        disable_vlm=False,
        run_naturalness=False,
    )

    try:
        result = optimize_with_gepa(cfg)
        _write_status(status_path, state="success", exit_code=0, work_dir=run_root)
        if isinstance(result, dict):
            write_json(run_root / "best_score.json", {**read_json(run_root / "best_score.json"), **result})
        return 0
    except Exception:
        _write_status(
            status_path,
            state="failed",
            exit_code=1,
            work_dir=run_root,
            error=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
