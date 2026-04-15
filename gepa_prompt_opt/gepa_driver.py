from __future__ import annotations

import csv
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregate import aggregate_candidate
from .deepseek_lm import DeepSeekClientConfig, make_deepseek_lm
from .dataset_manifest import load_loop_dataset
from .eval_pipeline import EvalPipelineConfig, run_eval_pipeline
from .helios_infer import HeliosV2VConfig, run_v2v_inference
from .io_utils import sha256_json, write_json
from .template import load_template, render_prompt, validate_template_data

FIXED_TEMPLATE_REFLECTION_PROMPT = """I am optimizing a JSON prompt template. The current template is:
```
<curr_param>
```

Below is evaluation data showing how this template performed:
```
<side_info>
```

The optimization goal is to improve a general prompt template for loop/return-to-start navigation across many videos.

Return an improved version of the template JSON, but obey all of these rules exactly:

1. Keep the top-level JSON structure exactly the same.
2. Keep the same top-level keys: `template_version`, `task`, `components`, `render`.
3. Keep `task` as `loop`.
4. Keep the `components` object with exactly these keys and no others:
   - `role_instruction`
   - `action_description`
   - `loop_completion_requirement`
   - `temporal_consistency_constraints`
   - `scene_preservation_constraints`
   - `negative_constraints`
5. You may only modify the text values of those existing component fields.
6. Do not add, remove, rename, or reorder component keys.
7. Keep `render.order` exactly unchanged.
8. Keep `render.separator` and `render.labels` unchanged.
9. Do not introduce any new fields such as extra guidance, notes, metadata, or comments.
10. Return only a valid JSON object inside ``` blocks.
"""


@dataclass(frozen=True)
class OptimizeConfig:
    repo_root: Path
    work_dir: Path
    dataset_manifest: Path
    seed_template: Path
    num_iterations: int = 10
    candidates_per_iteration: int = 4
    reflection_lm: str = "deepseek/deepseek-chat"

    # Inference (black-box script is parsed; not modified)
    inference_script_path: Path = Path("scripts/inference/helios-distilled_v2v.sh")
    inference_cuda_visible_devices: str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    num_frames: int = 240
    fps: int = 24
    eval_split: str = "train"

    # Evaluation (fixed computation)
    eval_script_path: Path = Path("eval/run_metrics.sh")
    disable_vlm: bool = False
    run_naturalness: bool = True


def _abs(repo_root: Path, p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def normalize_config(cfg: OptimizeConfig) -> OptimizeConfig:
    repo_root = cfg.repo_root.resolve()
    return OptimizeConfig(
        repo_root=repo_root,
        work_dir=_abs(repo_root, cfg.work_dir),
        dataset_manifest=_abs(repo_root, cfg.dataset_manifest),
        seed_template=_abs(repo_root, cfg.seed_template),
        num_iterations=cfg.num_iterations,
        candidates_per_iteration=cfg.candidates_per_iteration,
        reflection_lm=cfg.reflection_lm,
        inference_script_path=_abs(repo_root, cfg.inference_script_path),
        inference_cuda_visible_devices=str(cfg.inference_cuda_visible_devices),
        num_frames=cfg.num_frames,
        fps=cfg.fps,
        eval_split=cfg.eval_split,
        eval_script_path=_abs(repo_root, cfg.eval_script_path),
        disable_vlm=cfg.disable_vlm,
        run_naturalness=cfg.run_naturalness,
    )


def evaluate_template_candidate(
    cfg: OptimizeConfig,
    *,
    template_data: dict[str, Any],
    candidate_name: str,
) -> tuple[float, dict[str, Any]]:
    """
    Full candidate evaluation:
      - render prompts for all dataset videos
      - generate v2v outputs with Helios
      - run fixed evaluation pipeline
      - aggregate to objective J and side_info
    """
    cfg = normalize_config(cfg)
    cand_dir = cfg.work_dir / candidate_name
    gen_dir = cand_dir / "gen"
    eval_dir = cand_dir / "eval"
    eval_csv = eval_dir / "eval_input.csv"
    eval_out_base = cand_dir / "eval_out"

    cand_dir.mkdir(parents=True, exist_ok=True)
    write_json(cand_dir / "template.json", template_data)

    dataset = load_loop_dataset(cfg.repo_root, cfg.dataset_manifest)
    tmpl = validate_template_data(cfg.repo_root, template_data)
    selected_videos = dataset.videos if cfg.eval_split == "all" else [v for v in dataset.videos if v.split == cfg.eval_split]
    if not selected_videos:
        raise ValueError(f"No dataset videos matched eval_split={cfg.eval_split!r}")

    helios_cfg = HeliosV2VConfig(
        repo_root=cfg.repo_root,
        inference_script_path=cfg.inference_script_path,
        cuda_visible_devices=str(cfg.inference_cuda_visible_devices),
    )

    rows: list[dict[str, Any]] = []
    rendered_prompts: list[dict[str, Any]] = []
    for v in selected_videos:
        prompt = render_prompt(tmpl, v.variables)
        out_mp4 = gen_dir / f"{v.id}_{int(cfg.num_frames)}_ori_candidate.mp4"
        run_v2v_inference(
            helios_cfg,
            input_video_path=v.input_video_path,
            prompt=prompt,
            output_folder=gen_dir,
            output_mp4_path=out_mp4,
            num_frames=cfg.num_frames,
            fps=cfg.fps,
        )
        row = {"id": v.id, "prompt": prompt, "duration": v.duration_seconds, "video_path": str(out_mp4)}
        rows.append(row)
        rendered_prompts.append(
            {
                "id": v.id,
                "input_video_path": str(v.input_video_path),
                "prompt": prompt,
                "duration_seconds": v.duration_seconds,
                "video_path": str(out_mp4),
            }
        )

    write_json(cand_dir / "rendered_prompts.json", rendered_prompts)

    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "duration", "video_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    eval_cfg = EvalPipelineConfig(
        repo_root=cfg.repo_root,
        eval_script_path=cfg.eval_script_path,
        video_path_column="video_path",
        task_type="loop",
        disable_vlm=cfg.disable_vlm,
        run_naturalness=cfg.run_naturalness,
    )
    combined_report = run_eval_pipeline(
        eval_cfg,
        input_csv=eval_csv,
        base_output_dir=eval_out_base,
        experiment_name="candidate",
        dry_run=False,
    )

    j, side_info = aggregate_candidate(str(combined_report))
    write_json(cand_dir / "aggregate.json", {"J": j, "side_info": side_info})
    write_json(
        cand_dir / "candidate_summary.json",
        {
            "candidate_name": candidate_name,
            "J": j,
            "top_failure_modes": side_info.get("top_failure_modes", []),
            "paths": {
                "template": str((cand_dir / "template.json").resolve()),
                "rendered_prompts": str((cand_dir / "rendered_prompts.json").resolve()),
                "generated_videos_dir": str(gen_dir.resolve()),
                "eval_input_csv": str(eval_csv.resolve()),
                "eval_output_dir": str(eval_out_base.resolve()),
                "combined_report": str(combined_report.resolve()),
                "aggregate": str((cand_dir / "aggregate.json").resolve()),
            },
        },
    )
    return j, side_info


def optimize_with_gepa(cfg: OptimizeConfig) -> dict[str, Any]:
    """
    Uses GEPA to optimize the prompt template JSON.

    Requires:
      - `gepa` installed
      - GEPA LLM provider credentials configured (per GEPA docs)
    """
    cfg = normalize_config(cfg)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    seed = load_template(cfg.repo_root, cfg.seed_template).raw

    try:
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything  # type: ignore
    except Exception as e:
        raise RuntimeError("GEPA is not installed. Install it per GEPA docs to run optimization.") from e

    max_metric_calls = int(cfg.num_iterations) * int(cfg.candidates_per_iteration)
    eval_counter = itertools.count()

    def evaluator(candidate_text: str) -> tuple[float, dict[str, Any]]:
        try:
            template_data = json.loads(candidate_text)
            if not isinstance(template_data, dict):
                raise ValueError("Candidate must decode to a JSON object.")
            validate_template_data(cfg.repo_root, template_data)
        except Exception as e:
            return (
                -1.0,
                {
                    "objective": {
                        "J": -1.0,
                        "mean_score": 0.0,
                        "worst_p10_score": 0.0,
                        "variance": 0.0,
                        "mean_failure_penalty": 1.0,
                        "any_failure_rate": 1.0,
                    },
                    "candidate_error": str(e),
                },
            )
        eval_idx = next(eval_counter)
        cand_hash = sha256_json(template_data)[:16]
        candidate_name = f"eval_{eval_idx:04d}_cand_{cand_hash}"
        print(f"Evaluating {candidate_name}")
        return evaluate_template_candidate(cfg, template_data=template_data, candidate_name=candidate_name)

    reflection_callable = make_deepseek_lm(DeepSeekClientConfig(model=cfg.reflection_lm))

    result = optimize_anything(
        seed_candidate=json.dumps(seed, ensure_ascii=False),
        evaluator=evaluator,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=max_metric_calls,
                run_dir=str(cfg.work_dir / "gepa_state"),
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_callable,
                reflection_prompt_template=FIXED_TEMPLATE_REFLECTION_PROMPT,
            ),
        ),
    )

    best = json.loads(result.best_candidate)
    write_json(cfg.work_dir / "best_template.json", best)
    dataset = load_loop_dataset(cfg.repo_root, cfg.dataset_manifest)
    best_template = validate_template_data(cfg.repo_root, best)
    selected_videos = dataset.videos if cfg.eval_split == "all" else [v for v in dataset.videos if v.split == cfg.eval_split]
    best_rendered_prompts = [
        {
            "id": v.id,
            "input_video_path": str(v.input_video_path),
            "prompt": render_prompt(best_template, v.variables),
            "duration_seconds": v.duration_seconds,
        }
        for v in selected_videos
    ]
    write_json(cfg.work_dir / "best_rendered_prompts.json", best_rendered_prompts)
    write_json(
        cfg.work_dir / "best_score.json",
        {
            "best_score": getattr(result, "best_score", None),
            "best_template_path": str((cfg.work_dir / "best_template.json").resolve()),
            "best_rendered_prompts_path": str((cfg.work_dir / "best_rendered_prompts.json").resolve()),
        },
    )
    return {
        "best_template_path": str(cfg.work_dir / "best_template.json"),
        "best_rendered_prompts_path": str(cfg.work_dir / "best_rendered_prompts.json"),
        "best_score": getattr(result, "best_score", None),
    }
