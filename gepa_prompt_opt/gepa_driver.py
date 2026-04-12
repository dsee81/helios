from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregate import aggregate_candidate
from .deepseek_lm import DeepSeekClientConfig, make_deepseek_lm
from .dataset_manifest import load_loop_dataset
from .eval_pipeline import EvalPipelineConfig, run_eval_pipeline
from .helios_infer import HeliosV2VConfig, run_v2v_inference
from .io_utils import sha256_json, write_json
from .template import PromptTemplate, load_template, render_prompt


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
    num_frames: int = 240

    # Evaluation (fixed computation)
    eval_script_path: Path = Path("eval/run_metrics.sh")


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
        num_frames=cfg.num_frames,
        eval_script_path=_abs(repo_root, cfg.eval_script_path),
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
    tmpl = PromptTemplate(raw=template_data)

    helios_cfg = HeliosV2VConfig(repo_root=cfg.repo_root, inference_script_path=cfg.inference_script_path)

    rows: list[dict[str, Any]] = []
    for v in dataset.videos:
        prompt = render_prompt(tmpl, v.variables)
        out_mp4 = gen_dir / f"{v.id}.mp4"
        run_v2v_inference(
            helios_cfg,
            input_video_path=v.input_video_path,
            prompt=prompt,
            output_folder=gen_dir,
            output_mp4_path=out_mp4,
            num_frames=cfg.num_frames,
        )
        rows.append({"id": v.id, "prompt": prompt, "duration": v.duration_seconds, "video_path": str(out_mp4)})

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

    def evaluator(candidate_text: str) -> tuple[float, dict[str, Any]]:
        template_data = json.loads(candidate_text)
        cand_hash = sha256_json(template_data)[:16]
        candidate_name = f"cand_{cand_hash}"
        # GEPA may re-evaluate candidates; keep name stable and rely on caching in filesystem.
        print(f"Evaluating {candidate_name}")
        return evaluate_template_candidate(cfg, template_data=template_data, candidate_name=candidate_name)

    reflection_callable = make_deepseek_lm(DeepSeekClientConfig(model=cfg.reflection_lm))

    result = optimize_anything(
        seed_candidate=json.dumps(seed, ensure_ascii=False),
        evaluator=evaluator,
        objective="Optimize a general prompt template for loop/return-to-start navigation across many videos.",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=max_metric_calls, run_dir=str(cfg.work_dir / "gepa_state")),
            reflection=ReflectionConfig(reflection_lm=reflection_callable),
        ),
    )

    best = json.loads(result.best_candidate)
    write_json(cfg.work_dir / "best_template.json", best)
    return {"best_template_path": str(cfg.work_dir / "best_template.json"), "best_score": getattr(result, "best_score", None)}
