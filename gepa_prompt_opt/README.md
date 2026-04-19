# GEPA Prompt Optimization (Loop Task)

This directory contains a modular **GEPA-based prompt-template optimization system** for Helios-style video generation.

It optimizes a **prompt template** (not per-video prompts) for the **`loop` / return-to-start navigation task** by:

1. Rendering a candidate template into per-video prompts (with variables).
2. Generating videos via the existing black-box script:
   - `scripts/inference/helios-distilled_v2v.sh` (not modified)
3. Evaluating generated videos using the existing evaluation pipeline outputs (not changing evaluation logic).
4. Aggregating results across videos into a scalar objective `J` + structured feedback for GEPA.
5. Iterating via GEPA to improve template wording while keeping template structure fixed.

## Layout

- `gepa_prompt_opt/schemas/`: JSON schemas (dataset + template).
- `gepa_prompt_opt/`: Python package modules (dataset/template/infer/eval/aggregate/GEPA/CLI).
- `gepa_prompt_opt/tests/`: Standard-library unit tests.

## What you need to do

### Required inputs you must provide

1. **Dataset manifest JSON** (duration in seconds) pointing to your input videos + initial prompts.
   - Schema: `gepa_prompt_opt/schemas/loop_dataset_manifest.schema.json`
   - Minimum per video: `id`, `input_video_path`, `initial_prompt`, `duration_seconds`
   - Example: `gepa_prompt_opt/examples/loop_dataset_example.json`
2. **Seed template JSON** matching the fixed template structure.
   - Schema: `gepa_prompt_opt/schemas/loop_prompt_template.schema.json`
   - Example: `gepa_prompt_opt/examples/seed_loop_template.json`

### Environment prerequisites

3. **Helios v2v inference must work**
   - The optimizer invokes v2v by *parsing* `scripts/inference/helios-distilled_v2v.sh` and overriding:
     - `--video_path`, `--prompt`, `--output_folder`, and (optionally) `--num_frames`
   - You must have the weights and runtime deps needed by `infer_helios.py`.

4. **Evaluation pipeline must work**
   - The optimizer calls `eval/run_metrics.sh` with environment-variable overrides.
   - You must have eval checkpoints present under `eval/checkpoints/...` per `eval/README.md`.
   - If you want **VLM judgments**, ensure:
     - `transformers` is installed on the target system
     - the local VLM path in `eval/failure_explanation/vlm_judges.py` exists on the target system

5. **GEPA must be installed and configured**
   - Install GEPA per its docs.
   - The current default reflection model is local Qwen via `local:/root/dataDisk/dsee_temp_storage/Qwen/Qwen3-32B`.
   - Set `GEPA_LOCAL_QWEN_PATH` if Qwen lives somewhere else.
   - Set `GEPA_LOCAL_LM_DEVICE` to place Qwen on a separate GPU from Helios inference.
   - DeepSeek is still available by passing `--reflection_lm deepseek/deepseek-chat` and setting `DEEPSEEK_API_KEY`.

## Checklist (before first run)

- Create `loop_dataset.json` from your videos (set `duration_seconds` in seconds).
- Decide whether to add extra `variables` per video (recommended: `scene`, `goal`, `route_hint`).
- Create `seed_template.json` and ensure it renders with `{initial_prompt}` at minimum.
- Verify you can run one v2v sample manually via `scripts/inference/helios-distilled_v2v.sh` / `infer_helios.py`.
- Verify eval works on a small set (metrics checkpoints present; VLM deps installed if enabled).
- Install GEPA (`pip install gepa`) and set your provider credentials (e.g. `OPENAI_API_KEY`) as required by GEPA.

## Quick commands

```bash
python -m gepa_prompt_opt.selfcheck --repo_root .
python -m unittest discover -s gepa_prompt_opt/tests
```

## Run optimization (GEPA)

```bash
python -m gepa_prompt_opt.cli optimize-loop \
  --repo_root . \
  --dataset_manifest path/to/loop_dataset.json \
  --seed_template path/to/seed_template.json \
  --work_dir runs/loop_gepa_01 \
  --num_iterations 10 \
  --candidates_per_iteration 4 \
  --num_frames 240 \
  --eval_split train \
  --disable_vlm \
  --reflection_lm local:/root/dataDisk/dsee_temp_storage/Qwen/Qwen3-32B
```

Artifacts are written under `--work_dir` as:
- `iter_###_cand_##_XXXXXXXXXXXXXXX/template.json`
- `iter_###_cand_##_XXXXXXXXXXXXXXX/gen/<id>.mp4`
- `iter_###_cand_##_XXXXXXXXXXXXXXX/eval/eval_input.csv`
- `iter_###_cand_##_XXXXXXXXXXXXXXX/eval_out/candidate/combined_video_report.json`
- `iter_###_cand_##_XXXXXXXXXXXXXXX/aggregate.json`

## Module responsibilities (what each section does)

- `gepa_prompt_opt/dataset_manifest.py`: loads the dataset manifest, resolves video paths, and provides per-video variables (always includes `initial_prompt` and `duration_seconds`).
- `gepa_prompt_opt/template.py`: validates fixed template structure and renders a per-video prompt via `str.format(**variables)`.
- `gepa_prompt_opt/helios_infer.py`: reads `scripts/inference/helios-distilled_v2v.sh`, extracts the `python infer_helios.py ...` invocation, and runs it with overridden `--video_path/--prompt/--output_folder` using the current Python interpreter.
- `gepa_prompt_opt/eval_pipeline.py`: runs `eval/run_metrics.sh` with environment variables and returns the path to `combined_video_report.json`.
- `gepa_prompt_opt/aggregate.py`: computes per-video score + failure penalty from the combined report; aggregates to objective `J` and GEPA-friendly `side_info`.
- `gepa_prompt_opt/gepa_driver.py`: orchestrates the full candidate evaluation and plugs it into GEPA.
- `gepa_prompt_opt/local_hf_lm.py`: local Hugging Face reflection backend for Qwen-style causal LMs.
- `gepa_prompt_opt/cli.py`: command line interface.

For the recovered Qwen-enhanced cyclic objective, see `gepa_prompt_opt/EXPLANATION.md`.
