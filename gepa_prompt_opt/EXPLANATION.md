# GEPA Prompt Optimization Explanation

This module optimizes prompt templates for Helios loop-video generation. It does not train Helios weights. Instead, it searches for better prompt wording and structure, generates videos with Helios, scores the outputs, and feeds structured feedback back into GEPA so the next candidate prompt template can improve.

## What Is Being Optimized

The optimized object is a JSON prompt template with fixed structure:

- `role_instruction`
- `action_description`
- `loop_completion_requirement`
- `temporal_consistency_constraints`
- `scene_preservation_constraints`
- `negative_constraints`

GEPA is allowed to rewrite the text inside those fields, but the schema and render order stay fixed. This keeps prompt search controlled and makes candidates comparable.

## Main Flow

1. Load one or more seed templates.
2. Render the template against each video in the dataset manifest.
3. Run Helios V2V inference for each rendered prompt.
4. Run the evaluation pipeline on the generated videos.
5. Build a combined report with raw metrics and failure explanations.
6. Aggregate the report into a scalar objective `J` and structured `side_info`.
7. Ask the reflection model to propose a revised template.
8. Repeat for the configured GEPA budget.

The orchestration lives in `gepa_driver.py`.

## Local Qwen Reflection Model

The current recovered setup defaults to a local Hugging Face model rather than DeepSeek:

```text
local:/root/dataDisk/dsee_temp_storage/Qwen/Qwen3-32B
```

The local backend is implemented in `local_hf_lm.py`. It loads a causal LM with `transformers`, applies the tokenizer chat template when available, generates a reflection response, and strips `<think>...</think>` blocks before GEPA parses the candidate JSON.

Useful environment variables:

```bash
export GEPA_LOCAL_QWEN_PATH=/path/to/Qwen3-32B
export GEPA_LOCAL_LM_DEVICE=cuda:5
```

`GEPA_LOCAL_QWEN_PATH` changes the default model path. `GEPA_LOCAL_LM_DEVICE` controls which GPU the local reflection model uses. Helios inference GPU selection is separate and comes from the GEPA CLI `--inference_cuda_visible_devices` option.

DeepSeek is still available as a fallback by passing a non-`local:` model string, for example:

```bash
--reflection_lm deepseek/deepseek-chat
```

## Cyclic Loop Objective

The loop objective is intentionally not only "first frame versus last frame." A video can have a visually similar endpoint while failing to feel like a coherent loop, and a strict final-frame match can also be noisy.

The restored scoring path emphasizes:

- `loop_closure_score`: overall loop success estimate
- `cyclic_trajectory_score`: whether the whole path behaves like a cycle
- `revisit_near_end_score`: whether the video revisits the starting viewpoint near the end
- `seam_smoothness_score`: whether the ending can transition back to the start
- scene consistency and drift penalties

The VLM/failure explanation path computes these signals in `eval/failure_explanation/vlm_judges.py`, rule failures are handled in `failure_rules.py`, and GEPA aggregates them in `gepa_prompt_opt/aggregate.py`.

The per-video loop score weights are:

```text
0.30 * loop_closure_score
+ 0.25 * cyclic_trajectory_score
+ 0.20 * revisit_near_end_score
+ 0.10 * seam_smoothness_score
+ 0.10 * scene_consistency_score
+ 0.05 * low_drift_term
```

Loop closure failure is also penalized more heavily than before.

## Outputs

Each candidate directory contains:

- `template.json`: candidate template
- `rendered_prompts.json`: prompts rendered for each dataset video
- `gen/*.mp4`: Helios-generated videos
- `eval/eval_input.csv`: input CSV for evaluation
- `eval_out/candidate/gen/combined_video_report.json`: combined eval report
- `aggregate.json`: scalar `J` plus structured side information
- `candidate_summary.json`: paths and high-level failure summary

Run-level outputs include:

- `seed_bootstrap_summary.json`
- `gepa_state/`
- `best_template.json`
- `best_rendered_prompts.json`
- `best_score.json`

## Important Caveats

This optimization improves prompt templates only. If Helios does not strongly obey loop instructions, prompt optimization may plateau. For stronger loop control, combine this with reranking, endpoint conditioning, learned control tokens, or a loop-specific training objective.

The recovered repo has only been statically checked in the current environment. GPU generation and local Qwen loading require the model files and GPUs to be available again.
