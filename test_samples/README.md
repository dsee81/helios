# Test Samples (GEPA Loop Prompt Opt)

This folder contains a tiny dataset + seed template you can use to run a quick end-to-end smoke test of the GEPA prompt optimization pipeline.

## Files

- `test_samples/loop_dataset.json`: two local MP4 inputs from `eval/playground/toy-video/`
- `test_samples/seed_loop_template.json`: a fixed-structure loop template that uses `{initial_prompt}`

## Run (end-to-end)

```bash
python -m gepa_prompt_opt.cli optimize-loop \
  --repo_root . \
  --dataset_manifest test_samples/loop_dataset.json \
  --seed_template test_samples/seed_loop_template.json \
  --work_dir runs/test_samples_loop_01 \
  --num_iterations 1 \
  --candidates_per_iteration 1 \
  --num_frames 240
```

Notes:
- This will run actual v2v inference + evaluation, so it requires Helios + eval dependencies to be installed and available.
- GEPA must also be installed/configured in the environment running the command.
- `test_samples/loop_dataset.json` uses `"path_base": ".."` so that paths are resolved relative to the repo root.
