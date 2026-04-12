## Helios Inference Modes: Normal / DATE-lite / Gradient DATE

This repo supports three **inference-only** modes for Helios video generation:

1) **Normal Helios** (default): no text-embedding adaptation  
2) **DATE-lite**: a small, stable, **no-grad** text embedding update during sampling  
3) **Gradient DATE**: **gradient-based** adaptation of text embeddings during sampling (no weight updates)

Both DATE modes:
- are **inference-only**
- **do not** modify model weights
- are **opt-in** and default to off
- are **mutually exclusive** (enable only one)

---

## Enabling modes (works with existing shell entrypoints)

If you are using a fixed shell entrypoint (e.g. `scripts/inference/helios-distilled_v2v.sh`) that does not forward
custom flags, enable DATE via environment variables:

### Normal (default)

```bash
unset HELIOS_USE_DATE HELIOS_USE_DATE_GRAD
```

### DATE-lite

```bash
export HELIOS_USE_DATE=1
export HELIOS_DATE_STRENGTH=0.02
export HELIOS_DATE_UPDATE_FREQ=2
```

### Gradient DATE

```bash
export HELIOS_USE_DATE_GRAD=1
export HELIOS_DATE_LR=0.02
export HELIOS_DATE_UPDATE_FREQ=2
```

If both `HELIOS_USE_DATE` and `HELIOS_USE_DATE_GRAD` are set, inference will error.

---

## Enabling modes (CLI flags)

When running `infer_helios.py` directly:

### DATE-lite

```bash
python infer_helios.py ... --use_date --date_strength 0.02 --date_update_freq 2
```

### Gradient DATE

```bash
python infer_helios.py ... --use_date_grad --date_lr 0.02 --date_update_freq 2
```

---

## What each mode does

### DATE-lite (no-grad)

Implemented in `helios/utils/date_adapter.py`.

At every `date_update_freq` diffusion steps, it:
- converts the model output to an `x0_pred` estimate via `scheduler.convert_model_output(...)`
- mean-pools `x0_pred` over frames/spatial dims to a per-sample feature vector
- projects that vector into text-embedding dimension using a **deterministic** cached random projection
- applies a small bounded additive update to the text embeddings and detaches

### Gradient DATE (autograd)

Implemented in `helios/utils/date_gradient_adapter.py`.

At every `date_update_freq` diffusion steps, it:
- enables gradients **only** on the current text embedding tensor
- predicts an `x0_pred` estimate from `(latents, noise_pred, sigma)`
- extracts pooled video features using Helios’ `transformer.patch_embedding(x0_pred)`
- projects text into the transformer’s internal conditioning space via
  `transformer.condition_embedder.text_embedder(prompt_embeds)`
- minimizes `-cosine(video_feat, text_feat)` with `torch.autograd.grad`
- applies grad clipping (norm ≤ 1.0), a small step `date_lr`, EMA smoothing, detaches

Model weights remain frozen (no optimizer, no parameter grads).

---

## Files involved

- `infer_helios.py` (flags/env plumbing, mutual exclusion, logging, passes args into `pipe(...)`)
- `helios/diffusers_version/pipeline_helios_diffusers.py` (hooks into `stage1_sample` + `stage2_sample`)
- `helios/utils/date_adapter.py` (DATE-lite implementation)
- `helios/utils/date_gradient_adapter.py` (gradient DATE implementation)
