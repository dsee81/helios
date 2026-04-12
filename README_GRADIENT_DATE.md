## Gradient-based DATE for Helios (Inference-only)

This README explains the **rationale**, **design choices**, and the **exact code path** for the gradient-based DATE
(Diffusion Adaptive Text Embedding) adaptation implemented in this duplicated repo: `Helios_DATE_GRAD/`.

This implementation is:
- **Inference-only** (no training code touched)
- **Weight-preserving** (model weights are never updated)
- **Opt-in** (default behavior is identical to upstream Helios)
- **Minimal / removable** (contained to a small adapter + a few plumbing changes)

---

## What “Gradient DATE” means here

DATE is a test-time method that adapts the *text conditioning* during diffusion sampling.

In this repo, “gradient DATE” means:
1. Encode the prompt once to get `text_embeddings_original`
2. Create a working copy `text_embeddings_current = text_embeddings_original.clone()`
3. During the diffusion denoising loop, every `date_update_freq` steps:
   - enable gradients **only** on `text_embeddings_current`
   - compute the model’s prediction for the current latent
   - define a lightweight alignment loss
   - use `torch.autograd.grad` to get `d(loss)/d(text_embeddings_current)`
   - apply one small gradient descent step to `text_embeddings_current`
   - **detach** the result and continue sampling

No gradients are enabled for latents (we do not optimize latents), and we explicitly disable gradients for the
transformer weights.

---

## Why we pool video + text features

Helios generates **video** latents with shape roughly `[B, C, T, H, W]`.

If we update text embeddings using a per-frame or per-patch loss, the gradient signal can become noisy and
inconsistent across time (different frames may “pull” the embedding in different directions). You explicitly required
that the loss be **video-aware** and “avoid per-frame independent updates”.

So we use **mean pooling** to produce a single, stable feature vector:
- pool video features over `(T, H, W)` so the update is driven by the whole clip, not any single frame
- pool text token features over `(S)` (masking padded tokens) to get a single prompt representation

This yields a single alignment objective per update step, which is much more stable.

---

## Are video and text features in the same space?

Raw prompt embeddings (e.g., T5 token embeddings) are not comparable to video/latent features.

To avoid comparing incompatible spaces, we project text embeddings into Helios’ **internal conditioning space** using
the same projection the model uses for conditioning:

- **Text side:** `transformer.condition_embedder.text_embedder(prompt_embeds)`
- **Video side:** features derived from `transformer.patch_embedding(x0_pred)`

Both produce vectors in the transformer’s **inner hidden dimension**, so cosine similarity is computed between vectors
of the same dimensionality and (crucially) the same internal model space.

This is not as semantically grounded as DATE’s CLIP/ImageReward objectives, but it is a minimal inference-only signal
that requires no extra dependencies (you explicitly requested no CLIP-based loss).

---

## Why cosine similarity

Cosine similarity is:
- scale-invariant (important in mixed precision and with varying activation magnitudes)
- simple and cheap
- provides a stable gradient direction for “make these two representations more aligned”

We minimize `-cosine(video_feat, text_feat)` (maximize alignment).

---

## Stability / safety measures

To meet your stability requirements and keep the change safe:
- **No global `.backward()`**: we use `torch.autograd.grad(...)` for a local gradient w.r.t. text embeddings only.
- **Grad clipping:** per-sample grad norm is clipped to ≤ `1.0`.
- **Small LR:** default `date_lr=0.02`.
- **Detach after update:** the updated embedding is always detached before the next sampling step.
- **No in-place edits:** the update returns a new tensor.
- **EMA smoothing (optional):** default `ema_decay=0.9` blends the update with the previous embedding to reduce jitter.
- **Weights frozen:** when DATE is enabled we call `self.transformer.requires_grad_(False)` to avoid tracking/allocating
  gradients for model parameters.

When `use_date_grad=False`, none of this runs (no extra compute or memory).

---

## What exactly was implemented

### 1) A small adapter module

File: `helios/utils/date_gradient_adapter.py`

Key function: `update_text_embeddings_grad(...)`

It:
- predicts denoised latents `x0_pred` from `(latents, noise_pred, sigma)`
  - prefers `scheduler.convert_model_output(...)` when available
  - falls back to `x0_pred = latents - sigma * noise_pred` if needed
- extracts a pooled video feature via `transformer.patch_embedding(x0_pred).mean((2,3,4))`
- extracts a pooled text feature via `transformer.condition_embedder.text_embedder(text_emb)` then masked mean over tokens
- computes `loss = -cosine(video_feat, text_feat)`
- computes gradient w.r.t. `text_emb` only using `torch.autograd.grad`
- clips, steps, EMA-smooths, detaches, returns the updated embedding

### 2) Hooking into Helios’ sampling loops

File: `helios/diffusers_version/pipeline_helios_diffusers.py`

We added three new pipeline kwargs (all default off):
- `use_date_grad: bool = False`
- `date_lr: float = 0.02`
- `date_update_freq: int = 2`

We then integrated DATE updates into:
- `stage1_sample(...)`
- `stage2_sample(...)`

At each scheduled update step we:
- build `prompt_embeds_for_grad = prompt_embeds.detach().clone().requires_grad_(True)`
- run one conditional forward pass to get `noise_pred`
- call `update_text_embeddings_grad(...)` to get the new `prompt_embeds`
- detach and continue

### 3) CLI flags + logging + plumbing from the existing entrypoint

File: `infer_helios.py`

Flags:
- `--use_date_grad`
- `--date_lr`
- `--date_update_freq`

And the same can be enabled via environment variables (needed if you only run the fixed shell entrypoint):
- `HELIOS_USE_DATE_GRAD=1`
- `HELIOS_DATE_LR=0.02`
- `HELIOS_DATE_UPDATE_FREQ=2`

When enabled, `infer_helios.py` prints:
- `Gradient DATE enabled`
- `DATE learning rate: ...`
- `DATE update frequency: ...`

---

## How this differs from the official DATE repo

The official DATE repo (NeurIPS 2025) demonstrates DATE for **text-to-image** Stable Diffusion and uses external
text-conditioned evaluators (CLIP score or ImageReward) as the gradient signal.

In Helios, we intentionally avoid CLIP/ImageReward and instead use a lightweight internal alignment proxy that:
- is video-aware
- stays inference-only
- adds no heavy dependencies

---

## Quick usage

Using the required Helios entrypoint (no script changes):

```bash
export HELIOS_USE_DATE_GRAD=1
export HELIOS_DATE_LR=0.02
export HELIOS_DATE_UPDATE_FREQ=2
bash scripts/inference/helios-distilled_v2v.sh /path/to/input.mp4 "your prompt" ./output out_name
```

Direct Python (optional):

```bash
python infer_helios.py ... --use_date_grad --date_lr 0.02 --date_update_freq 2
```
