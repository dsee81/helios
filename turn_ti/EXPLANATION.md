# Turn-TI Tuner Explanation

Turn-TI is a textual-inversion-style tuner for Helios driving videos. It keeps the large Helios components frozen and learns a small turn-conditioned prompt embedding bank.

## What Is Being Trained

Only the `TurnBinEmbeddingBank` is trainable. Helios, the VAE, text encoder, tokenizer, scheduler, and video transformer are not finetuned.

The six learned turn bins are:

- `right_gentle`
- `right_strong`
- `straight_stable`
- `straight_wobbly`
- `left_gentle`
- `left_strong`

Each bin owns a small set of learned vectors. At training/inference time, the vectors for the selected bin are prepended to the normal frozen prompt embeddings. This gives Helios a learned control prefix for turn behavior.

## Initialization

The embedding bank is initialized from natural-language phrase embeddings created by the frozen Helios text-conditioning path.

Examples:

- `a driving scene with a gentle right turn`
- `a driving scene with a strong right turn`
- `a driving scene moving straight and stable`
- `a driving scene moving straight with slight wobble`
- `a driving scene with a gentle left turn`
- `a driving scene with a strong left turn`

Training learns a delta from these initial phrase-derived embeddings rather than starting from a purely random control bank.

## Training Data

The dataset is manifest-driven. Each row points to:

- source video path
- latent `.pt` payload path
- turn bin ID and name
- exact chunk frame range
- prompt text or fallback prompt

Latent payloads contain precomputed `prompt_embed` and `vae_latent` tensors. This avoids rerunning the text encoder and VAE preprocessing on every training step.

The current training path expects each sample to have at least two latent sections:

- one or more history sections
- one target section

The history sections condition Helios. The target section is the latent target the model tries to predict.

## Loss Function

The objective is a weighted sum:

```text
total_loss =
  reconstruction_weight * reconstruction
+ anchor_weight * anchor
+ neighbor_smoothness_weight * neighbor_smoothness
+ temporal_smoothness_weight * temporal_smoothness
```

Where:

- `reconstruction`: latent-space L1 loss between generated future latents and target future latents
- `anchor`: keeps learned embeddings close to phrase-derived initialization
- `neighbor_smoothness`: keeps adjacent turn-bin embeddings from becoming unrelated
- `temporal_smoothness`: regularizes generated latent motion over time

This is latent-space supervision, not RGB pixel reconstruction. That was changed to reduce memory pressure and align the loss with what Helios directly predicts.

## Validation Split

The recovered training code adds a leakage-safe validation split:

- default validation ratio: `0.1`
- split grouping key: source `video_path`
- result: chunks from the same video cannot appear in both train and validation

This matters because random row-level splitting would leak adjacent or related chunks across splits and overstate generalization.

## TensorBoard Logging

When `log_with: tensorboard` is enabled, training logs:

- `train/loss`
- `train/reconstruction`
- `train/anchor`
- `train/neighbor_smoothness`
- `train/temporal_smoothness`
- `train/lr`
- `val/loss`
- `val/reconstruction`
- `val/anchor`
- `val/neighbor_smoothness`
- `val/temporal_smoothness`

Validation runs every `val_every` steps. The full config currently sets `val_every: 100`.

## Checkpointing

Checkpoints save the learned embedding bank, optimizer state, step, and metadata. They do not save Helios model weights.

The trainer saves:

- periodic checkpoints: `turn_ti_checkpoint_XXXXXXXX.pt`
- best validation checkpoint: `turn_ti_checkpoint_best.pt`
- final checkpoint: `turn_ti_checkpoint_final.pt`

The "best" checkpoint is selected by lowest `val/loss`.

## Current Full Config Defaults

The restored full config uses:

```text
max_train_steps: 3000
save_every: 100
val_ratio: 0.1
val_split_seed: 43
val_every: 100
log_with: tensorboard
```

These are optimizer steps, not epochs. With batch size 1, one epoch is roughly the number of manifest rows in the training split.

## Important Caveats

This tuner gives Helios a small learned control prefix. It is intentionally lightweight. If turn behavior needs stronger control, the next step would be a more expressive adapter, LoRA, trajectory conditioning, or additional motion-aware losses.

The recovered repo has been statically checked in the current environment. Full training and inference still require the Helios weights, latent files, manifest, GPU access, and the expected Python environment.
