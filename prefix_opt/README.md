# Prefix Optimization for Helios V2V

This folder implements a prefix-only action-conditioning pipeline for Helios video-to-video generation.

The core idea is:

`E_final = concat(P_action, text_encoder(prompt))`

where:

- `P_action` is a trainable continuous prefix
- `text_encoder(prompt)` is the frozen Helios text embedding
- Helios model weights remain frozen
- only the prefix parameters are optimized

This README explains:

1. what each file does
2. how data is expected to look
3. the full training path
4. the loss functions
5. how backpropagation reaches the prefix
6. current implementation assumptions and caveats

## High-level architecture

The package is designed around one goal: learn a small action-conditioned embedding bank that can steer Helios V2V generation toward ego-motion behaviors like:

- `w`: forward
- `a`: turn left
- `s`: backward / reverse-like action bucket
- `d`: turn right
- `stop`: inferred from low-motion SCAND chunks

The implementation does not finetune:

- the Helios text encoder
- the Helios transformer
- the Helios VAE
- the Helios scheduler

Instead, it learns only a prefix tensor bank:

- shape: `[5, prefix_length, 4096]`
- 5 actions correspond to `w/a/s/d/stop`
- 4096 matches the Helios UMT5 text embedding width

This is intentionally lightweight, easy to checkpoint, and easy to verify for gradient flow.

## Folder overview

### [actions.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/actions.py)

Defines the canonical action vocabulary and mappings.

Main responsibilities:

- maps action strings to IDs
- centralizes the 5-action setup
- normalizes action names passed at inference time

Key mapping:

- `w -> 0`
- `a -> 1`
- `s -> 2`
- `d -> 3`
- `stop -> 4`

Why this matters:

- all dataset code
- loss code
- prefix lookup
- inference control

must agree on one action ordering.

### [utils.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/utils.py)

Small utility helpers used across the package.

Main responsibilities:

- seed setting
- directory creation
- canonical sample-key extraction from filenames
- image conversion helpers
- tensor-to-video conversion for non-differentiable metrics

Important function:

- `canonical_sample_key(...)`

This strips trailing metadata like frame count / resolution suffixes from latent filenames so that:

- video chunks
- latent `.pt` files
- SCAND `.csv` files

can be aligned using the same base sample identifier.

### [config.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/config.py)

Defines the configuration schema for the whole pipeline.

It splits config into logical sections:

- `PrefixDataConfig`
- `PrefixModelConfig`
- `PrefixGenerationConfig`
- `PrefixLossConfig`
- `PrefixTrainConfig`
- `PrefixInferenceConfig`

Main responsibilities:

- make the pipeline configurable without touching code
- keep dataset/model/generation/loss/training settings separate
- load YAML configs into typed dataclasses

This file is the contract for all runtime behavior.

### [conditioning.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/conditioning.py)

This is the core prefix-conditioning module.

Main responsibilities:

- define the trainable prefix bank
- gather the right prefix rows from action IDs
- concatenate prefix tokens to prompt embeddings

Main class:

- `ActionPrefixBank`

This holds:

- `self.prefix` with shape `[5, prefix_length, hidden_size]`

Main functions:

- `ensure_batched_prompt_embeds(...)`
- `concat_action_prefix(...)`
- `build_conditioned_prompt_embeds(...)`

The most important operation in the whole project happens here:

1. take frozen prompt embeddings from Helios or from precomputed latent files
2. gather the action prefix for each sample in the batch
3. concatenate prefix tokens in front of the text tokens

Result:

- original prompt shape: `[B, L, 4096]`
- prefix shape: `[B, P, 4096]`
- final conditioned prompt shape: `[B, P + L, 4096]`

This is the exact place where action-conditioning is introduced.

### [dataset.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/dataset.py)

This builds the V2V training dataset.

Main responsibilities:

- scan 4 action-specific video roots: `w/a/s/d`
- align each video chunk with:
  - one latent `.pt` file
  - one SCAND `.csv` chunk
- infer whether a chunk should actually be treated as `stop`
- return all tensors and metadata needed for training

Main class:

- `ActionLatentVideoDataset`

Each sample returns:

- `sample_id`
- `action_name`
- `action_id`
- `prompt_raw`
- `prompt_embed`
- `video_latent_sections`
- `first_frame_image`
- `source_video`
- `velocity`
- `yaw_rate`
- `csv_path`
- `video_path`
- `latent_path`

Important behavior:

1. It uses the latent files produced by `get_short-latents.py`
2. It loads `prompt_embed` directly from those files
3. It loads `vae_latent` directly from those files
4. It loads the source chunk video from disk for source-preservation and source-motion losses

#### Stop inference

`stop` is not stored as a fifth folder.

Instead, the dataset computes:

- `max(abs(v_calculated))`
- `max(abs(w_calculated))`

for each SCAND CSV chunk.

If both are below threshold, the sample is relabeled as `stop`.

This allows:

- a sample that physically lives under `w/a/s/d`
- but has effectively no motion

to be treated as `stop` during training.

#### Why both video and latents are loaded

The latent files are needed because the training path is designed around precomputed Helios latents.

The raw source video is still useful because:

- we want source-consistency losses
- we want source-motion regularization
- we may want non-differentiable motion metrics for evaluation

#### Collation

`collate_prefix_opt_batch(...)` builds a batch dictionary that:

- stacks action IDs
- concatenates prompt embeddings
- stacks latent sections
- stacks source videos
- keeps variable-length CSV-derived tensors as Python lists

The variable-length motion arrays stay as lists because clips may not all have the exact same CSV length.

### [motion.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/motion.py)

This file contains motion-estimation utilities used for both:

- differentiable training losses
- non-differentiable logging metrics

It has two layers of motion analysis.

#### 1. Differentiable motion proxy

Main function:

- `soft_motion_statistics(...)`

This is the training-time motion estimator.

It operates directly on generated frames using PyTorch ops only, so gradients can flow through it.

It estimates:

- `signed_velocity`
- `yaw_rate`
- `temporal_difference`
- `framewise_velocity`
- `framewise_yaw`
- `action_logits`

How it works:

1. converts RGB frames to grayscale
2. treats grayscale energy as a soft attention map
3. computes soft spatial statistics:
   - radial energy mean
   - horizontal energy mean
4. compares those statistics across adjacent frames

Interpretation:

- radial expansion / contraction is used as a proxy for forward or backward ego-motion
- horizontal centroid drift is used as a proxy for turning / yaw

This is not a full physical motion estimator, but it is differentiable and gives training a stable signal.

#### 2. Non-differentiable logging metrics

Functions:

- `_farneback_metrics_single(...)`
- `_visual_odometry_single(...)`
- `nondiff_motion_metrics(...)`

These use OpenCV to estimate:

- optical-flow magnitude
- mean horizontal flow
- approximate visual-odometry translation
- approximate visual-odometry rotation

Why they are separate:

- these metrics are helpful for monitoring
- but they are not used for backpropagation
- OpenCV operations break the computational graph

Also included:

- `temporal_smoothness_loss(...)`
- `drift_penalty(...)`

These are differentiable regularizers used in training.

### [losses.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/losses.py)

This file defines the training objective.

Main function:

- `compute_prefix_losses(...)`

Inputs:

- generated video
- source video
- velocity targets from SCAND
- yaw targets from SCAND
- action IDs
- loss configuration

Outputs:

- scalar total loss
- logging dictionary
- differentiable motion estimates

#### Loss terms

1. `velocity_loss`

- compares predicted differentiable signed velocity against the SCAND `v_calculated` mean
- uses `smooth_l1_loss`

2. `yaw_loss`

- compares predicted differentiable yaw proxy against the SCAND `w_calculated` mean
- uses `smooth_l1_loss`

3. `direction_loss`

- compares the sign of predicted velocity against the sign of target velocity
- intended to make “forward-ish” vs “reverse-ish” behavior separable

4. `action_ce`

- uses `action_logits` from the differentiable motion estimator
- classifies generated motion into 5 buckets
- trains the prefix to make the generated video motion pattern compatible with the target action class

5. `source_consistency`

- L1 frame reconstruction-style loss between generated video and source video
- acts as a preservation regularizer so the V2V result does not drift too far from the input chunk

6. `source_motion`

- compares generated differentiable motion statistics to source-video statistics
- this is a softer preservation term at the motion level

7. `temporal_smoothness`

- penalizes second-order frame differences
- discourages jitter and temporal noise

8. `drift_penalty`

- penalizes excessive frame-mean drift across time
- acts as a simple anti-collapse regularizer

#### Temporal alignment

Generated and source videos may have different lengths.

Before computing losses, the function aligns them by truncating both to the shortest shared temporal extent.

That avoids shape mismatches when:

- the source chunk is longer than the generated section
- or generation produces a smaller number of decoded frames

### [checkpointing.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/checkpointing.py)

Simple save/load utilities for the prefix-only checkpoints.

Main responsibilities:

- save prefix weights
- optionally save optimizer state
- load prefix weights back for resumed training or inference

Checkpoint payload contains:

- current step
- prefix bank state dict
- optional optimizer state
- optional metadata

This keeps checkpointing minimal and clean because Helios itself is frozen.

### [generator.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/generator.py)

This file wraps the frozen Helios model components and exposes generation helpers for training and inference.

Main class:

- `HeliosPrefixV2VGenerator`

Main responsibilities:

- load Helios tokenizer
- load Helios text encoder
- load Helios VAE
- load Helios transformer
- build Helios pipeline
- freeze all those components
- provide helper methods for prefix-conditioned V2V generation

Important helper:

- `flatten_video_latent_sections(...)`

The short-latent artifacts store chunk latents section-wise. This function reshapes:

- from `[B, S, C, T, H, W]`
- to `[B, C, S*T, H, W]`

so the Helios pipeline can consume them as a contiguous latent history.

#### Training generation path

Main method:

- `generate_training_video(...)`

What it does:

1. flatten precomputed latent sections
2. derive a prefix-frame proxy from the first latent slice
3. pad history latents if they are shorter than Helios history requirements
4. build the history tensors expected by Helios stage-1 sampling
5. sample one or more latent sections with `pipeline.stage1_sample(...)`
6. decode the generated latent sections using the frozen VAE

The important point is:

- this generation path is not wrapped in `torch.no_grad()`
- all Helios weights are frozen, but autograd still tracks operations
- so gradients can flow from frame loss back through the frozen network to the prefix embeddings

#### Inference generation path

Main method:

- `run_inference(...)`

For now this reuses the same internal training-generation route and returns a generated video tensor.

This keeps training and inference consistent:

- same prompt-conditioning behavior
- same latent preparation logic
- same frozen Helios backbone

### [train.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/train.py)

This is the main training entrypoint.

Run it with:

```bash
python -m prefix_opt.train --config prefix_opt/configs/prefix_opt_v1.yaml
```

Main responsibilities:

- load config
- create dataset and dataloader
- initialize Accelerate
- load frozen Helios wrapper
- create the trainable prefix bank
- optimize only the prefix bank
- save checkpoints
- log losses and grad norms

#### Training loop, step by step

For each batch:

1. load:
   - prompt embeddings
   - source videos
   - precomputed latent sections
   - SCAND motion arrays
   - action IDs

2. build conditioned prompt embeddings:
   - gather action prefix from `ActionPrefixBank`
   - concatenate prefix tokens with the frozen text embeddings

3. run `generate_training_video(...)`
   - Helios generates a new video conditioned on:
     - source latent history
     - source prefix-frame latent proxy
     - action-conditioned prompt embedding

4. compute loss:
   - compare generated video against motion targets and source regularizers

5. call `accelerator.backward(total_loss)`

6. inspect prefix gradient norm

7. if grad norm stays zero too long after warmup:
   - raise an error

8. clip gradients, step optimizer, step LR scheduler

9. log metrics

10. periodically save prefix checkpoints

#### Gradient verification

This script explicitly checks:

- whether `prefix_bank.prefix.grad` exists
- whether its norm is nonzero

This is important because the whole premise of the project depends on:

- gradients actually reaching the prefix
- while Helios weights stay frozen

If the prefix grad remains zero for too many steps after warmup, training aborts.

### [infer.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/infer.py)

This is the inference entrypoint.

Run it with:

```bash
python -m prefix_opt.infer --config prefix_opt/configs/prefix_opt_v1.yaml --checkpoint outputs/prefix_opt/prefix_checkpoint_00000100.pt --action w
```

Main responsibilities:

- load config
- load one dataset sample
- load prefix checkpoint
- choose target action
- build conditioned prompt embeddings
- generate a prefix-conditioned video
- export it to MP4

This is useful for:

- qualitative inspection
- comparing action outputs from the same source chunk
- sanity-checking checkpoint behavior

### [configs/prefix_opt_v1.yaml](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/configs/prefix_opt_v1.yaml)

This is the default config.

It controls:

- action video roots
- latent root
- CSV root
- stop thresholds
- Helios model path
- prefix length
- generation hyperparameters
- loss weights
- optimizer and logging settings

This is the first file to edit when adapting the pipeline to your machine and dataset layout.

### [tests/test_prefix_opt.py](C:/Users/davin/Desktop/dso/code/Helios/prefix_opt/tests/test_prefix_opt.py)

Lightweight smoke tests for the pure Python logic.

Current tests verify:

- stop inference logic
- prefix concatenation shape
- gradient reaches prefix while a frozen layer stays grad-free
- latent-section flattening shape

These tests do not require:

- full Helios weights
- diffusers runtime
- actual video loading

That makes them good sanity checks in limited environments.

## Expected data layout

The implementation assumes:

1. 4 action roots exist for videos:

- `w/`
- `a/`
- `s/`
- `d/`

2. precomputed short latents exist under a latent root

3. SCAND CSV chunks exist under a CSV root

The code tries to align them by canonical base filename.

Example idea:

```text
data/
  scand_chunks/
    w/
      sample_0001.mp4
    a/
      sample_0002.mp4
    s/
      sample_0003.mp4
    d/
      sample_0004.mp4
  scand_latents/
    w/
      sample_0001_121_384_640.pt
  scand_csv/
    w/
      sample_0001.csv
```

The latent and CSV roots may also be flat; the dataset supports both:

- `root/action_name/...`
- `root/...`

as long as canonical filenames still match.

## Full training process

This section describes the training path end-to-end.

### Step 1: prepare a batch

The dataloader returns:

- `prompt_embed`
- `video_latent_sections`
- `source_video`
- `velocity`
- `yaw_rate`
- `action_id`

At this stage:

- `prompt_embed` is frozen text conditioning from Helios preprocessing
- `video_latent_sections` is precomputed VAE latent history
- `source_video` is used only for regularization and monitoring

### Step 2: gather the action prefix

The prefix bank is indexed by `action_id`.

If batch action IDs are:

```text
[0, 3]
```

the prefix bank returns:

```text
[prefix_w, prefix_d]
```

with shape:

```text
[B, prefix_length, 4096]
```

### Step 3: concatenate prefix and prompt

If original prompt embedding shape is:

```text
[B, L, 4096]
```

then the conditioned prompt becomes:

```text
[B, prefix_length + L, 4096]
```

This tensor is what Helios receives as `encoder_hidden_states`.

### Step 4: run frozen Helios generation

The generator:

1. flattens latent sections
2. builds history latent tensors
3. calls Helios `stage1_sample(...)`
4. decodes generated latents through the frozen VAE

The result is:

- a generated video tensor in pixel space

### Step 5: estimate differentiable motion

The generated video is passed into `soft_motion_statistics(...)`.

This produces:

- signed velocity proxy
- yaw proxy
- action logits
- temporal statistics

These are differentiable with respect to the generated frames.

### Step 6: compare against SCAND targets

The SCAND chunk provides:

- `v_calculated`
- `w_calculated`

The implementation currently reduces each chunk to a mean target for:

- velocity
- yaw rate

Those targets supervise the differentiable motion estimates.

### Step 7: compute total loss

The total scalar loss is the weighted sum of:

- velocity adherence
- yaw adherence
- direction consistency
- action classification
- source consistency
- source motion consistency
- temporal smoothness
- drift penalty

### Step 8: backpropagate

This is the crucial part.

The code calls:

```python
accelerator.backward(total_loss)
```

Because:

- the generated video depends on Helios outputs
- Helios outputs depend on `encoder_hidden_states`
- `encoder_hidden_states` includes the action prefix

autograd can compute:

```text
d(loss) / d(prefix)
```

even though the Helios weights are frozen.

Freezing means:

- Helios parameters have `requires_grad=False`
- their weights are not updated

It does not mean:

- the forward path is detached

So the network still acts as a differentiable function from prefix to generated frames.

### Step 9: update only the prefix

Only the prefix bank is registered with the optimizer.

That means:

- gradients may flow through Helios
- but optimizer steps only change the prefix tensor bank

This is exactly the soft-prompt / prefix-tuning setup you requested.

## How backpropagation works here

This is the core ML-systems explanation.

Let:

- `P` = trainable action prefix
- `T(prompt)` = frozen text embedding
- `G(...)` = frozen Helios generator
- `M(...)` = differentiable motion estimator
- `L(...)` = scalar loss

The forward chain is:

```text
E_final = concat(P, T(prompt))
generated_video = G(source_latents, E_final)
motion_stats = M(generated_video)
loss = L(motion_stats, SCAND_targets, source_video)
```

Autograd computes:

```text
dL/dP = dL/dM * dM/dgenerated_video * dgenerated_video/dE_final * dE_final/dP
```

Important details:

- `dE_final/dP` is straightforward because `concat` is differentiable
- `dgenerated_video/dE_final` exists because Helios is used in a normal forward graph
- `dM/dgenerated_video` exists because the differentiable motion proxy uses PyTorch ops
- therefore `dL/dP` exists

This is why the differentiable motion proxy matters.

If the loss depended only on:

- OpenCV flow
- OpenCV visual odometry
- any detached NumPy pipeline

then gradients would not reach the prefix.

That is why the implementation currently uses:

- differentiable proxies for optimization
- OpenCV metrics only for monitoring / diagnostics

## Current limitations and caveats

This implementation is a solid v1 structure, but there are important caveats.

### 1. The motion loss is proxy-based

The differentiable motion estimator is not a true SLAM system.

It is a differentiable surrogate using image statistics.

Pros:

- stable gradients
- easy backprop
- low engineering overhead

Cons:

- not physically exact
- may not fully capture true ego-motion

This is the biggest gap between v1 and an eventual stronger system.

### 2. Non-differentiable flow / VO is logging only

OpenCV Farneback flow and the current visual-odometry estimate are used for metrics, not optimization.

That is because they break the computational graph.

### 3. The V2V “first-frame” conditioning is approximated

The implementation uses the first latent slice as a prefix-frame proxy.

If your preprocessing later provides a dedicated first-frame latent, the V2V path can be improved.

### 4. Inference currently reuses the training-generation path

This keeps behavior consistent, but it is not yet a polished production inference pipeline.

### 5. SCAND supervision is chunk-mean based

Right now the loss uses mean chunk-level motion targets.

A future version could use:

- frame-aligned supervision
- integrated trajectory losses
- more faithful time warping between generated frames and CSV timestamps

### 6. Multi-GPU is supported structurally, not benchmarked here

The code uses Accelerate and optimizes only the prefix bank, which is compatible with multi-GPU training.

But actual throughput and memory behavior still need to be validated on your target machine.

## Suggested next improvements

If you want to push this beyond the current v1, the best next steps are:

1. Replace or augment the differentiable motion proxy with a stronger differentiable ego-motion estimator
2. Add a more faithful trajectory loss using integrated `x/y/yaw`
3. Add a held-out evaluation script that compares no-prefix vs learned-prefix action adherence quantitatively
4. Add optional visualization dumps of:
   - predicted motion
   - target motion
   - source motion
   - prefix norm evolution
5. Add proper validation loops and metric aggregation across actions

## Commands

Training:

```bash
python -m prefix_opt.train --config prefix_opt/configs/prefix_opt_v1.yaml
```

Inference:

```bash
python -m prefix_opt.infer --config prefix_opt/configs/prefix_opt_v1.yaml --checkpoint outputs/prefix_opt/prefix_checkpoint_00000100.pt --action w
```

Tests:

```bash
python -m unittest prefix_opt.tests.test_prefix_opt
```

## Final summary

This package implements a frozen-Helios, prefix-only V2V action-control setup.

The trainable object is only the action prefix bank.

The training loop works by:

1. concatenating the action prefix to frozen prompt embeddings
2. generating a video through frozen Helios
3. computing differentiable motion-and-regularization losses on the generated frames
4. backpropagating through the frozen network into the prefix
5. updating only the prefix parameters

So conceptually, this is not finetuning Helios itself.

It is learning how to bias Helios’s conditioning space with a small action-specific continuous prompt.
