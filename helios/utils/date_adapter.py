from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class _DateLiteState:
    seed: int = 0
    _proj_cache: dict[tuple[int, int, str, int | None], torch.Tensor] = field(default_factory=dict)

    def get_projection(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (in_dim, out_dim, device.type, device.index)
        proj = self._proj_cache.get(key)
        if proj is not None and proj.device == device and proj.dtype == dtype:
            return proj

        # Deterministic projection that does not affect global RNG state.
        # Created on CPU then moved to the target device for broad compatibility.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + (in_dim * 1009) + (out_dim * 9176))
        proj_cpu = torch.randn((in_dim, out_dim), generator=gen, device="cpu", dtype=torch.float32)
        proj_cpu = proj_cpu / (proj_cpu.norm(dim=0, keepdim=True) + 1e-8)

        proj = proj_cpu.to(device=device, dtype=dtype)
        self._proj_cache[key] = proj
        return proj


_DATE_LITE_STATE = _DateLiteState()


def update_text_embeddings(
    text_emb: torch.Tensor,
    *,
    noise_pred: torch.Tensor,
    latents: torch.Tensor,
    scheduler,
    strength: float,
    step: int,
    update_freq: int,
    state: _DateLiteState = _DATE_LITE_STATE,
) -> torch.Tensor:
    """
    DATE-lite: a small, inference-only, no-grad update to text embeddings.

    - Video-aware: aggregates across the temporal dimension (frames) when present.
    - Stable: small additive update + bounded activation + detach.
    """
    if update_freq <= 0:
        return text_emb
    if (step % update_freq) != 0:
        return text_emb
    if strength <= 0:
        return text_emb

    # Convert model output to an x0 prediction using the scheduler's current step.
    # Avoid passing deprecated `timestep` by providing `sigma` explicitly when possible.
    sigma = None
    try:
        step_index = getattr(scheduler, "step_index", None)
        sigmas = getattr(scheduler, "sigmas", None)
        if step_index is not None and sigmas is not None:
            sigma = sigmas[step_index]
    except Exception:
        sigma = None

    try:
        x0_pred = scheduler.convert_model_output(noise_pred, sample=latents, sigma=sigma)
    except Exception:
        # If conversion fails for any reason, skip the update (non-destructive behavior).
        return text_emb

    # Aggregate video/space into a per-sample feature vector.
    x0_f32 = x0_pred.float()
    if x0_f32.ndim == 5:
        # [B, C, F, H, W] -> [B, C]
        feature = x0_f32.mean(dim=(2, 3, 4))
    elif x0_f32.ndim == 4:
        # [B, C, H, W] -> [B, C]
        feature = x0_f32.mean(dim=(2, 3))
    else:
        feature = x0_f32.flatten(start_dim=2).mean(dim=2)

    # Normalize for stability.
    feature = feature - feature.mean(dim=1, keepdim=True)
    feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-6)

    embed_dim = text_emb.shape[-1]
    proj = state.get_projection(
        in_dim=feature.shape[1],
        out_dim=embed_dim,
        device=text_emb.device,
        dtype=torch.float32,
    )
    update_vec = torch.tanh(feature @ proj)  # [B, D] in fp32

    # Broadcast over sequence length: [B, 1, D] -> [B, S, D]
    update_seq = update_vec[:, None, :].to(dtype=text_emb.dtype)
    if text_emb.ndim != 3:
        return text_emb
    update_seq = update_seq.expand(-1, text_emb.shape[1], -1)

    updated = (text_emb + (float(strength) * update_seq)).detach()
    return updated

