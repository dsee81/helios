import torch
import torch.nn.functional as F


def _safe_mean_pool_tokens(token_embeds: torch.Tensor) -> torch.Tensor:
    """
    token_embeds: [B, S, D] with zero-padded tokens.
    Returns: [B, D]
    """
    if token_embeds.ndim != 3:
        raise ValueError(f"Expected token_embeds to be rank-3 [B,S,D], got shape {tuple(token_embeds.shape)}")

    mask = (token_embeds.abs().sum(dim=-1) > 0).to(token_embeds.dtype)  # [B, S]
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
    pooled = (token_embeds * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
    return pooled


def _get_sigma_for_timestep(scheduler, timestep) -> torch.Tensor:
    """
    Returns sigma for the current step as a scalar tensor on the correct device.
    Falls back to scheduler.step_index when available.
    """
    if getattr(scheduler, "sigmas", None) is None:
        raise ValueError("Scheduler does not expose `sigmas`; cannot compute x0 prediction for DATE.")

    device = scheduler.sigmas.device
    if getattr(scheduler, "step_index", None) is not None and scheduler.step_index is not None:
        return scheduler.sigmas[scheduler.step_index].to(device=device)

    try:
        idx = scheduler.index_for_timestep(timestep)
        return scheduler.sigmas[idx].to(device=device)
    except Exception:
        # Conservative fallback for first step if scheduler index is not initialized.
        return scheduler.sigmas[0].to(device=device)


def update_text_embeddings_grad(
    text_emb: torch.Tensor,
    noise_pred: torch.Tensor,
    latents: torch.Tensor,
    timestep,
    *,
    scheduler,
    transformer,
    lr: float = 0.02,
    grad_clip_norm: float = 1.0,
    ema_decay: float = 0.9,
) -> torch.Tensor:
    """
    Gradient-based DATE update (inference-only, weights unchanged).

    Requirements:
    - Uses torch.autograd.grad (no global backward)
    - Updates only `text_emb`
    - Detaches updated embedding
    - Video-aware: aggregates across time dimension
    """
    if not text_emb.requires_grad:
        raise ValueError("`text_emb` must have requires_grad=True inside DATE update.")

    if lr <= 0:
        return text_emb.detach()

    # 1) Predict denoised sample x0 (latent-space)
    sigma = _get_sigma_for_timestep(scheduler, timestep).to(device=latents.device, dtype=latents.dtype)
    try:
        # Prefer scheduler's official conversion logic when available.
        x0_pred = scheduler.convert_model_output(noise_pred, sample=latents, sigma=sigma)
    except Exception:
        # Minimal fallback for the common Helios "flow_prediction": x0 = x_t - sigma * flow
        x0_pred = latents - sigma * noise_pred

    # 2) Compute pooled video feature (stable across frames)
    # patch_embedding: [B, inner_dim, T', H', W'] -> pool -> [B, inner_dim]
    patch_dtype = next(iter(transformer.patch_embedding.parameters())).dtype
    video_tokens = transformer.patch_embedding(x0_pred.to(dtype=patch_dtype))
    video_feat = video_tokens.mean(dim=(2, 3, 4))

    # 3) Compute pooled text feature in the same inner-dim space
    text_proj = transformer.condition_embedder.text_embedder(text_emb)
    text_feat = _safe_mean_pool_tokens(text_proj)

    # 4) Alignment loss: maximize cosine similarity
    video_feat = F.normalize(video_feat.float(), dim=-1, eps=1e-6)
    text_feat = F.normalize(text_feat.float(), dim=-1, eps=1e-6)
    loss = -F.cosine_similarity(video_feat, text_feat, dim=-1).mean()

    # 5) Gradient w.r.t. text embedding only
    grad = torch.autograd.grad(loss, text_emb, retain_graph=False, create_graph=False)[0]

    # 6) Clip gradients (norm <= grad_clip_norm), apply SGD step
    if grad_clip_norm is not None and grad_clip_norm > 0:
        grad_flat = grad.reshape(grad.shape[0], -1)
        grad_norm = torch.linalg.vector_norm(grad_flat, ord=2, dim=1, keepdim=True).clamp_min(1e-6)
        scale = (grad_clip_norm / grad_norm).clamp_max(1.0)
        grad = grad * scale.view(grad.shape[0], *([1] * (grad.ndim - 1)))

    old = text_emb.detach()
    updated = (text_emb - lr * grad).detach()

    if ema_decay is not None and 0.0 <= ema_decay < 1.0:
        updated = (ema_decay * old) + ((1.0 - ema_decay) * updated)

    return updated
