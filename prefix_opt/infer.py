from __future__ import annotations

import argparse

import torch
from diffusers.utils import export_to_video

from .actions import ACTION_TO_ID, normalize_action_name
from .checkpointing import load_prefix_checkpoint
from .conditioning import ActionPrefixBank, build_conditioned_prompt_embeds
from .config import load_config
from .dataset import ActionLatentVideoDataset
from .generator import HeliosPrefixV2VGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Run Helios prefix-only V2V inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_path = args.output or cfg.inference.output_path
    action_name = normalize_action_name(args.action)

    dataset = ActionLatentVideoDataset(
        action_dirs=cfg.data.action_dirs,
        latent_root=cfg.data.latent_root,
        csv_root=cfg.data.csv_root,
        video_glob=cfg.data.video_glob,
        latent_glob=cfg.data.latent_glob,
        csv_glob=cfg.data.csv_glob,
        prompt_fallback=cfg.data.prompt_fallback,
        num_frames=cfg.data.num_frames,
        height=cfg.data.height,
        width=cfg.data.width,
        cache_metadata=cfg.data.cache_metadata,
        force_rebuild=False,
        v_stop_thresh=cfg.data.v_stop_thresh,
        w_stop_thresh=cfg.data.w_stop_thresh,
        velocity_column=cfg.data.velocity_column,
        yaw_rate_column=cfg.data.yaw_rate_column,
    )
    sample = dataset[args.sample_index]
    generator = HeliosPrefixV2VGenerator(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    prefix_bank = ActionPrefixBank(
        prefix_length=cfg.model.prefix_length,
        hidden_size=generator.transformer.config.text_dim,
        init_std=cfg.model.prefix_init_std,
    ).to(generator.device)
    if args.checkpoint and not cfg.inference.disable_prefix:
        load_prefix_checkpoint(args.checkpoint, prefix_bank=prefix_bank, optimizer=None, map_location="cpu")

    prompt_embeds = sample["prompt_embed"].to(generator.device, dtype=generator.transformer.dtype)
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    negative_prompt_embeds = generator.encode_prompt_text([cfg.model.negative_prompt])
    action_ids = torch.tensor([ACTION_TO_ID[action_name]], device=generator.device, dtype=torch.long)
    conditioned = build_conditioned_prompt_embeds(
        prompt_embeds=prompt_embeds,
        action_ids=action_ids,
        prefix_bank=prefix_bank,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    generated_video = generator.run_inference(
        prompt_embeds=conditioned.prompt_embeds,
        negative_prompt_embeds=conditioned.negative_prompt_embeds,
        video_latent_sections=sample["video_latent_sections"].unsqueeze(0).to(generator.device),
    )
    frames = ((generated_video[0].detach().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    export_to_video(frames, output_path, fps=cfg.inference.fps)
    print(f"Saved prefix-conditioned video to {output_path}")


if __name__ == "__main__":
    main()
