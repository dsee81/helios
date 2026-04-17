from __future__ import annotations

import argparse

import torch
from diffusers.utils import export_to_video

from prefix_opt.generator import HeliosPrefixV2VGenerator, split_history_target_sections

from .bins import TURN_BIN_TO_ID, normalize_turn_bin_name
from .checkpointing import load_turn_ti_checkpoint
from .conditioning import TurnBinEmbeddingBank, build_conditioned_prompt_embeds
from .config import load_config
from .dataset import TurnBinLatentVideoDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run Helios turn-bin textual inversion inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_path = args.output or cfg.inference.output_path
    bin_name = normalize_turn_bin_name(args.bin)

    dataset = TurnBinLatentVideoDataset(
        manifest_path=cfg.data.manifest_path,
        prompt_fallback=cfg.data.prompt_fallback,
        num_frames=cfg.data.num_frames,
        height=cfg.data.height,
        width=cfg.data.width,
        load_source_video=cfg.data.load_source_video,
        cache_metadata=cfg.data.cache_metadata,
        force_rebuild=False,
        strict_paths=cfg.data.strict_paths,
    )
    sample = dataset[args.sample_index]
    generator = HeliosPrefixV2VGenerator(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    embedding_bank = TurnBinEmbeddingBank.from_phrases(
        generator=generator,
        init_phrases=cfg.model.init_phrases,
        num_vectors=cfg.model.num_vectors,
        init_scale=cfg.model.init_scale,
        learnable_delta=cfg.model.learnable_delta,
    ).to(generator.device)
    if args.checkpoint and not cfg.inference.disable_bank:
        load_turn_ti_checkpoint(args.checkpoint, embedding_bank=embedding_bank, optimizer=None, map_location="cpu")

    prompt_embeds = sample["prompt_embed"].to(generator.device, dtype=generator.transformer.dtype)
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    negative_prompt_embeds = generator.encode_prompt_text([cfg.model.negative_prompt])
    bin_ids = torch.tensor([TURN_BIN_TO_ID[bin_name]], device=generator.device, dtype=torch.long)
    conditioned = build_conditioned_prompt_embeds(
        prompt_embeds=prompt_embeds,
        bin_ids=bin_ids,
        embedding_bank=embedding_bank,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    history_sections, _ = split_history_target_sections(
        sample["video_latent_sections"].unsqueeze(0),
        num_generation_sections=cfg.generation.num_generation_sections,
    )
    generated_video = generator.run_inference(
        prompt_embeds=conditioned.prompt_embeds,
        negative_prompt_embeds=conditioned.negative_prompt_embeds,
        video_latent_sections=history_sections.to(generator.device),
    )
    frames = ((generated_video[0].detach().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    export_to_video(frames, output_path, fps=cfg.inference.fps)
    print(f"Saved turn-bin conditioned video to {output_path}")


if __name__ == "__main__":
    main()
