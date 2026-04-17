from __future__ import annotations

import argparse
import itertools
import time

import torch
from torch.utils.data import DataLoader

from prefix_opt.generator import HeliosPrefixV2VGenerator, flatten_video_latent_sections, split_history_target_sections

from turn_ti.conditioning import TurnBinEmbeddingBank, build_conditioned_prompt_embeds
from turn_ti.config import load_config
from turn_ti.dataset import TurnBinLatentVideoDataset, collate_turn_ti_batch
from turn_ti.losses import compute_turn_ti_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Run an instrumented sanity step for Turn-TI.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=3)
    return parser.parse_args()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("stage=config_loaded", flush=True)

    dataset = TurnBinLatentVideoDataset(
        manifest_path=cfg.data.manifest_path,
        prompt_fallback=cfg.data.prompt_fallback,
        num_frames=cfg.data.num_frames,
        height=cfg.data.height,
        width=cfg.data.width,
        load_source_video=cfg.data.load_source_video,
        cache_metadata=cfg.data.cache_metadata,
        force_rebuild=cfg.data.force_rebuild,
        strict_paths=cfg.data.strict_paths,
    )
    print(f"stage=dataset_ready size={len(dataset)}", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        prefetch_factor=cfg.train.prefetch_factor if cfg.train.dataloader_num_workers > 0 else None,
        collate_fn=collate_turn_ti_batch,
    )
    print("stage=dataloader_ready", flush=True)

    print("stage=generator_construct_start", flush=True)
    generator = HeliosPrefixV2VGenerator(cfg)
    print("stage=generator_constructed", flush=True)
    print("stage=generator_to_device_start", flush=True)
    generator = generator.to(device)
    print("stage=generator_ready", flush=True)
    print("stage=embedding_bank_init_start", flush=True)
    embedding_bank = TurnBinEmbeddingBank.from_phrases(
        generator=generator,
        init_phrases=cfg.model.init_phrases,
        num_vectors=cfg.model.num_vectors,
        init_scale=cfg.model.init_scale,
        learnable_delta=cfg.model.learnable_delta,
    ).to(device)
    print("stage=embedding_bank_ready", flush=True)
    optimizer = torch.optim.AdamW(
        embedding_bank.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_epsilon,
        weight_decay=cfg.train.adam_weight_decay,
    )
    print("stage=optimizer_ready", flush=True)
    print("stage=negative_prompt_encode_start", flush=True)
    negative_prompt_embeds = generator.encode_prompt_text([cfg.model.negative_prompt]).to(generator.transformer.dtype)
    print("stage=negative_prompt_ready", flush=True)
    generator.offload_text_encoder()
    print("stage=text_encoder_offloaded", flush=True)

    print(f"device={device}")
    print(f"dataset_size={len(dataset)}")
    print(f"train_batch_size={cfg.train.train_batch_size}")
    print(f"num_steps={args.num_steps}")

    step_durations = []
    loader_iter = iter(loader)
    for step in range(args.num_steps):
        fetch_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        fetch_end = time.perf_counter()

        sync_if_cuda(device)
        step_start = time.perf_counter()

        batch_size = batch["prompt_embeds"].shape[0]
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=generator.transformer.dtype)
        negative_batch = negative_prompt_embeds.expand(batch_size, -1, -1).to(device)
        conditioned = build_conditioned_prompt_embeds(
            prompt_embeds=prompt_embeds,
            bin_ids=batch["bin_ids"].to(device),
            embedding_bank=embedding_bank,
            negative_prompt_embeds=negative_batch,
        )
        history_sections, target_sections = split_history_target_sections(
            batch["video_latent_sections"],
            num_generation_sections=cfg.generation.num_generation_sections,
        )
        generated_latents = generator.generate_training_latents(
            prompt_embeds=conditioned.prompt_embeds,
            negative_prompt_embeds=conditioned.negative_prompt_embeds,
            video_latent_sections=history_sections.to(device),
            num_generation_sections=cfg.generation.num_generation_sections,
        )
        target_latents = flatten_video_latent_sections(target_sections).to(device, dtype=generated_latents.dtype)
        total_loss, logs = compute_turn_ti_losses(
            generated_latents=generated_latents,
            target_latents=target_latents,
            embedding_bank=embedding_bank,
            loss_cfg=cfg.loss,
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(embedding_bank.parameters(), cfg.train.max_grad_norm)
        optimizer.step()

        sync_if_cuda(device)
        step_end = time.perf_counter()

        fetch_time = fetch_end - fetch_start
        step_time = step_end - step_start
        step_durations.append(step_time)
        print(
            f"step={step + 1} "
            f"fetch_s={fetch_time:.3f} "
            f"step_s={step_time:.3f} "
            f"loss={logs['loss']:.4f} "
            f"recon={logs['reconstruction']:.4f} "
            f"anchor={logs['anchor']:.4f} "
            f"neighbor={logs['neighbor_smoothness']:.4f} "
            f"temporal={logs['temporal_smoothness']:.4f}"
        )

    mean_step = sum(step_durations) / len(step_durations)
    print(f"mean_step_s={mean_step:.3f}")


if __name__ == "__main__":
    main()
