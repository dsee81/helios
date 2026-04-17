from __future__ import annotations

import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from prefix_opt.generator import HeliosPrefixV2VGenerator, flatten_video_latent_sections, split_history_target_sections
from prefix_opt.utils import ensure_dir, set_seed

from .checkpointing import load_turn_ti_checkpoint, save_turn_ti_checkpoint
from .conditioning import TurnBinEmbeddingBank, build_conditioned_prompt_embeds
from .config import load_config
from .dataset import TurnBinLatentVideoDataset, collate_turn_ti_batch
from .losses import compute_turn_ti_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train turn-bin textual inversion style conditioning for Helios.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dir(cfg.train.output_dir)
    ensure_dir(os.path.join(cfg.train.output_dir, cfg.train.logging_dir))
    set_seed(cfg.train.seed)

    if cfg.train.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.model.mixed_precision,
        log_with=cfg.train.log_with,
        project_config=ProjectConfiguration(
            project_dir=cfg.train.output_dir,
            logging_dir=os.path.join(cfg.train.output_dir, cfg.train.logging_dir),
        ),
    )
    if cfg.train.single_gpu_only and accelerator.num_processes != 1:
        raise RuntimeError(
            f"Turn-TI MVP is currently restricted to one process / one GPU, got {accelerator.num_processes} processes."
        )
    if accelerator.is_main_process and cfg.train.log_with:
        accelerator.init_trackers("turn-ti", config={"config_path": args.config})

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
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        prefetch_factor=cfg.train.prefetch_factor if cfg.train.dataloader_num_workers > 0 else None,
        collate_fn=collate_turn_ti_batch,
    )

    generator = HeliosPrefixV2VGenerator(cfg).to(accelerator.device)
    if len(cfg.model.bin_names) != cfg.model.num_bins or len(cfg.model.init_phrases) != cfg.model.num_bins:
        raise ValueError("bin_names and init_phrases must have exactly num_bins entries.")

    embedding_bank = TurnBinEmbeddingBank.from_phrases(
        generator=generator,
        init_phrases=cfg.model.init_phrases,
        num_vectors=cfg.model.num_vectors,
        init_scale=cfg.model.init_scale,
        learnable_delta=cfg.model.learnable_delta,
    )
    optimizer = torch.optim.AdamW(
        embedding_bank.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_epsilon,
        weight_decay=cfg.train.adam_weight_decay,
    )
    lr_scheduler = get_scheduler(
        cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,
        num_training_steps=cfg.train.max_train_steps,
    )

    global_step = 0
    if args.resume:
        payload = load_turn_ti_checkpoint(args.resume, embedding_bank=embedding_bank, optimizer=optimizer, map_location="cpu")
        global_step = int(payload.get("step", 0))

    embedding_bank, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        embedding_bank, optimizer, train_loader, lr_scheduler
    )
    negative_prompt_embeds = generator.encode_prompt_text([cfg.model.negative_prompt]).to(generator.transformer.dtype)
    generator.offload_text_encoder()

    while global_step < cfg.train.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(embedding_bank):
                batch_size = batch["prompt_embeds"].shape[0]
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=generator.transformer.dtype)
                negative_batch = negative_prompt_embeds.expand(batch_size, -1, -1).to(accelerator.device)
                conditioned = build_conditioned_prompt_embeds(
                    prompt_embeds=prompt_embeds,
                    bin_ids=batch["bin_ids"].to(accelerator.device),
                    embedding_bank=accelerator.unwrap_model(embedding_bank),
                    negative_prompt_embeds=negative_batch,
                )
                history_sections, target_sections = split_history_target_sections(
                    batch["video_latent_sections"],
                    num_generation_sections=cfg.generation.num_generation_sections,
                )
                generated_latents = generator.generate_training_latents(
                    prompt_embeds=conditioned.prompt_embeds,
                    negative_prompt_embeds=conditioned.negative_prompt_embeds,
                    video_latent_sections=history_sections.to(accelerator.device),
                    num_generation_sections=cfg.generation.num_generation_sections,
                )
                target_latents = flatten_video_latent_sections(target_sections).to(
                    accelerator.device, dtype=generated_latents.dtype
                )
                total_loss, logs = compute_turn_ti_losses(
                    generated_latents=generated_latents,
                    target_latents=target_latents,
                    embedding_bank=accelerator.unwrap_model(embedding_bank),
                    loss_cfg=cfg.loss,
                )
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(embedding_bank.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                logs["lr"] = float(lr_scheduler.get_last_lr()[0])
                logs["mean_bin_id"] = float(batch["bin_ids"].float().mean().item())

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process and global_step % cfg.train.save_every == 0:
                    save_turn_ti_checkpoint(
                        output_dir=cfg.train.output_dir,
                        step=global_step,
                        embedding_bank=accelerator.unwrap_model(embedding_bank),
                        optimizer=optimizer,
                        metadata={"config": args.config},
                    )
                if global_step >= cfg.train.max_train_steps:
                    break
        if global_step >= cfg.train.max_train_steps:
            break

    if accelerator.is_main_process:
        save_turn_ti_checkpoint(
            output_dir=cfg.train.output_dir,
            step=global_step,
            embedding_bank=accelerator.unwrap_model(embedding_bank),
            optimizer=optimizer,
            metadata={"config": args.config},
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
