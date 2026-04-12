from __future__ import annotations

import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from .checkpointing import save_prefix_checkpoint
from .conditioning import ActionPrefixBank, build_conditioned_prompt_embeds
from .config import load_config
from .dataset import ActionLatentVideoDataset, collate_prefix_opt_batch
from .generator import HeliosPrefixV2VGenerator
from .losses import compute_prefix_losses
from .utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train prefix-only V2V action control for Helios.")
    parser.add_argument("--config", type=str, required=True)
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
    if accelerator.is_main_process and cfg.train.log_with:
        accelerator.init_trackers("prefix-opt", config={"config_path": args.config})

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
        force_rebuild=cfg.data.force_rebuild,
        v_stop_thresh=cfg.data.v_stop_thresh,
        w_stop_thresh=cfg.data.w_stop_thresh,
        velocity_column=cfg.data.velocity_column,
        yaw_rate_column=cfg.data.yaw_rate_column,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        prefetch_factor=cfg.train.prefetch_factor if cfg.train.dataloader_num_workers > 0 else None,
        collate_fn=collate_prefix_opt_batch,
    )

    generator = HeliosPrefixV2VGenerator(cfg).to(accelerator.device)
    prefix_bank = ActionPrefixBank(
        prefix_length=cfg.model.prefix_length,
        hidden_size=generator.transformer.config.text_dim,
        init_std=cfg.model.prefix_init_std,
    )
    optimizer = torch.optim.AdamW(
        prefix_bank.parameters(),
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

    prefix_bank, optimizer, train_loader, lr_scheduler = accelerator.prepare(prefix_bank, optimizer, train_loader, lr_scheduler)
    negative_prompt_embeds = generator.encode_prompt_text([cfg.model.negative_prompt]).expand(
        cfg.train.train_batch_size, -1, -1
    )

    zero_grad_counter = 0
    global_step = 0
    while global_step < cfg.train.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(prefix_bank):
                batch_size = batch["prompt_embeds"].shape[0]
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=generator.transformer.dtype)
                negative_batch = negative_prompt_embeds[:batch_size].to(accelerator.device, dtype=generator.transformer.dtype)
                conditioned = build_conditioned_prompt_embeds(
                    prompt_embeds=prompt_embeds,
                    action_ids=batch["action_ids"].to(accelerator.device),
                    prefix_bank=accelerator.unwrap_model(prefix_bank),
                    negative_prompt_embeds=negative_batch,
                )
                _, generated_video = generator.generate_training_video(
                    prompt_embeds=conditioned.prompt_embeds,
                    negative_prompt_embeds=conditioned.negative_prompt_embeds,
                    video_latent_sections=batch["video_latent_sections"].to(accelerator.device),
                    num_generation_sections=cfg.generation.num_generation_sections,
                )
                source_video = batch["source_videos"].to(accelerator.device, dtype=generated_video.dtype)
                total_loss, logs, _ = compute_prefix_losses(
                    generated_video=generated_video,
                    source_video=source_video,
                    velocity_targets=batch["velocity"],
                    yaw_targets=batch["yaw_rate"],
                    action_ids=batch["action_ids"].to(accelerator.device),
                    loss_cfg=cfg.loss,
                    global_step=global_step,
                )
                accelerator.backward(total_loss)

                grad_norm = None
                prefix_module = accelerator.unwrap_model(prefix_bank)
                if prefix_module.prefix.grad is not None:
                    grad_norm = float(prefix_module.prefix.grad.norm().detach().item())
                logs["prefix_grad_norm"] = grad_norm or 0.0

                if global_step >= cfg.train.gradient_check_warmup_steps:
                    if grad_norm is None or grad_norm == 0.0:
                        zero_grad_counter += 1
                    else:
                        zero_grad_counter = 0
                    if zero_grad_counter > cfg.train.zero_grad_tolerance_steps:
                        raise RuntimeError("Prefix gradients stayed zero for too many steps.")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prefix_bank.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                logs["lr"] = float(lr_scheduler.get_last_lr()[0])

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process and global_step % cfg.train.save_every == 0:
                    save_prefix_checkpoint(
                        output_dir=cfg.train.output_dir,
                        step=global_step,
                        prefix_bank=accelerator.unwrap_model(prefix_bank),
                        optimizer=optimizer,
                        metadata={"config": args.config},
                    )
                if global_step >= cfg.train.max_train_steps:
                    break
        if global_step >= cfg.train.max_train_steps:
            break

    if accelerator.is_main_process:
        save_prefix_checkpoint(
            output_dir=cfg.train.output_dir,
            step=global_step,
            prefix_bank=accelerator.unwrap_model(prefix_bank),
            optimizer=optimizer,
            metadata={"config": args.config},
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
