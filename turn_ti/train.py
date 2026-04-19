from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Subset

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


def build_video_grouped_split(dataset: TurnBinLatentVideoDataset, val_ratio: float, seed: int) -> tuple[Subset, Subset | None, dict]:
    """
    Split by source video so adjacent chunks from one video cannot leak across train/val.
    """
    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        indices_by_video[str(sample.video_path)].append(idx)

    all_indices = list(range(len(dataset)))
    if val_ratio <= 0.0 or len(indices_by_video) < 2:
        return Subset(dataset, all_indices), None, {
            "train_samples": len(all_indices),
            "val_samples": 0,
            "train_videos": len(indices_by_video),
            "val_videos": 0,
        }

    groups = list(indices_by_video.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    target_val = max(1, int(round(len(dataset) * val_ratio)))
    val_video_keys: set[str] = set()
    val_indices: list[int] = []
    for video_path, group_indices in groups:
        if len(val_video_keys) >= len(groups) - 1:
            break
        val_video_keys.add(video_path)
        val_indices.extend(group_indices)
        if len(val_indices) >= target_val:
            break

    val_set = set(val_indices)
    train_indices = [idx for idx in all_indices if idx not in val_set]
    if not train_indices or not val_indices:
        return Subset(dataset, all_indices), None, {
            "train_samples": len(all_indices),
            "val_samples": 0,
            "train_videos": len(indices_by_video),
            "val_videos": 0,
        }

    train_video_keys = set(indices_by_video) - val_video_keys
    split_info = {
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_videos": len(train_video_keys),
        "val_videos": len(val_video_keys),
        "val_ratio": float(val_ratio),
        "val_split_seed": int(seed),
    }
    return Subset(dataset, train_indices), Subset(dataset, sorted(val_indices)), split_info


def make_loader(dataset, cfg, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        prefetch_factor=cfg.train.prefetch_factor if cfg.train.dataloader_num_workers > 0 else None,
        collate_fn=collate_turn_ti_batch,
    )


def compute_batch_losses(batch, *, cfg, accelerator, generator, embedding_bank, negative_prompt_embeds):
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
    return compute_turn_ti_losses(
        generated_latents=generated_latents,
        target_latents=target_latents,
        embedding_bank=accelerator.unwrap_model(embedding_bank),
        loss_cfg=cfg.loss,
    )


@torch.no_grad()
def evaluate_validation(val_loader, *, cfg, accelerator, generator, embedding_bank, negative_prompt_embeds) -> dict[str, float]:
    if val_loader is None:
        return {}

    was_training = embedding_bank.training
    embedding_bank.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in val_loader:
        _, logs = compute_batch_losses(
            batch,
            cfg=cfg,
            accelerator=accelerator,
            generator=generator,
            embedding_bank=embedding_bank,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if cfg.train.max_eval_batches and count >= cfg.train.max_eval_batches:
            break

    if was_training:
        embedding_bank.train()
    return {key: value / max(1, count) for key, value in totals.items()}


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
    train_dataset, val_dataset, split_info = build_video_grouped_split(
        dataset,
        val_ratio=cfg.train.val_ratio,
        seed=cfg.train.val_split_seed,
    )
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False) if val_dataset is not None else None
    if accelerator.is_main_process:
        print(f"Turn-TI split: {split_info}")
        if cfg.train.log_with:
            accelerator.log({f"split/{key}": value for key, value in split_info.items()}, step=0)

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
    best_val_loss = float("inf")
    if args.resume:
        payload = load_turn_ti_checkpoint(args.resume, embedding_bank=embedding_bank, optimizer=optimizer, map_location="cpu")
        global_step = int(payload.get("step", 0))
        best_val_loss = float(payload.get("metadata", {}).get("best_val_loss", best_val_loss))

    if val_loader is not None:
        embedding_bank, optimizer, train_loader, lr_scheduler, val_loader = accelerator.prepare(
            embedding_bank, optimizer, train_loader, lr_scheduler, val_loader
        )
    else:
        embedding_bank, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            embedding_bank, optimizer, train_loader, lr_scheduler
        )
    negative_prompt_embeds = generator.encode_prompt_text([cfg.model.negative_prompt]).to(generator.transformer.dtype)
    generator.offload_text_encoder()

    while global_step < cfg.train.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(embedding_bank):
                total_loss, logs = compute_batch_losses(
                    batch,
                    cfg=cfg,
                    accelerator=accelerator,
                    generator=generator,
                    embedding_bank=embedding_bank,
                    negative_prompt_embeds=negative_prompt_embeds,
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
                accelerator.log({f"train/{key}": value for key, value in logs.items()}, step=global_step)
                if (
                    val_loader is not None
                    and cfg.train.val_every > 0
                    and global_step % cfg.train.val_every == 0
                ):
                    val_logs = evaluate_validation(
                        val_loader,
                        cfg=cfg,
                        accelerator=accelerator,
                        generator=generator,
                        embedding_bank=embedding_bank,
                        negative_prompt_embeds=negative_prompt_embeds,
                    )
                    if val_logs:
                        accelerator.log({f"val/{key}": value for key, value in val_logs.items()}, step=global_step)
                        val_loss = float(val_logs.get("loss", float("inf")))
                        if accelerator.is_main_process and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_turn_ti_checkpoint(
                                output_dir=cfg.train.output_dir,
                                step=global_step,
                                embedding_bank=accelerator.unwrap_model(embedding_bank),
                                optimizer=optimizer,
                                metadata={
                                    "config": args.config,
                                    "best_val_loss": best_val_loss,
                                    "split": split_info,
                                },
                                filename="turn_ti_checkpoint_best.pt",
                            )
                if accelerator.is_main_process and global_step % cfg.train.save_every == 0:
                    save_turn_ti_checkpoint(
                        output_dir=cfg.train.output_dir,
                        step=global_step,
                        embedding_bank=accelerator.unwrap_model(embedding_bank),
                        optimizer=optimizer,
                        metadata={"config": args.config, "best_val_loss": best_val_loss, "split": split_info},
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
            metadata={"config": args.config, "best_val_loss": best_val_loss, "split": split_info},
            filename="turn_ti_checkpoint_final.pt",
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
