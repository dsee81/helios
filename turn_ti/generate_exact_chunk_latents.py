from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel
from video_reader import PyVideoReader

from diffusers import AutoencoderKLWan

from helios.utils.utils_base import encode_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate exact chunk latents for turn_ti from raw videos.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="BestWishYsh/Helios-Base")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def align_dimension(value: int, alignment: int = 2) -> int:
    return int(round(value / alignment) * alignment)


def load_exact_video_chunk(video_path: str, start_frame: int, end_frame: int, height: int, width: int) -> torch.Tensor:
    reader_info = PyVideoReader(video_path, threads=0)
    total_frames, original_height, original_width = reader_info.get_shape()
    if end_frame > total_frames:
        raise ValueError(f"Requested frames [{start_frame}, {end_frame}) exceed video length {total_frames} for {video_path}")

    original_aspect_ratio = original_width / original_height
    if width > height:
        target_width = width
        target_height = int(width / original_aspect_ratio)
    else:
        target_height = height
        target_width = int(height * original_aspect_ratio)
    target_height = align_dimension(target_height, 2)
    target_width = align_dimension(target_width, 2)

    reader = PyVideoReader(video_path, target_height=target_height, target_width=target_width, threads=0)
    frame_indices = list(range(start_frame, end_frame))
    frames = torch.from_numpy(reader.get_batch(frame_indices)).float()  # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

    _, _, cur_h, cur_w = frames.shape
    aspect_ratio_original = cur_h / cur_w
    aspect_ratio_target = height / width
    if aspect_ratio_original >= aspect_ratio_target:
        new_h = int(cur_w * aspect_ratio_target)
        top = (cur_h - new_h) // 2
        frames = frames[:, :, top : top + new_h, :]
    else:
        new_w = int(cur_h / aspect_ratio_target)
        left = (cur_w - new_w) // 2
        frames = frames[:, :, :, left : left + new_w]

    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    return (frames / 127.5) - 1.0


def to_pil_first_frame(frames: torch.Tensor) -> Image.Image:
    first_frame = ((frames[0] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(first_frame.permute(1, 2, 0).cpu().numpy()).convert("RGB")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    ).to(device)
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    text_encoder.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_rows is not None:
        rows = rows[args.start_index : args.start_index + args.max_rows]
    else:
        rows = rows[args.start_index :]

    frame_window_size = (9 - 1) * 4 + 1  # 33
    for row in tqdm(rows, desc="Generating exact chunk latents"):
        latent_path = Path(row["latent_path"])
        if latent_path.exists() and not args.overwrite:
            continue

        frames = load_exact_video_chunk(
            video_path=row["video_path"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            height=args.height,
            width=args.width,
        )
        if frames.shape[0] < frame_window_size:
            continue

        pixel_values = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=device, dtype=vae.dtype)
        num_sections = pixel_values.shape[2] // frame_window_size
        if num_sections == 0:
            continue

        history_latent_list = []
        with torch.no_grad():
            for section_idx in range(num_sections):
                start = section_idx * frame_window_size
                end = start + frame_window_size
                cur_pixels = pixel_values[:, :, start:end, :, :]
                cur_latent = vae.encode(cur_pixels).latent_dist.sample()
                cur_latent = (cur_latent - latents_mean) * latents_std
                history_latent_list.append(cur_latent)
            vae_latents = torch.stack(history_latent_list, dim=1)[0].cpu().detach()

            prompt_embeds, _ = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=[row.get("prompt") or "A forward-facing driving scene."],
                device=device,
            )
            prompt_embed = prompt_embeds[0].cpu().detach()

        latent_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vae_latent": vae_latents,
            "prompt_embed": prompt_embed,
            "first_frames_image": to_pil_first_frame(frames),
            "prompt_raw": row.get("prompt") or "A forward-facing driving scene.",
        }
        torch.save(payload, latent_path)


if __name__ == "__main__":
    main()
