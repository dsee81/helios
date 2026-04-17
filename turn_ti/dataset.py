from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from video_reader import PyVideoReader

from prefix_opt.utils import maybe_to_pil


@dataclass
class TurnBinSampleRecord:
    sample_id: str
    bin_id: int
    bin_name: str
    video_path: str
    latent_path: str
    csv_path: str
    prompt: str
    chunk_index: int
    start_frame: int
    end_frame: int
    chunk_length: int


class TurnBinLatentVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        prompt_fallback: str = "A driving scene from a forward-facing camera.",
        num_frames: int | None = None,
        height: int = 384,
        width: int = 640,
        load_source_video: bool = False,
        cache_metadata: bool = True,
        force_rebuild: bool = False,
        strict_paths: bool = True,
    ):
        self.manifest_path = manifest_path
        self.prompt_fallback = prompt_fallback
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.load_source_video = load_source_video
        self.strict_paths = strict_paths

        manifest_stem = Path(manifest_path).stem
        metadata_cache_path = os.path.join(
            os.path.dirname(manifest_path),
            f"turn_ti_dataset_cache_{manifest_stem}.pkl",
        )
        if cache_metadata and os.path.exists(metadata_cache_path) and not force_rebuild:
            with open(metadata_cache_path, "rb") as handle:
                self.samples = pickle.load(handle)
        else:
            self.samples = self._build_records()
            if cache_metadata:
                with open(metadata_cache_path, "wb") as handle:
                    pickle.dump(self.samples, handle)

    def _validate_path(self, path: str, label: str) -> None:
        if not path:
            raise ValueError(f"Manifest row is missing required {label}.")
        if self.strict_paths and not os.path.exists(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")

    def _build_records(self) -> List[TurnBinSampleRecord]:
        manifest = pd.read_csv(self.manifest_path)
        required_columns = {
            "sample_id",
            "bin_id",
            "bin_name",
            "video_path",
            "latent_path",
            "csv_path",
            "chunk_index",
            "start_frame",
            "end_frame",
            "chunk_length",
        }
        missing = required_columns.difference(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        samples: List[TurnBinSampleRecord] = []
        for row in manifest.to_dict(orient="records"):
            self._validate_path(str(row["video_path"]), "video_path")
            self._validate_path(str(row["latent_path"]), "latent_path")
            self._validate_path(str(row["csv_path"]), "csv_path")
            samples.append(
                TurnBinSampleRecord(
                    sample_id=str(row["sample_id"]),
                    bin_id=int(row["bin_id"]),
                    bin_name=str(row["bin_name"]),
                    video_path=str(row["video_path"]),
                    latent_path=str(row["latent_path"]),
                    csv_path=str(row["csv_path"]),
                    prompt=str(row.get("prompt", self.prompt_fallback) or self.prompt_fallback),
                    chunk_index=int(row["chunk_index"]),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    chunk_length=int(row["chunk_length"]),
                )
            )
        if not samples:
            raise RuntimeError("No aligned samples were found for turn-bin textual inversion training.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_exact_video_chunk(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        reader_info = PyVideoReader(video_path, threads=0)
        total_frames, original_height, original_width = reader_info.get_shape()
        if end_frame > total_frames:
            raise ValueError(
                f"Requested frames [{start_frame}, {end_frame}) exceed video length {total_frames} for {video_path}"
            )

        original_aspect_ratio = original_width / original_height
        if self.width > self.height:
            target_width = self.width
            target_height = int(self.width / original_aspect_ratio)
        else:
            target_height = self.height
            target_width = int(self.height * original_aspect_ratio)
        target_height = int(round(target_height / 2) * 2)
        target_width = int(round(target_width / 2) * 2)

        reader = PyVideoReader(video_path, target_height=target_height, target_width=target_width, threads=0)
        frame_indices = list(range(start_frame, end_frame))
        frames = torch.from_numpy(reader.get_batch(frame_indices)).float()  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        _, _, cur_h, cur_w = frames.shape
        aspect_ratio_original = cur_h / cur_w
        aspect_ratio_target = self.height / self.width
        if aspect_ratio_original >= aspect_ratio_target:
            new_h = int(cur_w * aspect_ratio_target)
            top = (cur_h - new_h) // 2
            frames = frames[:, :, top : top + new_h, :]
        else:
            new_w = int(cur_h / aspect_ratio_target)
            left = (cur_w - new_w) // 2
            frames = frames[:, :, :, left : left + new_w]

        frames = F.interpolate(frames, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return frames

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        latent_payload = torch.load(sample.latent_path, map_location="cpu", weights_only=False)
        prompt_embed = latent_payload["prompt_embed"]
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        first_frame_image = maybe_to_pil(latent_payload.get("first_frames_image"))
        source_video = None
        if self.load_source_video:
            source_video = self._load_exact_video_chunk(
                video_path=sample.video_path,
                start_frame=sample.start_frame,
                end_frame=sample.end_frame,
            )
            if self.num_frames is not None and source_video.shape[0] > self.num_frames:
                source_video = source_video[: self.num_frames]
            source_video = (source_video.float() / 127.5) - 1.0
            source_video = source_video.permute(1, 0, 2, 3).contiguous()
        if first_frame_image is None and source_video is not None:
            first_frame = ((source_video[:, 0] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            first_frame_image = Image.fromarray(first_frame.permute(1, 2, 0).cpu().numpy()).convert("RGB")
        elif first_frame_image is None:
            raise ValueError(f"Missing first frame image and source video is disabled for sample {sample.sample_id}.")

        return {
            "sample_id": sample.sample_id,
            "bin_name": sample.bin_name,
            "bin_id": torch.tensor(sample.bin_id, dtype=torch.long),
            "prompt_raw": latent_payload.get("prompt_raw", sample.prompt),
            "prompt_embed": prompt_embed.float(),
            "video_latent_sections": latent_payload["vae_latent"].float(),
            "first_frame_image": first_frame_image,
            "source_video": source_video.float() if source_video is not None else None,
            "csv_path": sample.csv_path,
            "video_path": sample.video_path,
            "latent_path": sample.latent_path,
            "chunk_index": sample.chunk_index,
            "start_frame": sample.start_frame,
            "end_frame": sample.end_frame,
            "chunk_length": sample.chunk_length,
        }


def collate_turn_ti_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    collated = {
        "sample_ids": [item["sample_id"] for item in batch],
        "bin_names": [item["bin_name"] for item in batch],
        "bin_ids": torch.stack([item["bin_id"] for item in batch], dim=0),
        "prompt_raws": [item["prompt_raw"] for item in batch],
        "prompt_embeds": torch.cat([item["prompt_embed"] for item in batch], dim=0),
        "video_latent_sections": torch.stack([item["video_latent_sections"] for item in batch], dim=0),
        "first_frame_images": [item["first_frame_image"] for item in batch],
        "csv_paths": [item["csv_path"] for item in batch],
        "video_paths": [item["video_path"] for item in batch],
        "latent_paths": [item["latent_path"] for item in batch],
        "chunk_indices": torch.tensor([item["chunk_index"] for item in batch], dtype=torch.long),
        "start_frames": torch.tensor([item["start_frame"] for item in batch], dtype=torch.long),
        "end_frames": torch.tensor([item["end_frame"] for item in batch], dtype=torch.long),
        "chunk_lengths": torch.tensor([item["chunk_length"] for item in batch], dtype=torch.long),
    }
    if batch[0]["source_video"] is not None:
        collated["source_videos"] = torch.stack([item["source_video"] for item in batch], dim=0)
    return collated
