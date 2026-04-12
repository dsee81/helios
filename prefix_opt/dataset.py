from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .actions import ACTION_TO_ID, DEFAULT_ACTION_ORDER
from .utils import canonical_sample_key, find_preferred_root, maybe_to_pil


@dataclass
class SampleRecord:
    sample_id: str
    action_name: str
    action_id: int
    video_path: str
    latent_path: str
    csv_path: str


def infer_action_id_from_motion(
    action_name: str,
    motion_df: pd.DataFrame,
    velocity_column: str,
    yaw_rate_column: str,
    v_stop_thresh: float,
    w_stop_thresh: float,
) -> int:
    max_abs_v = float(motion_df[velocity_column].abs().max())
    max_abs_w = float(motion_df[yaw_rate_column].abs().max())
    if max_abs_v < v_stop_thresh and max_abs_w < w_stop_thresh:
        return ACTION_TO_ID["stop"]
    return ACTION_TO_ID[action_name]


class ActionLatentVideoDataset(Dataset):
    def __init__(
        self,
        action_dirs: Dict[str, str],
        latent_root: str,
        csv_root: str,
        video_glob: str = "**/*.mp4",
        latent_glob: str = "**/*.pt",
        csv_glob: str = "**/*.csv",
        prompt_fallback: str = "A driving scene from a forward-facing camera.",
        num_frames: Optional[int] = None,
        height: int = 384,
        width: int = 640,
        cache_metadata: bool = True,
        force_rebuild: bool = False,
        v_stop_thresh: float = 0.05,
        w_stop_thresh: float = 0.05,
        velocity_column: str = "v_calculated",
        yaw_rate_column: str = "w_calculated",
    ):
        self.action_dirs = action_dirs
        self.latent_root = latent_root
        self.csv_root = csv_root
        self.video_glob = video_glob
        self.latent_glob = latent_glob
        self.csv_glob = csv_glob
        self.prompt_fallback = prompt_fallback
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.v_stop_thresh = v_stop_thresh
        self.w_stop_thresh = w_stop_thresh
        self.velocity_column = velocity_column
        self.yaw_rate_column = yaw_rate_column

        metadata_cache_path = os.path.join(latent_root, "prefix_opt_dataset_cache.pkl")
        if cache_metadata and os.path.exists(metadata_cache_path) and not force_rebuild:
            with open(metadata_cache_path, "rb") as handle:
                self.samples = pickle.load(handle)
        else:
            self.samples = self._build_records()
            if cache_metadata:
                os.makedirs(os.path.dirname(metadata_cache_path), exist_ok=True)
                with open(metadata_cache_path, "wb") as handle:
                    pickle.dump(self.samples, handle)

    def _index_files(self, root: str, pattern: str) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            stem = os.path.splitext(os.path.basename(path))[0]
            key = canonical_sample_key(stem)
            if key not in index:
                index[key] = path
        return index

    def _build_records(self) -> List[SampleRecord]:
        samples: List[SampleRecord] = []
        for action_name in DEFAULT_ACTION_ORDER:
            if action_name not in self.action_dirs:
                continue
            video_root = self.action_dirs[action_name]
            latent_root = find_preferred_root(self.latent_root, action_name)
            csv_root = find_preferred_root(self.csv_root, action_name)
            latent_index = self._index_files(latent_root, self.latent_glob)
            csv_index = self._index_files(csv_root, self.csv_glob)
            for video_path in glob.glob(os.path.join(video_root, self.video_glob), recursive=True):
                sample_id = os.path.splitext(os.path.basename(video_path))[0]
                key = canonical_sample_key(sample_id)
                latent_path = latent_index.get(key)
                csv_path = csv_index.get(key)
                if latent_path is None or csv_path is None:
                    continue
                samples.append(
                    SampleRecord(
                        sample_id=key,
                        action_name=action_name,
                        action_id=ACTION_TO_ID[action_name],
                        video_path=video_path,
                        latent_path=latent_path,
                        csv_path=csv_path,
                    )
                )
        if not samples:
            raise RuntimeError("No aligned samples were found for prefix optimization training.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        latent_payload = torch.load(sample.latent_path, map_location="cpu", weights_only=False)
        motion_df = pd.read_csv(sample.csv_path)
        action_id = infer_action_id_from_motion(
            action_name=sample.action_name,
            motion_df=motion_df,
            velocity_column=self.velocity_column,
            yaw_rate_column=self.yaw_rate_column,
            v_stop_thresh=self.v_stop_thresh,
            w_stop_thresh=self.w_stop_thresh,
        )
        prompt_embed = latent_payload["prompt_embed"]
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        from eval.utils.utils import load_video

        source_video = load_video(
            sample.video_path,
            num_frames=self.num_frames,
            return_tensor=True,
            width=self.width,
            height=self.height,
        )
        source_video = (source_video.float() / 127.5) - 1.0
        source_video = source_video.permute(1, 0, 2, 3).contiguous()
        first_frame_image = maybe_to_pil(latent_payload.get("first_frames_image"))
        if first_frame_image is None:
            first_frame = ((source_video[:, 0] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            first_frame_image = Image.fromarray(first_frame.permute(1, 2, 0).cpu().numpy()).convert("RGB")

        velocity = torch.tensor(motion_df[self.velocity_column].to_numpy(), dtype=torch.float32)
        yaw_rate = torch.tensor(motion_df[self.yaw_rate_column].to_numpy(), dtype=torch.float32)

        return {
            "sample_id": sample.sample_id,
            "action_name": sample.action_name,
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "prompt_raw": latent_payload.get("prompt_raw", self.prompt_fallback),
            "prompt_embed": prompt_embed.float(),
            "video_latent_sections": latent_payload["vae_latent"].float(),
            "first_frame_image": first_frame_image,
            "source_video": source_video.float(),
            "velocity": velocity,
            "yaw_rate": yaw_rate,
            "csv_path": sample.csv_path,
            "video_path": sample.video_path,
            "latent_path": sample.latent_path,
        }


def collate_prefix_opt_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "action_names": [item["action_name"] for item in batch],
        "action_ids": torch.stack([item["action_id"] for item in batch], dim=0),
        "prompt_raws": [item["prompt_raw"] for item in batch],
        "prompt_embeds": torch.cat([item["prompt_embed"] for item in batch], dim=0),
        "video_latent_sections": torch.stack([item["video_latent_sections"] for item in batch], dim=0),
        "first_frame_images": [item["first_frame_image"] for item in batch],
        "source_videos": torch.stack([item["source_video"] for item in batch], dim=0),
        "velocity": [item["velocity"] for item in batch],
        "yaw_rate": [item["yaw_rate"] for item in batch],
        "csv_paths": [item["csv_path"] for item in batch],
        "video_paths": [item["video_path"] for item in batch],
        "latent_paths": [item["latent_path"] for item in batch],
    }
