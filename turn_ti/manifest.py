from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from prefix_opt.utils import canonical_sample_key, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Build a turn-bin training manifest from chunk bin CSV output.")
    parser.add_argument("--chunk-csv", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--latent-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--video-glob", type=str, default="**/*.mp4")
    parser.add_argument("--latent-glob", type=str, default="**/*.pt")
    parser.add_argument("--default-prompt", type=str, default="A driving scene from a forward-facing camera.")
    return parser.parse_args()


def index_paths(root: str, pattern: str) -> Dict[str, str]:
    root_path = Path(root)
    index: Dict[str, str] = {}
    for path in root_path.glob(pattern):
        if not path.is_file():
            continue
        stem = path.stem
        index.setdefault(stem, str(path))
        index.setdefault(canonical_sample_key(stem), str(path))
    return index


def candidate_keys(sample_id: str) -> Iterable[str]:
    sample_id = str(sample_id)
    yield sample_id
    yield canonical_sample_key(sample_id)
    if sample_id.isdigit():
        yield f"clip_{sample_id}"
        yield canonical_sample_key(f"clip_{sample_id}")


def resolve_path(index: Dict[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in index:
            return index[key]
    return None


def main():
    args = parse_args()
    chunk_df = pd.read_csv(args.chunk_csv)
    required_columns = {"csv_path", "chunk_index", "start_frame", "end_frame", "chunk_length", "bin2sig6_id", "bin2sig6_name"}
    missing = required_columns.difference(chunk_df.columns)
    if missing:
        raise ValueError(f"Chunk CSV is missing required columns: {sorted(missing)}")

    video_index = index_paths(args.video_root, args.video_glob)
    latent_index = index_paths(args.latent_root, args.latent_glob)

    rows: List[dict] = []
    missing_video = 0
    missing_latent = 0
    for row in chunk_df.to_dict(orient="records"):
        csv_path = str(row["csv_path"])
        csv_stem = Path(csv_path).stem
        sample_id = csv_stem[:-7] if csv_stem.endswith("_synced") else csv_stem
        keys = list(candidate_keys(sample_id))
        video_path = resolve_path(video_index, keys)
        latent_path = resolve_path(latent_index, keys)
        if video_path is None:
            missing_video += 1
            continue
        if latent_path is None:
            missing_latent += 1
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "bin_id": int(row["bin2sig6_id"]),
                "bin_name": str(row["bin2sig6_name"]),
                "video_path": video_path,
                "latent_path": latent_path,
                "csv_path": csv_path,
                "prompt": row.get("prompt_raw", args.default_prompt),
                "chunk_index": int(row["chunk_index"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "chunk_length": int(row["chunk_length"]),
            }
        )

    if not rows:
        raise RuntimeError("Manifest builder could not align any rows with both a video and a latent file.")

    manifest = pd.DataFrame(rows).sort_values(["bin_id", "sample_id", "chunk_index"]).reset_index(drop=True)
    ensure_dir(os.path.dirname(args.output) or ".")
    manifest.to_csv(args.output, index=False)

    print(f"Wrote manifest to {args.output}")
    print(f"Rows: {len(manifest)}")
    print("Bin counts:")
    print(manifest["bin_name"].value_counts().sort_index().to_string())
    print(f"Skipped rows missing video: {missing_video}")
    print(f"Skipped rows missing latent: {missing_latent}")


if __name__ == "__main__":
    main()
