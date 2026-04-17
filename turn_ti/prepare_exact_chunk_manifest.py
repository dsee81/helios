from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build an exact 120-frame turn_ti manifest from raw videos and 2-signal bins.")
    parser.add_argument("--chunk_csv", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--csv_root", type=str, required=True)
    parser.add_argument("--latent_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A forward-facing driving scene.")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    video_root = Path(args.video_root)
    csv_root = Path(args.csv_root)
    latent_root = Path(args.latent_root)
    output_path = Path(args.output)

    latent_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_video = 0
    missing_csv = 0
    with open(args.chunk_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            csv_name = str(row["csv"])
            sample_id = csv_name[:-11] if csv_name.endswith("_synced.csv") else Path(csv_name).stem
            chunk_index = int(row["chunk_idx"])
            chunk_length = int(row["n_frames"])
            start_frame = chunk_index * chunk_length
            end_frame = start_frame + chunk_length

            video_path = video_root / f"{sample_id}.mp4"
            csv_path = csv_root / csv_name
            latent_path = latent_root / f"{sample_id}_{start_frame}-{end_frame}_{chunk_length}_{args.height}_{args.width}.pt"

            if not video_path.exists():
                missing_video += 1
                continue
            if not csv_path.exists():
                missing_csv += 1
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "bin_id": int(row["bin2sig6_id"]),
                    "bin_name": str(row["bin2sig6_name"]),
                    "video_path": str(video_path.resolve()),
                    "latent_path": str(latent_path.resolve()),
                    "csv_path": str(csv_path.resolve()),
                    "prompt": args.prompt,
                    "chunk_index": chunk_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "chunk_length": chunk_length,
                }
            )

    fieldnames = [
        "sample_id",
        "bin_id",
        "bin_name",
        "video_path",
        "latent_path",
        "csv_path",
        "prompt",
        "chunk_index",
        "start_frame",
        "end_frame",
        "chunk_length",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest to {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"Skipped rows missing video: {missing_video}")
    print(f"Skipped rows missing csv: {missing_csv}")


if __name__ == "__main__":
    main()
