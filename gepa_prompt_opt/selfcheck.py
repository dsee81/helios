from __future__ import annotations

import argparse
from pathlib import Path

from .helios_infer import HeliosV2VConfig, build_infer_command
from .io_utils import read_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, required=True)
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()

    # Schemas should be readable JSON
    _ = read_json(repo_root / "gepa_prompt_opt" / "schemas" / "loop_dataset_manifest.schema.json")
    _ = read_json(repo_root / "gepa_prompt_opt" / "schemas" / "loop_prompt_template.schema.json")

    # Inference script should be parseable into an argv we can override
    infer_script = repo_root / "scripts" / "inference" / "helios-distilled_v2v.sh"
    cfg = HeliosV2VConfig(repo_root=repo_root, inference_script_path=infer_script)
    env, argv2 = build_infer_command(
        cfg,
        input_video_path=repo_root / "example" / "car.mp4",
        prompt="test prompt",
        output_folder=repo_root / "output_helios" / "tmp_selfcheck",
        num_frames=240,
    )
    if env.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Expected CUDA_VISIBLE_DEVICES=0")
    for flag in ["--video_path", "--prompt", "--output_folder"]:
        if flag not in argv2:
            raise RuntimeError(f"Expected {flag} in argv")

    print("Selfcheck OK (schemas + inference script parsing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
