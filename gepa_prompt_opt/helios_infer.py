from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class HeliosV2VConfig:
    repo_root: Path
    inference_script_path: Path  # scripts/inference/helios-distilled_v2v.sh
    cuda_visible_devices: str = "0"


def _extract_command_tokens(script_text: str) -> tuple[dict[str, str], list[str]]:
    """
    Extract the 'CUDA_VISIBLE_DEVICES=... python infer_helios.py ...' command from the bash script.
    We treat the script as a configuration source and do not modify it.
    """
    lines = script_text.splitlines()
    cmd_lines: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "python infer_helios.py" in stripped:
            capturing = True
        if capturing:
            cmd_lines.append(stripped)
            if not stripped.endswith("\\"):
                break

    if not cmd_lines:
        raise ValueError("Could not find 'python infer_helios.py' command in inference script.")

    joined = " ".join([l[:-1].strip() if l.endswith("\\") else l for l in cmd_lines])
    tokens = shlex.split(joined)

    env: dict[str, str] = {}
    argv: list[str] = []
    for t in tokens:
        if not argv and "=" in t and not t.startswith("--") and t.split("=", 1)[0].isidentifier():
            k, v = t.split("=", 1)
            env[k] = v
        else:
            argv.append(t)
    if not argv or argv[0] != "python":
        raise ValueError(f"Unexpected extracted command argv: {argv[:3]}")
    return env, argv


def _override_arg(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    if flag in out:
        idx = out.index(flag)
        if idx + 1 >= len(out):
            raise ValueError(f"Flag {flag} present but missing value")
        out[idx + 1] = value
        return out
    return out + [flag, value]


def build_infer_command(
    config: HeliosV2VConfig,
    *,
    input_video_path: Path,
    prompt: str,
    output_folder: Path,
    num_frames: int | None = None,
    fps: int | None = None,
) -> tuple[dict[str, str], list[str]]:
    script_text = config.inference_script_path.read_text(encoding="utf-8")
    base_env, base_argv = _extract_command_tokens(script_text)

    env = dict(os.environ)
    env.update(base_env)
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    argv = list(base_argv)
    if argv and argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    argv = _override_arg(argv, "--video_path", str(input_video_path))
    argv = _override_arg(argv, "--prompt", prompt)
    argv = _override_arg(argv, "--output_folder", str(output_folder))
    if num_frames is not None:
        argv = _override_arg(argv, "--num_frames", str(int(num_frames)))
    if fps is not None:
        argv = _override_arg(argv, "--fps", str(int(fps)))
    return env, argv


def run_v2v_inference(
    config: HeliosV2VConfig,
    *,
    input_video_path: Path,
    prompt: str,
    output_folder: Path,
    output_mp4_path: Path,
    num_frames: int | None = None,
    fps: int | None = None,
    timeout_seconds: int | None = None,
) -> Path:
    """
    Runs inference and returns `output_mp4_path`.

    `infer_helios.py` writes output mp4s into `output_folder`. We detect the newest mp4 and
    rename it to `output_mp4_path` for stable downstream evaluation.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    output_mp4_path.unlink(missing_ok=True)

    env, argv = build_infer_command(
        config,
        input_video_path=input_video_path,
        prompt=prompt,
        output_folder=output_folder,
        num_frames=num_frames,
        fps=fps,
    )

    before = {p.resolve() for p in output_folder.glob("*.mp4")}
    subprocess.run(argv, env=env, cwd=str(config.repo_root), check=True, timeout=timeout_seconds)
    after = sorted([p.resolve() for p in output_folder.glob("*.mp4")], key=lambda p: p.stat().st_mtime)
    new_files = [p for p in after if p not in before]
    if not new_files:
        raise RuntimeError(f"No new mp4 produced in output_folder: {output_folder}")
    newest = new_files[-1]
    newest.replace(output_mp4_path)
    if num_frames is not None:
        _normalize_output_video(output_mp4_path, target_num_frames=int(num_frames), target_fps=fps)
    return output_mp4_path


def _normalize_output_video(path: Path, *, target_num_frames: int, target_fps: int | None) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open generated video for normalization: {path}")

    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_fps = float(target_fps or src_fps)

    if src_frames == target_num_frames and abs(src_fps - desired_fps) < 0.01:
        cap.release()
        return

    tmp_path = path.with_name(path.stem + "_normalized.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), desired_fps, (width, height))
    written = 0
    while written < target_num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Video normalization wrote zero frames: {path}")

    tmp_path.replace(path)
