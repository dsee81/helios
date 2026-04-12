from __future__ import annotations

import argparse
from pathlib import Path

from .gepa_driver import OptimizeConfig, optimize_with_gepa


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gepa_prompt_opt")
    sub = p.add_subparsers(dest="cmd", required=True)

    opt = sub.add_parser("optimize-loop", help="Run GEPA prompt-template optimization for loop task.")
    opt.add_argument("--repo_root", type=str, required=True)
    opt.add_argument("--dataset_manifest", type=str, required=True)
    opt.add_argument("--seed_template", type=str, required=True)
    opt.add_argument("--work_dir", type=str, required=True)
    opt.add_argument("--num_iterations", type=int, default=10)
    opt.add_argument("--candidates_per_iteration", type=int, default=4)
    opt.add_argument("--num_frames", type=int, default=240)
    opt.add_argument(
        "--reflection_lm",
        type=str,
        default="deepseek/deepseek-chat",
        help="LiteLLM model string used by GEPA to propose edits (e.g. deepseek/deepseek-chat).",
    )

    args = p.parse_args(argv)

    if args.cmd == "optimize-loop":
        cfg = OptimizeConfig(
            repo_root=Path(args.repo_root),
            work_dir=Path(args.work_dir),
            dataset_manifest=Path(args.dataset_manifest),
            seed_template=Path(args.seed_template),
            num_iterations=int(args.num_iterations),
            candidates_per_iteration=int(args.candidates_per_iteration),
            num_frames=int(args.num_frames),
            reflection_lm=str(args.reflection_lm),
        )
        optimize_with_gepa(cfg)
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
