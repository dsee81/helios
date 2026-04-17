#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/dataDisk/Helios"
GPU_ID="${GPU_ID:-4}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python3 -m turn_ti.train --config /root/dataDisk/Helios/turn_ti/turn_ti_2signal_120_filtered_sanity.yaml
