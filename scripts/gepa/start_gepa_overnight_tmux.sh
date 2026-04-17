#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-gepa_overnight_$(date -u +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gepa_overnight_${TIMESTAMP}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CANDIDATES_PER_ITERATION="${CANDIDATES_PER_ITERATION:-1}"
NUM_FRAMES="${NUM_FRAMES:-240}"
FPS="${FPS:-30}"
REFLECTION_LM="${REFLECTION_LM:-deepseek/deepseek-chat}"

mkdir -p "${RUN_ROOT}/logs"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

CMD="cd \"${REPO_ROOT}\" && .venv/bin/python -m gepa_prompt_opt.overnight_runner \
  --repo_root \"${REPO_ROOT}\" \
  --run_root \"${RUN_ROOT}\" \
  --video_dir \"${REPO_ROOT}/gepa_inf_samples/gepa_samples\" \
  --seed_template \"gepa_prompt_opt/examples/seed_loop_template.json\" \
  --reflection_lm \"${REFLECTION_LM}\" \
  --cuda_visible_devices \"${CUDA_VISIBLE_DEVICES_VALUE}\" \
  --num_iterations \"${NUM_ITERATIONS}\" \
  --candidates_per_iteration \"${CANDIDATES_PER_ITERATION}\" \
  --num_frames \"${NUM_FRAMES}\" \
  --fps \"${FPS}\" \
  > \"${RUN_ROOT}/logs/runner.log\" 2> \"${RUN_ROOT}/logs/runner.err\""

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${CMD}'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Run root: ${RUN_ROOT}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs:"
echo "  stdout: ${RUN_ROOT}/logs/runner.log"
echo "  stderr: ${RUN_ROOT}/logs/runner.err"
