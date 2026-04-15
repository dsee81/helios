#!/bin/bash
set -euo pipefail

INPUT_CSV="${INPUT_CSV:-playground/helios_t2v_prompts.csv}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-playground/results}"
PLAYGROUND_DIR="${PLAYGROUND_DIR:-playground}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
VIDEO_DIR_OVERRIDE="${VIDEO_DIR:-}"
VIDEO_PATH_COLUMN="${VIDEO_PATH_COLUMN:-}"
TASK_TYPE="${TASK_TYPE:-reconstruct_original}"
SCORE_TYPE="${SCORE_TYPE:-rating}"
NUM_WORKERS="${NUM_WORKERS:-32}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_METRICS="${PARALLEL_METRICS:-1}"
RUN_NATURALNESS="${RUN_NATURALNESS:-1}"
DRY_RUN="${DRY_RUN:-0}"
DISABLE_VLM="${DISABLE_VLM:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_metric_jobs() {
    local output_dir="$1"
    local video_dir="$2"
    local baseline_name
    local metric_output_dir

    baseline_name=$(basename "$video_dir")
    metric_output_dir="$output_dir/$baseline_name"

    echo "Processing output_dir: $output_dir"
    echo "VIDEO_DIR=$video_dir"

    mkdir -p "$output_dir"

    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi

    if [ "$PARALLEL_METRICS" = "1" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 0_get_aesthetic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
                --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &

            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 1_get_motion_amplitude.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --num_workers "$NUM_WORKERS" &

            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 2_get_motion_smoothness.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &

            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 3_get_semantic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --semantic_model_path "checkpoints/ViCLIP" &

            if [ "$RUN_NATURALNESS" = "1" ]; then
                CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 4_get_naturalness.py \
                    --input_csv "$INPUT_CSV" \
                    --video_dir "$video_dir" \
                    --output_path "$output_dir" \
                    --api_key "$API_KEY" \
                    --base_url "$BASE_URL" \
                    --num_workers "$NUM_WORKERS" &
            fi

            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 5_get_drifting_aesthetic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
                --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &

            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 6_get_drifting_motion_smoothness.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &

            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 7_get_drifting_semantic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --semantic_model_path "checkpoints/ViCLIP" &

            if [ "$RUN_NATURALNESS" = "1" ]; then
                CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 8_get_drifting_naturalness.py \
                    --input_csv "$INPUT_CSV" \
                    --video_dir "$video_dir" \
                    --output_path "$output_dir" \
                    --api_key "$API_KEY" \
                    --base_url "$BASE_URL" \
                    --num_workers "$NUM_WORKERS" &
            fi

            wait
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 0_get_aesthetic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
                --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth"

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 1_get_motion_amplitude.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --num_workers "$NUM_WORKERS"

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 2_get_motion_smoothness.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --smoothness_model_path "checkpoints/amt_model/amt-s.pth"

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 3_get_semantic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --semantic_model_path "checkpoints/ViCLIP"

            if [ "$RUN_NATURALNESS" = "1" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 4_get_naturalness.py \
                    --input_csv "$INPUT_CSV" \
                    --video_dir "$video_dir" \
                    --output_path "$output_dir" \
                    --api_key "$API_KEY" \
                    --base_url "$BASE_URL" \
                    --num_workers "$NUM_WORKERS"
            fi

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 5_get_drifting_aesthetic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
                --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth"

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 6_get_drifting_motion_smoothness.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --smoothness_model_path "checkpoints/amt_model/amt-s.pth"

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 7_get_drifting_semantic.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --semantic_model_path "checkpoints/ViCLIP"

        if [ "$RUN_NATURALNESS" = "1" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" 8_get_drifting_naturalness.py \
                --input_csv "$INPUT_CSV" \
                --video_dir "$video_dir" \
                --output_path "$output_dir" \
                --api_key "$API_KEY" \
                --base_url "$BASE_URL" \
                --num_workers "$NUM_WORKERS"
        fi
    fi

    "$PYTHON_BIN" 9_merge_all_scores.py \
        --input_dir "$metric_output_dir" \
        --is_long

    local build_args=(
        --input_csv "$INPUT_CSV"
        --experiment_output_dir "$metric_output_dir"
        --task_type "$TASK_TYPE"
    )
    if [ -n "$VIDEO_PATH_COLUMN" ]; then
        build_args+=(--video_path_column "$VIDEO_PATH_COLUMN")
    fi
    if [ "$DISABLE_VLM" = "1" ]; then
        build_args+=(--disable_vlm)
    fi

    "$PYTHON_BIN" 11_build_combined_report.py "${build_args[@]}"
}

echo "=== Helios Eval Pipeline ==="
echo "INPUT_CSV=$INPUT_CSV"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "TASK_TYPE=$TASK_TYPE"
echo "GPU_ID=$GPU_ID"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "SCORE_TYPE=$SCORE_TYPE"
echo "VIDEO_PATH_COLUMN=$VIDEO_PATH_COLUMN"
echo "PARALLEL_METRICS=$PARALLEL_METRICS"
echo "RUN_NATURALNESS=$RUN_NATURALNESS"
echo "DRY_RUN=$DRY_RUN"

if [ -n "$EXPERIMENT_NAME" ]; then
    if [ -z "$VIDEO_DIR_OVERRIDE" ]; then
        echo "VIDEO_DIR must be set when EXPERIMENT_NAME is provided." >&2
        exit 1
    fi
    EXPERIMENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXPERIMENT_NAME"
    echo "EXPERIMENT_OUTPUT_DIR=$EXPERIMENT_OUTPUT_DIR"
    run_metric_jobs "$EXPERIMENT_OUTPUT_DIR" "$VIDEO_DIR_OVERRIDE"
else
    for MODEL_DIR in "$PLAYGROUND_DIR"/*/ ; do
        if [ ! -d "$MODEL_DIR" ]; then
            continue
        fi
        MODEL_NAME=$(basename "$MODEL_DIR")
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME"
        run_metric_jobs "$OUTPUT_DIR" "$MODEL_DIR"
    done

    "$PYTHON_BIN" 10_merge_all_results.py \
        --input_dir "$BASE_OUTPUT_DIR" \
        --score_type "$SCORE_TYPE"
fi
