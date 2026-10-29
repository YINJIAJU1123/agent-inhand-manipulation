#!/usr/bin/env bash
# Independent teacher-driven RGB capture and frozen pose-network diagnosis.
set -Eeuo pipefail
: "${RUNTIME_ROOT:?Set the independent runtime root}"
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with available capacity}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
out="$RUNTIME_ROOT/outputs/pose_rgb_$run_id"
mkdir "$out"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
export MAX_JOBS="${MAX_JOBS:-2}"
cd "$RUNTIME_ROOT/viserDex_revo3"
stage=capture
printf '%s\n' "$stage" > "$out/current_stage.txt"
bash scripts/run_newton.sh scripts/pose_estimator/collect_revo3_occlusion.py \
  --task Isaac-ViserDex-Repose-Revo3 \
  --teacher_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/teacher/model_4800.pt" \
  --hikrobot_calibration config/cameras \
  --num_envs 16 --samples 1024 --max_steps 3000 --stride 4 --burn_in 120 \
  --seed 182911 --output_dir "$out/data" --viz none --device cuda:0 \
  > "$out/capture.log" 2>&1
stage=frozen_cnn
printf '%s\n' "$stage" > "$out/current_stage.txt"
bash scripts/run_newton.sh scripts/pose_estimator/evaluate_pose_dataset.py \
  --checkpoint checkpoints/pose_estimator/cnn_6202_final.pth \
  --dataset "$out/data" --output "$out/frozen_cnn_report.json" \
  --batch_size 64 --workers 0 --device cuda:0 > "$out/frozen_cnn.log" 2>&1
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
