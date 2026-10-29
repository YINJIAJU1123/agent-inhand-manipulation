#!/usr/bin/env bash
# Episode-disjoint rendered data for RSLCube semantic interface adaptation.
set -Eeuo pipefail
: "${RUNTIME_ROOT:?Set the independent runtime root}"
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with available capacity}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
out="$RUNTIME_ROOT/outputs/grounding_rgb_$run_id"
mkdir "$out"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
export MAX_JOBS="${MAX_JOBS:-2}"
cd "$RUNTIME_ROOT/viserDex_revo3"
for split in train validation; do
  stage="$split"
  seed=182811
  samples=4096
  if [[ "$split" == validation ]]; then seed=182812; samples=1024; fi
  printf '%s\n' "$stage" > "$out/current_stage.txt"
  bash scripts/run_newton.sh scripts/pose_estimator/collect_revo3_occlusion.py \
    --task Isaac-ViserDex-Repose-Revo3 \
    --teacher_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/teacher/model_4800.pt" \
    --hikrobot_calibration config/cameras \
    --num_envs 16 --samples "$samples" --max_steps 5000 --stride 4 --burn_in 120 \
    --seed "$seed" --output_dir "$out/$split" --viz none --device cuda:0 > "$out/$split.log" 2>&1
done
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
