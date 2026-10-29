#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?Set independent runtime root}"
: "${CUDA_VISIBLE_DEVICES:?Select a free GPU}"
: "${TRAINING_RUN:?Set adapted student training directory}"
: "${TRAINING_PID:?Set its supervisor PID}"
out="$RUNTIME_ROOT/outputs/control_collection_${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir "$out"
stage=waiting_for_student
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
printf '%s\n' "$stage" > "$out/current_stage.txt"
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
project="$RUNTIME_ROOT/VLMmanipulation"
mkdir "$out/sources"
cp "$project/scripts/pose_latent_student_v1/"*.py "$out/sources/"
while [[ ! -f "$TRAINING_RUN/adapt/completion.json" ]]; do
  if [[ -f "$TRAINING_RUN/process_status.txt" ]] || ! kill -0 "$TRAINING_PID" 2>/dev/null; then
    echo 'Student stopped without completed artifact' >&2
    exit 1
  fi
  sleep 20
done
"$VISERDEX_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["completed"]' "$TRAINING_RUN/adapt/completion.json"
if [[ -n "${WAIT_FOR_RUN:-}" ]]; then
  stage=waiting_for_gpu_predecessor
  printf '%s\n' "$stage" > "$out/current_stage.txt"
  for predecessor in $WAIT_FOR_RUN; do
    while [[ ! -f "$RUNTIME_ROOT/outputs/$predecessor/process_status.txt" ]]; do sleep 20; done
    grep -qx 'exit_code=0' "$RUNTIME_ROOT/outputs/$predecessor/process_status.txt"
  done
fi
cd "$RUNTIME_ROOT/viserDex_revo3"
common=(
  --student_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
  --checkpoint "$TRAINING_RUN/adapt/latent_student_00400.pt"
  --teacher_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/teacher/model_4800.pt"
  --text_cache "$project/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
  --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
  --pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth"
  --camera_calibration "$PWD/config/cameras"
  --num_envs "${NUM_ENVS:-16}" --stride "${STRIDE:-5}" --viz none --device cuda:0
  env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024
)
if [[ "${CONTINUOUS_PROTOCOL:-0}" == 1 ]]; then common+=(--continuous_protocol); fi
if [[ "${GROUNDING_CAPTURE:-0}" == 1 ]]; then common+=(--grounding_capture); fi
if [[ "${MAINTENANCE_WAIT_ONLY:-0}" == 1 ]]; then common+=(--maintenance_wait_only); fi
if [[ -n "${MAINTENANCE_CHECKPOINT:-}" ]]; then common+=(--maintenance_checkpoint "$MAINTENANCE_CHECKPOINT"); fi
if [[ -n "${LATENCY_PROFILE:-}" ]]; then common+=(--latency_profile "$LATENCY_PROFILE"); fi
for split in train validation; do
  stage="$split"
  printf '%s\n' "$stage" > "$out/current_stage.txt"
  seed="${TRAIN_SEED:-182831}"
  steps="${TRAIN_STEPS:-2400}"
  if [[ "$split" == validation ]]; then seed="${VALIDATION_SEED:-182832}"; steps="${VALIDATION_STEPS:-1200}"; fi
  "$VISERDEX_PYTHON" "$out/sources/run_logged.py" --log "$out/$split.log" -- bash scripts/run_newton.sh \
    "$out/sources/collect_control.py" "${common[@]}" --output "$out/$split" \
    --steps "$steps" --seed "$seed" --split "$split"
  if [[ "${GROUNDING_CAPTURE:-0}" == 1 ]]; then
    stage="label_$split"
    printf '%s\n' "$stage" > "$out/current_stage.txt"
    "$VISERDEX_PYTHON" "$out/sources/label_grounding.py" --dataset "$out/$split/grounding" \
      --output "$out/labels/$split" --split "$split" > "$out/label_$split.log" 2>&1
  fi
done
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
