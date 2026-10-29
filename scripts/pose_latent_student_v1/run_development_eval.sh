#!/usr/bin/env bash
# Matched low-level diagnostics; does not compute autonomous agent success.
set -Eeuo pipefail
: "${RUNTIME_ROOT:?Set the independent runtime root}"
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with available capacity}"
: "${TRAINING_RUN:?Set the two-stage training output directory}"
: "${TRAINING_PID:?Set the live training supervisor PID}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
out="$RUNTIME_ROOT/outputs/pose_eval_$run_id"
mkdir "$out"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
release="$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
evaluator_directory="${EVALUATOR_DIRECTORY:-$RUNTIME_ROOT/VLMmanipulation/scripts/pose_latent_student_v1}"
mkdir "$out/sources"
cp "$evaluator_directory/"*.py "$out/sources/"
script="$out/sources/evaluate.py"
common=(
  --student_checkpoint "$release"
  --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
  --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
  --episodes 20 --viz none --device cuda:0
  env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024
)
if [[ "${POSE_SOURCE:-simulated}" == gs_cnn ]]; then
  common+=(--pose_source gs_cnn
    --pose_checkpoint "$RUNTIME_ROOT/viserDex_revo3/checkpoints/pose_estimator/cnn_6202_final.pth"
    --camera_calibration "$RUNTIME_ROOT/viserDex_revo3/config/cameras")
fi
if [[ -n "${WAIT_FOR_RUN:-}" ]]; then
  stage=waiting_for_gpu_predecessor
  printf '%s\n' "$stage" > "$out/current_stage.txt"
  for predecessor in $WAIT_FOR_RUN; do
    while [[ ! -f "$RUNTIME_ROOT/outputs/$predecessor/process_status.txt" ]]; do sleep 20; done
    grep -qx 'exit_code=0' "$RUNTIME_ROOT/outputs/$predecessor/process_status.txt"
  done
fi
seeds=(182901 182902)
if [[ -n "${FORMAL_RELEASE:-}" ]]; then
  [[ "${POSE_SOURCE:-simulated}" == gs_cnn ]] || { echo 'Formal policy diagnostics require POSE_SOURCE=gs_cnn' >&2; exit 1; }
  cp "$FORMAL_RELEASE" "$out/formal_release.json"
  common+=(--formal_release "$out/formal_release.json")
  seeds=(183101 183102)
fi
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  [[ -z "${FORMAL_RELEASE:-}" ]] || { echo 'Preparation cannot use a formal release' >&2; exit 1; }
  common+=(--prepare_only)
  seeds=(182901)
fi
for policy in released latent; do
  checkpoint="$release"
  if [[ "$policy" == latent ]]; then
    stage=waiting_for_training
    printf '%s\n' "$stage" > "$out/current_stage.txt"
    while [[ ! -f "$TRAINING_RUN/adapt/completion.json" ]]; do
      if [[ -f "$TRAINING_RUN/process_status.txt" ]] || ! kill -0 "$TRAINING_PID" 2>/dev/null; then
        echo 'Training stopped without an adaptation completion artifact' >&2
        exit 1
      fi
      sleep 20
    done
    "$VISERDEX_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["completed"]' "$TRAINING_RUN/adapt/completion.json"
    checkpoint="$TRAINING_RUN/adapt/latent_student_00400.pt"
  fi
  condition=()
  if [[ -n "${FORMAL_RELEASE:-}" ]]; then
    name=released_pose_goal_student
    if [[ "$policy" == latent ]]; then name=migrated_pose_latent_student; fi
    condition=(--formal_condition "$name")
  fi
  for seed in "${seeds[@]}"; do
    stage="${policy}_${seed}"
    printf '%s\n' "$stage" > "$out/current_stage.txt"
    "$VISERDEX_PYTHON" "$out/sources/run_logged.py" --log "$out/$stage.log" -- bash scripts/run_newton.sh "$script" "${common[@]}" --policy "$policy" \
      --checkpoint "$checkpoint" --seed "$seed" --output "$out/$stage" "${condition[@]}"
    if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
      "$VISERDEX_PYTHON" -c 'import json,sys; result=json.load(open(sys.argv[1])); assert result["status"]=="prepared" and result["recorded_episodes"]==0,result' "$out/$stage/result.json"
    fi
  done
done
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
