#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
: "${PROVIDER:?}"
: "${RUN_ID:?}"
out="$RUNTIME_ROOT/outputs/$RUN_ID"
mkdir "$out"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
project="$RUNTIME_ROOT/VLMmanipulation"
mkdir "$out/sources"
cp "${SOURCE_DIR:-$project/scripts/pose_latent_student_v1}/"*.py "$out/sources/"
export ASTRA_BRIDGE_URL="${ASTRA_BRIDGE_URL:-http://127.0.0.1:18781}"
export ASTRA_BRIDGE_KEY_FILE="$RUNTIME_ROOT/outputs/live_services_20261029/bridge.key"
common=(
  --project_root "$project"
  --student_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
  --checkpoint "$RUNTIME_ROOT/outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt"
  --text_cache "$project/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
  --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
  --pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth"
  --camera_calibration "$PWD/config/cameras"
  --actor_checkpoint "${ACTOR_CHECKPOINT:-$RUNTIME_ROOT/outputs/control_modules_20261029_v1/fit/maintenance_actor.pt}"
  --maintenance "${MAINTENANCE:-fixed_slow}" --provider "$PROVIDER"
  --vlm_url "${VLM_URL:-http://127.0.0.1:18782}"
  --rule_url "${RULE_URL:-http://127.0.0.1:18783}"
  --stop_mode "${STOP_MODE:-learned}"
  --episodes 20 --viz none --device cuda:0
  env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024
)
if [[ -n "${EXECUTION_ACTOR_CHECKPOINT:-}" ]]; then common+=(--execution_actor_checkpoint "$EXECUTION_ACTOR_CHECKPOINT"); fi
if [[ -n "${STOP_CHECKPOINT:-}" ]]; then common+=(--stop_checkpoint "$STOP_CHECKPOINT"); fi
if [[ -n "${STOP_CALIBRATION:-}" ]]; then
  cp "$STOP_CALIBRATION" "$out/stop_calibration.json"
  common+=(--stop_calibration "$out/stop_calibration.json")
fi
if [[ "${COMPLETION_DURING_WAIT:-0}" == 1 ]]; then common+=(--completion_during_wait); fi
seeds=(182921 182922)
if [[ -n "${FORMAL_RELEASE:-}" ]]; then
  : "${FORMAL_CONDITION:?A frozen condition is required}"
  cp "$FORMAL_RELEASE" "$out/formal_release.json"
  common+=(--formal_release "$out/formal_release.json" --formal_condition "$FORMAL_CONDITION")
  seeds=(183101 183102)
elif [[ -n "${FORMAL_CONDITION:-}" ]]; then
  echo 'FORMAL_CONDITION requires FORMAL_RELEASE' >&2
  exit 1
fi
expected_status=completed
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  [[ -z "${FORMAL_RELEASE:-}" ]] || { echo 'Preparation cannot use a formal release' >&2; exit 1; }
  common+=(--prepare_only)
  seeds=(182921)
  expected_status=prepared
fi
if [[ -n "${WAIT_FOR_RUN:-}" ]]; then
  stage=waiting_for_gpu_predecessor
  printf '%s\n' "$stage" > "$out/current_stage.txt"
  for predecessor in $WAIT_FOR_RUN; do
    while [[ ! -f "$RUNTIME_ROOT/outputs/$predecessor/process_status.txt" ]]; do sleep 20; done
    grep -qx 'exit_code=0' "$RUNTIME_ROOT/outputs/$predecessor/process_status.txt"
  done
fi
for seed in "${seeds[@]}"; do
  stage="${PROVIDER}_${seed}"
  printf '%s\n' "$stage" > "$out/current_stage.txt"
  "$VISERDEX_PYTHON" "$out/sources/run_logged.py" --log "$out/$stage.log" -- bash scripts/run_newton.sh "$out/sources/evaluate_agent.py" "${common[@]}" \
    --seed "$seed" --output "$out/$stage"
  "$VISERDEX_PYTHON" -c 'import json,sys; result=json.load(open(sys.argv[1])); assert result["status"]==sys.argv[2],result["status"]' "$out/$stage/result.json" "$expected_status"
done
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
