#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}" "${STUDY_ROOT:?}" "${CONDITION:?}" "${CUDA_VISIBLE_DEVICES:?}"
case "$CONDITION" in
 astra_full) provider=astra;;
 finetuned_vlm_bare) provider=vlm;;
 rule_vlm_bare) provider=rule;;
 *) exit 2;;
esac
out="$STUDY_ROOT/runs/$CONDITION"
mkdir -p "$STUDY_ROOT/runs"
mkdir "$out"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
export ASTRA_BRIDGE_URL=http://127.0.0.1:18811
export ASTRA_BRIDGE_KEY_FILE="$RUNTIME_ROOT/outputs/live_services_20261029/bridge.key"
common=(--project_root "$RUNTIME_ROOT/VLMmanipulation"
 --student_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
 --checkpoint "$RUNTIME_ROOT/outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt"
 --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
 --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
 --pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth"
 --camera_calibration "$PWD/config/cameras" --provider "$provider"
 --vlm_url http://127.0.0.1:18794 --rule_url http://127.0.0.1:18795
 --stop_mode measured --episodes 20 --viz none --device cuda:0
 --formal_release "$STUDY_ROOT/formal_release.json" --formal_condition "$CONDITION"
 env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024)
if [[ "$CONDITION" == astra_full ]]; then
 common+=(--maintenance learned
  --actor_checkpoint "$RUNTIME_ROOT/outputs/control_modules_20261029_v1/fit/maintenance_actor.pt"
  --execution_actor_checkpoint "$RUNTIME_ROOT/outputs/control_agent_schedule_20261029_v6/fit/maintenance_actor.pt")
else
 common+=(--maintenance continue)
fi
for seed in 183411 183412; do
 printf '%s\n' "$seed" > "$out/current_seed.txt"
 "$VISERDEX_PYTHON" "$STUDY_ROOT/sources/run_logged.py" --log "$out/${provider}_${seed}.log" -- \
  bash scripts/run_newton.sh "$STUDY_ROOT/sources/evaluate_agent.py" "${common[@]}" \
  --seed "$seed" --output "$out/${provider}_${seed}"
 "$VISERDEX_PYTHON" -c 'import json,sys;r=json.load(open(sys.argv[1]));assert r["status"]=="completed" and r["recorded_episodes"]==20,r' "$out/${provider}_${seed}/result.json"
done
