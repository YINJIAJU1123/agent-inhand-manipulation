#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}" "${STUDY_ROOT:?}" "${CONDITION:?}" "${CUDA_VISIBLE_DEVICES:?}" "${STAGE:?}"
case "$CONDITION" in
 correct_active_goal) mode=active; intervention=none;;
 retrained_no_explicit_goal) mode=center; intervention=none;;
 shuffled_controller_goal) mode=active; intervention=cyclic_shift;;
 *) exit 2;;
esac
case "$STAGE" in smoke|formal) ;; *) exit 2;; esac
out="$STUDY_ROOT/$STAGE/$CONDITION"
mkdir -p "$STUDY_ROOT/$STAGE"
mkdir "$out"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
export ASTRA_BRIDGE_URL=http://127.0.0.1:18812
export ASTRA_BRIDGE_KEY_FILE="$RUNTIME_ROOT/outputs/live_services_20261029/bridge.key"
common=(--project_root "$RUNTIME_ROOT/VLMmanipulation"
 --student_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
 --checkpoint "$RUNTIME_ROOT/outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt"
 --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
 --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
 --pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth"
 --camera_calibration "$PWD/config/cameras"
 --stop_mode measured --viz none --device cuda:0 --maintenance learned
 --actor_checkpoint "$STUDY_ROOT/weights/waiting_${mode}/maintenance_actor.pt"
 --execution_actor_checkpoint "$STUDY_ROOT/weights/execution_${mode}/maintenance_actor.pt"
 --controller_goal_intervention "$intervention"
 env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024)
if [[ "$STAGE" == smoke ]]; then
 seeds=(182941); episodes=2
 common+=(--provider replay --replay_plan "$STUDY_ROOT/smoke_plan.json")
else
 seeds=(183421 183422); episodes=20
 common+=(--provider astra --formal_release "$STUDY_ROOT/formal_release.json" --formal_condition "$CONDITION")
fi
for seed in "${seeds[@]}"; do
 printf '%s\n' "$seed" > "$out/current_seed.txt"
 "$VISERDEX_PYTHON" "$STUDY_ROOT/sources/run_logged.py" --log "$out/${seed}.log" -- \
  bash scripts/run_newton.sh "$STUDY_ROOT/sources/evaluate_agent.py" "${common[@]}" \
  --episodes "$episodes" --seed "$seed" --output "$out/$seed"
 "$VISERDEX_PYTHON" -c 'import json,sys;r=json.load(open(sys.argv[1]));assert r["status"]=="completed" and r["recorded_episodes"]==int(sys.argv[2]),r' "$out/$seed/result.json" "$episodes"
done
