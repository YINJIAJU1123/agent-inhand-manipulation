#!/usr/bin/env bash
# Additional student DAgger with a fixed sensor source and fresh development states.
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
: "${SOURCE_DIR:?}"
: "${OUTPUT_DIR:?}"
: "${POSE_SOURCE:?}"
[[ "$POSE_SOURCE" == simulated || "$POSE_SOURCE" == gs_cnn ]] || exit 2
mkdir "$OUTPUT_DIR"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$OUTPUT_DIR/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
export MAX_JOBS=2
cd "$RUNTIME_ROOT/viserDex_revo3"
release="$RUNTIME_ROOT/prepared_release_20261029"
old_student="$RUNTIME_ROOT/outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt"
common=(
 --student_checkpoint "$release/student/student_1000.pt"
 --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
 --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
 --viz none --device cuda:0
 env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024
)
visual=(--pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth" --camera_calibration "$PWD/config/cameras")
stage=training
printf '%s\n' "$stage" > "$OUTPUT_DIR/current_stage.txt"
"$VISERDEX_PYTHON" "$SOURCE_DIR/run_logged.py" --log "$OUTPUT_DIR/fit.log" -- \
 bash scripts/run_newton.sh "$SOURCE_DIR/train.py" "${common[@]}" "${visual[@]}" \
 --stage adapt --initial_checkpoint "$old_student" --teacher_checkpoint "$release/teacher/model_4800.pt" \
 --pose_source "$POSE_SOURCE" --num_envs 16 --max_iterations 400 --rollout_steps 45 \
 --epochs 4 --env_batch_size 16 --learning_rate 0.0003 --save_interval 100 --seed 183201 \
 --output "$OUTPUT_DIR/fit"
"$VISERDEX_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["completed"] and d["updates"]==400,d' "$OUTPUT_DIR/fit/completion.json"
policies=(candidate)
if [[ "$POSE_SOURCE" == simulated ]]; then policies+=(released frozen_latent); fi
for name in "${policies[@]}"; do
 policy=latent
 checkpoint="$OUTPUT_DIR/fit/latent_student_00400.pt"
 if [[ "$name" == released ]]; then policy=released; checkpoint="$release/student/student_1000.pt"; fi
 if [[ "$name" == frozen_latent ]]; then checkpoint="$old_student"; fi
 for seed in 183221 183222; do
  stage="${name}_${seed}"
  printf '%s\n' "$stage" > "$OUTPUT_DIR/current_stage.txt"
  "$VISERDEX_PYTHON" "$SOURCE_DIR/run_logged.py" --log "$OUTPUT_DIR/$stage.log" -- \
   bash scripts/run_newton.sh "$SOURCE_DIR/evaluate.py" "${common[@]}" "${visual[@]}" \
   --pose_source gs_cnn --policy "$policy" --checkpoint "$checkpoint" \
   --episodes 20 --seed "$seed" --output "$OUTPUT_DIR/$stage"
  "$VISERDEX_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="completed" and d["recorded_episodes"]==20,d' "$OUTPUT_DIR/$stage/result.json"
 done
done
stage=finished
printf '%s\n' "$stage" > "$OUTPUT_DIR/current_stage.txt"
