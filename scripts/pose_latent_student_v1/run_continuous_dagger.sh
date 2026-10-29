#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
out="$RUNTIME_ROOT/outputs/control_continuous_dagger_20261029_v2"
mkdir "$out"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
project="$RUNTIME_ROOT/VLMmanipulation"
mkdir "$out/sources"
cp "$project/scripts/pose_latent_student_v1/"*.py "$out/sources/"
common=(
 --student_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
 --checkpoint "$RUNTIME_ROOT/outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt"
 --teacher_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/teacher/model_4800.pt"
 --text_cache "$project/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
 --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
 --pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth"
 --camera_calibration "$PWD/config/cameras"
 --maintenance_checkpoint "$RUNTIME_ROOT/outputs/control_modules_20261029_v1/fit/maintenance_actor.pt"
 --latency_profile "$project/outputs/pose_latent_student_v1/empirical_latency_20261029.json"
 --continuous_protocol --num_envs 10 --stride 1 --viz none --device cuda:0
)
for split in train validation; do
 stage="collect_$split"
 printf '%s\n' "$stage" > "$out/current_stage.txt"
 seed=182861
 steps=4800
 if [[ "$split" == validation ]]; then seed=182862; steps=2400; fi
 bash scripts/run_newton.sh "$out/sources/collect_control.py" "${common[@]}" \
  --seed "$seed" --steps "$steps" --split "$split" --output "$out/$split" > "$out/$stage.log" 2>&1
done
stage=fit_dagger
printf '%s\n' "$stage" > "$out/current_stage.txt"
"$VISERDEX_PYTHON" "$out/sources/train_modules.py" --data "$out" \
 --text_cache "$project/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt" \
 --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda" \
 --actor_warm_start "$RUNTIME_ROOT/outputs/control_modules_20261029_v1/fit/maintenance_actor.pt" \
 --output "$out/fit" --updates 2400 --seed 182871 > "$out/fit.log" 2>&1
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
