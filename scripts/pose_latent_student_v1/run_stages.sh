#!/usr/bin/env bash
# Run the two migration stages sequentially in the prepared independent runtime.
set -Eeuo pipefail
: "${RUNTIME_ROOT:?Set the independent runtime root}"
: "${CUDA_VISIBLE_DEVICES:?Select the physical GPU after checking its free capacity}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
release="${RELEASE_DIR:-$RUNTIME_ROOT/prepared_release_20261029}"
out="$RUNTIME_ROOT/outputs/pose_latent_$run_id"
mkdir "$out"
stage=setup
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
script="$RUNTIME_ROOT/VLMmanipulation/scripts/pose_latent_student_v1/train.py"
common=(
  --student_checkpoint "$release/student/student_1000.pt"
  --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
  --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
  --num_envs 128 --rollout_steps 45 --epochs 4 --env_batch_size 32
  --save_interval 20 --seed 182801 --viz none --device cuda:0
)
stage=interface
printf '%s\n' "$stage" > "$out/current_stage.txt"
bash scripts/run_newton.sh "$script" "${common[@]}" \
  --stage interface --max_iterations 200 --output "$out/interface" > "$out/interface.log" 2>&1
"$VISERDEX_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["completed"]' "$out/interface/completion.json"
stage=adapt
printf '%s\n' "$stage" > "$out/current_stage.txt"
bash scripts/run_newton.sh "$script" "${common[@]}" \
  --stage adapt --max_iterations 400 --output "$out/adapt" \
  --initial_checkpoint "$out/interface/latent_student_00200.pt" \
  --teacher_checkpoint "$release/teacher/model_4800.pt" > "$out/adapt.log" 2>&1
"$VISERDEX_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["completed"]' "$out/adapt/completion.json"
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
