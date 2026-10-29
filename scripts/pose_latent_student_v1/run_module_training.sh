#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?Set independent runtime root}"
: "${CUDA_VISIBLE_DEVICES:?Select a free GPU}"
: "${COLLECTION_RUN:?Set collection output directory}"
: "${COLLECTION_PID:?Set live collection supervisor PID}"
out="$RUNTIME_ROOT/outputs/control_modules_${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir "$out"
stage=waiting_for_collection
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
printf '%s\n' "$stage" > "$out/current_stage.txt"
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
while [[ ! -f "$COLLECTION_RUN/validation/completion.json" ]]; do
  if [[ -f "$COLLECTION_RUN/process_status.txt" ]] || ! kill -0 "$COLLECTION_PID" 2>/dev/null; then
    echo 'Collection stopped before validation completed' >&2
    exit 1
  fi
  sleep 20
done
stage=training
printf '%s\n' "$stage" > "$out/current_stage.txt"
"$VISERDEX_PYTHON" "$RUNTIME_ROOT/VLMmanipulation/scripts/pose_latent_student_v1/train_modules.py" \
 --data "$COLLECTION_RUN" \
 --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt" \
 --original_asset "$RUNTIME_ROOT/viserDex_revo3/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda" \
 --output "$out/fit" --updates 1200 --seed 182841 > "$out/training.log" 2>&1
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
