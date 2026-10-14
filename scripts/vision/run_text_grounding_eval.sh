#!/usr/bin/env bash
# Evaluate learned text-only grounding against canonical and wrong-command controls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${EVAL_PYTHON:?Set EVAL_PYTHON to the IsaacLab Python executable}"
: "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT}"
: "${GROUNDING_CHECKPOINT:?Set GROUNDING_CHECKPOINT}"
report_dir="${REPORT_DIR:-outputs/visual_student_v3/reports/text_grounding_v2}"
mkdir -p "$report_dir"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
common=(--checkpoint "$STUDENT_CHECKPOINT" --prompt-adapter "$GROUNDING_CHECKPOINT"
  --language-mode vlm --semantic-camera --semantic-grid-size 16 --semantic-depth-grid-size 16
  --vision-stride 2 --max-steps 600 --num_envs "${NUM_ENVS:-32}" --headless)
if [[ -n "${TEXT_FEATURE_CACHE:-}" ]]; then
  common+=(--text-feature-cache "$TEXT_FEATURE_CACHE")
else
  : "${SIGLIP_MODEL:?Set SIGLIP_MODEL or TEXT_FEATURE_CACHE}"
  common+=(--model "$SIGLIP_MODEL")
fi
for case_spec in 'canonical 123 0' 'heldout 123 0' 'heldout 456 0' 'heldout 123 1'; do
  read -r style seed offset <<< "$case_spec"
  name="${style}_seed${seed}_offset${offset}"
  report="$report_dir/$name.json"
  if [[ -e "$report" || -e "$report_dir/$name.log" ]]; then
    echo "Refusing to overwrite $name; choose a fresh REPORT_DIR" >&2
    exit 1
  fi
  count="${EPISODES:-384}"
  if [[ "$offset" != 0 ]]; then count="${CONTROL_EPISODES:-192}"; fi
  echo "START $name episodes=$count $(date -u +%FT%TZ)"
  "$EVAL_PYTHON" scripts/vision/evaluate_feature_student.py "${common[@]}" \
    --prompt-style "$style" --seed "$seed" --instruction-color-offset "$offset" \
    --episodes "$count" --report "$report" > "$report_dir/$name.log" 2>&1
  "$EVAL_PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); print({k:x[k] for k in ("episodes","success_rate","drop_rate","grounding_accuracy_against_scored_goal")})' "$report"
done
