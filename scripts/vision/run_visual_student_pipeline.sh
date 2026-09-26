#!/usr/bin/env bash
# Reproducible visual-student pipeline for the frozen v1 contract.
set -euo pipefail

mode=${1:?usage: run_visual_student_pipeline.sh collect|cache|train|eval|all}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
python_bin=${VISUAL_PYTHON:-${PYTHON_BIN:-}}
if [[ -z "$python_bin" ]]; then
  for candidate in /home/jiaju/.micromamba/envs/viserdex/bin/python /opt/isaaclab-env/bin/python python; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      python_bin="$candidate"
      break
    fi
  done
fi
: "${python_bin:?no Python runtime found; set VISUAL_PYTHON}"
manifest=${VISUAL_FREEZE_MANIFEST:-$root/configs/visual_student_freeze.json}
output=${VISUAL_OUTPUT:-$root/outputs/visual_student_v1}
checkpoint=${TEACHER_CHECKPOINT:?set TEACHER_CHECKPOINT to the frozen model_1999.pt}
source_data=${VISUAL_SOURCE_DATA:-$output/rollouts_merged.pt}
cache=${VISUAL_FEATURE_CACHE:-$output/semantic_features.pt}
action_scale=${VISUAL_ACTION_SCALE:-6.0}
task=${VISUAL_TASK:-BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0}
num_envs=${VISUAL_NUM_ENVS:-64}
episodes=${VISUAL_EPISODES:-387}
stride=${VISUAL_STRIDE:-2}
mkdir -p "$output"
if [[ ! -f "$checkpoint" ]]; then
  echo "frozen teacher checkpoint not found: $checkpoint" >&2
  exit 3
fi
expected_sha=$("$python_bin" - "$manifest" <<'PYJSON'
import json, sys
print(json.load(open(sys.argv[1]))["teacher"]["sha256"])
PYJSON
)
actual_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "teacher checkpoint SHA-256 mismatch: expected $expected_sha got $actual_sha" >&2
  exit 3
fi
manifest_scale=$("$python_bin" - "$manifest" "$action_scale" <<'PYJSON'
import json, sys
manifest_scale = float(json.load(open(sys.argv[1]))["action_contract"]["action_scale"])
requested_scale = float(sys.argv[2])
if abs(manifest_scale - requested_scale) > 1e-8:
    raise SystemExit(f"action scale {requested_scale} disagrees with freeze manifest {manifest_scale}")
print(manifest_scale)
PYJSON
)

collect() {
  "$python_bin" scripts/rsl_rl/collect_visual_rollouts.py \
    --task "$task" --checkpoint "$checkpoint" --episodes "$episodes" \
    --num_envs "$num_envs" --stride "$stride" --seed "${VISUAL_COLLECT_SEED:-0}" \
    --freeze-manifest "$manifest" --goal-yaw "${VISUAL_GOAL_YAW:-0.0}" \
    --output "$output/rollout_seed${VISUAL_COLLECT_SEED:-0}.pt" --headless --enable_cameras
}

cache_features() {
  "$python_bin" scripts/vision/cache_semantic_features.py \
    --data "$source_data" --output "$cache"
}

train_one() {
  local mode="$1"
  local seed="$2"
  local prefix="$output/student_${mode}_seed${seed}"
  "$python_bin" scripts/vision/train_feature_student.py \
    --data "$cache" --output "${prefix}.pt" \
    --seed "$seed" --memory-mode "$mode" --freeze-manifest "$manifest"
  "$python_bin" scripts/vision/evaluate_offline_student.py \
    --data "$cache" --checkpoint "${prefix}.pt" \
    --seed "$seed" --report "${prefix}_offline.json"
}

live_eval_one() {
  local mode="$1"
  local seed="$2"
  local prefix="$output/student_${mode}_seed${seed}"
  "$python_bin" scripts/vision/evaluate_feature_student.py \
    --checkpoint "${prefix}.pt" --task "$task" \
    --semantic-camera --episodes "${VISUAL_EVAL_EPISODES:-72}" \
    --num_envs "${VISUAL_EVAL_ENVS:-4}" --max-steps "${VISUAL_EVAL_STEPS:-600}" \
    --vision-stride "$stride" --action-scale "$action_scale" \
    --freeze-manifest "$manifest" --report "${prefix}_live.json" \
    --headless --enable_cameras
}

case "$mode" in
  collect) collect ;;
  cache) cache_features ;;
  train)
    for mode_name in ${VISUAL_MEMORY_MODES:-plain evidence}; do
      for seed in ${VISUAL_TRAIN_SEEDS:-0 1 2}; do train_one "$mode_name" "$seed"; done
    done
    ;;
  eval)
    for mode_name in ${VISUAL_MEMORY_MODES:-plain evidence}; do
      for seed in ${VISUAL_TRAIN_SEEDS:-0 1 2}; do live_eval_one "$mode_name" "$seed"; done
    done
    ;;
  all)
    collect
    "$python_bin" scripts/vision/merge_rollout_shards.py --output "$source_data" "$output"/rollout_seed*.pt
    cache_features
    for mode_name in ${VISUAL_MEMORY_MODES:-plain evidence}; do
      for seed in ${VISUAL_TRAIN_SEEDS:-0 1 2}; do train_one "$mode_name" "$seed"; done
    done
    for mode_name in ${VISUAL_MEMORY_MODES:-plain evidence}; do
      for seed in ${VISUAL_TRAIN_SEEDS:-0 1 2}; do live_eval_one "$mode_name" "$seed"; done
    done
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
