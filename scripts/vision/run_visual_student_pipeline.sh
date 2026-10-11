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
raw_data=${VISUAL_RAW_DATA:-$output/rollouts_merged.pt}
source_data=${VISUAL_SOURCE_DATA:-$output/rollouts_filtered.pt}
language_mode=${VISUAL_LANGUAGE_MODE:-onehot}
language_model=${VISUAL_LANGUAGE_MODEL:-google/siglip2-base-patch16-224}
base_cache=${VISUAL_BASE_CACHE:-}
semantic_cache=${VISUAL_SEMANTIC_CACHE:-$output/semantic_rgbd_features.pt}
cache=${VISUAL_FEATURE_CACHE:-$output/${language_mode}_features.pt}
action_scale=${VISUAL_ACTION_SCALE:-1.0}
task=${VISUAL_TASK:-BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0}
num_envs=${VISUAL_NUM_ENVS:-64}
episodes=${VISUAL_EPISODES:-387}
stride=${VISUAL_STRIDE:-2}
quality_policy=${VISUAL_QUALITY_POLICY:-success_only}
split_manifest=${VISUAL_SPLIT_MANIFEST:-$output/episode_split.json}
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

audit_rollouts() {
  local data="$1"
  local report="$2"
  "$python_bin" scripts/vision/audit_visual_rollouts.py --data "$data" --report "$report" --expected-stride "$stride"
}

filter_rollouts() {
  "$python_bin" scripts/vision/filter_visual_rollouts.py \
    --data "$raw_data" --output "$source_data" --policy "$quality_policy" \
    --report "$output/rollouts_filtered.json"
}

cache_features() {
  # Always build the deterministic RGB-D cache first.  Language modes are
  # overlays, so one-hot/hash/SigLIP experiments share identical visual
  # features, actions, episode splits and freeze metadata.
  if [[ -n "$base_cache" ]]; then
    cp "$base_cache" "$semantic_cache"
  else
    "$python_bin" scripts/vision/cache_semantic_features.py \
      --data "$source_data" --output "$semantic_cache"
  fi
  case "$language_mode" in
    onehot)
      if [[ "$semantic_cache" != "$cache" ]]; then cp "$semantic_cache" "$cache"; fi
      ;;
    hash)
      "$python_bin" scripts/vision/cache_hash_text_features.py \
        --data "$semantic_cache" --output "$cache"
      ;;
    vlm)
      canonical_flag=()
      if [[ "${VISUAL_CANONICAL_TEXT:-0}" == "1" ]]; then canonical_flag+=(--canonical-from-target-face); fi
      "$python_bin" scripts/vision/cache_siglip_text_features.py \
        --data "$semantic_cache" --output "$cache" --model "$language_model" "${canonical_flag[@]}"
      ;;
    *) echo "unsupported VISUAL_LANGUAGE_MODE: $language_mode" >&2; exit 2 ;;
  esac
}

create_split() {
  "$python_bin" scripts/vision/create_episode_split.py \
    --data "$cache" --output "$split_manifest" --seed "${VISUAL_SPLIT_SEED:-0}"
}

train_one() {
  local mode="$1"
  local seed="$2"
  local prefix="$output/student_${mode}_seed${seed}"
  "$python_bin" scripts/vision/train_feature_student.py \
    --data "$cache" --output "${prefix}.pt" \
    --seed "$seed" --memory-mode "$mode" --freeze-manifest "$manifest" \
    --split-manifest "$split_manifest"
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
    --language-mode "$language_mode" --model "$language_model" \
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
    "$python_bin" scripts/vision/merge_rollout_shards.py --output "$raw_data" "$output"/rollout_seed*.pt
    audit_rollouts "$raw_data" "$output/rollouts_raw_audit.json"
    filter_rollouts
    audit_rollouts "$source_data" "$output/rollouts_filtered_audit.json"
    cache_features
    create_split
    for mode_name in ${VISUAL_MEMORY_MODES:-plain evidence}; do
      for seed in ${VISUAL_TRAIN_SEEDS:-0 1 2}; do train_one "$mode_name" "$seed"; done
    done
    for mode_name in ${VISUAL_MEMORY_MODES:-plain evidence}; do
      for seed in ${VISUAL_TRAIN_SEEDS:-0 1 2}; do live_eval_one "$mode_name" "$seed"; done
    done
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
