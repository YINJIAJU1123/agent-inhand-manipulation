#!/usr/bin/env bash
set -euo pipefail

# Revo3 semantic-surface state teacher.  CUDA_VISIBLE_DEVICES is intentionally
# configurable so several independent 5090 jobs can be launched safely.
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1000}"

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd -- "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT_DIR/source/BrainCo_DexHand:${PYTHONPATH:-}"
export OMNI_KIT_ALLOW_ROOT=1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR/isaaclab/logs" "$ROOT_DIR/logs/semantic_reorient"

exec python scripts/rsl_rl/train.py \
  --task BrainCo-Direct-Revo3-SemanticReorient-Cube-v0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --headless "$@"

