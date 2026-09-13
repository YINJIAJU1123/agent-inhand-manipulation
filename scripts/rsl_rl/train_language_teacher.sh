#!/usr/bin/env bash
set -euo pipefail

# Language-conditioned privileged teacher.  The task uses a six-dimensional
# structured goal vector (the same contract later rendered as text or encoded
# by CLIP/SigLIP) and keeps Revo3's 21-D action interface unchanged.
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1000}"

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT_DIR/source/BrainCo_DexHand:${PYTHONPATH:-}"
export OMNI_KIT_ALLOW_ROOT=1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR/isaaclab/logs" "$ROOT_DIR/logs/language_teacher"

exec python scripts/rsl_rl/train.py \
  --task BrainCo-Direct-Revo3-SemanticReorient-Cube-v0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --headless "$@"
