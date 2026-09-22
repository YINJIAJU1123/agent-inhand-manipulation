#!/usr/bin/env bash
set -euo pipefail

# Full VisERDex-style Revo3 teacher protocol on the 5090/brainco host.
stage=${1:?usage: run_viserdex_teacher_v2.sh smoke|canary|train}
export PYTHONPATH="source/BrainCo_DexHand:scripts/rsl_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/jiaju/tmp
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${VISERDEX_GPU:-0}
runtime=/home/jiaju/.micromamba/envs/viserdex/bin/python

case "$stage" in
  smoke)
    timeout -k 30s 15m "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 16 --seed 123 --max_iterations 3 \
      --run_name viserdex_teacher_v2_smoke
    ;;
  canary)
    timeout -k 30s 2h "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 512 --seed 123 --max_iterations 150 \
      --run_name viserdex_teacher_v3_canary
    ;;
  train)
    timeout -k 30s 12h "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 2048 --seed 123 --max_iterations 2500 \
      --run_name viserdex_teacher_v2_full
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
