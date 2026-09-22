#!/usr/bin/env bash
set -euo pipefail

# Full VisERDex-style Revo3 teacher protocol on the 5090/brainco host.
stage=${1:?usage: run_viserdex_teacher_v2.sh smoke|canary|bootstrap|adapt|train}
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
  bootstrap)
    # First learn the same goal-conditioned task with deterministic, responsive
    # actuation.  The full protocol is resumed from this checkpoint only after
    # the orientation metric moves; this separates PPO/task issues from
    # randomized actuator dynamics.
    timeout -k 30s 6h "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 512 --seed 123 --max_iterations 300 \
      --run_name viserdex_teacher_v4_bootstrap \
      env.success_tolerance=0.16 env.ema_alpha_min=0.35 env.ema_alpha_max=0.35 \
      env.action_delay_min_steps=0 env.action_delay_max_steps=0
    ;;
  adapt)
    : "${TEACHER_LOAD_RUN:?set TEACHER_LOAD_RUN to the source run directory name}"
    : "${TEACHER_CHECKPOINT:?set TEACHER_CHECKPOINT to the source checkpoint filename}"
    timeout -k 30s 12h "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 512 --seed 123 --max_iterations 1500 \
      --resume --load_run "$TEACHER_LOAD_RUN" --checkpoint "$TEACHER_CHECKPOINT" \
      --run_name viserdex_teacher_v5_scaled
    ;;
  train)
    timeout -k 30s 12h "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 2048 --seed 123 --max_iterations 2500 \
      --run_name viserdex_teacher_v3_fixed
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
