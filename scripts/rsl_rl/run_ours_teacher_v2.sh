#!/usr/bin/env bash
set -euo pipefail

# Ours teacher: same VisERDex-style Revo3 interface, with hold-aware reward
# shaping and a slower stability curriculum.  The launcher is intentionally
# explicit about the source checkpoint and run name for paper traceability.
stage=${1:?usage: run_ours_teacher_v2.sh smoke|train}
export PYTHONPATH="source/BrainCo_DexHand:scripts/rsl_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=${TMPDIR:-/home/jiaju/tmp}
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${OURS_GPU:-0}
runtime=${ISAAC_PYTHON:-/home/jiaju/.micromamba/envs/viserdex/bin/python}

case "$stage" in
  smoke)
    timeout -k 30s 15m "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-OursTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs 16 --seed "${OURS_SEED:-123}" \
      --max_iterations 3 --run_name ours_teacher_v2_smoke
    ;;
  train)
    : "${OURS_LOAD_RUN:?set OURS_LOAD_RUN to the T1 source run directory name}"
    : "${OURS_CHECKPOINT:?set OURS_CHECKPOINT to the T1 source checkpoint filename}"
    timeout -k 30s "${OURS_TIMEOUT:-12h}" "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-OursTeacher-Cube-v0 \
      --headless --device cuda:0 --num_envs "${OURS_NUM_ENVS:-512}" \
      --seed "${OURS_SEED:-123}" --max_iterations "${OURS_MAX_ITERS:-1500}" \
      --resume --load_run "$OURS_LOAD_RUN" --checkpoint "$OURS_CHECKPOINT" \
      --run_name "${OURS_RUN_NAME:-ours_teacher_v2_hold05_s${OURS_SEED:-123}}"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
