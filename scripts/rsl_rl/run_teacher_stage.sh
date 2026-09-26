#!/usr/bin/env bash
# Run from RevoLab root on the existing Isaac Lab host.
set -euo pipefail
stage=${1:?usage: run_teacher_stage.sh smoke|baseline|hold|train}
export PYTHONPATH="source/BrainCo_DexHand:scripts/rsl_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES
teacher_tmpdir=${TEACHER_TMPDIR:-${TMPDIR:-/tmp/viserdex-teacher}}
export TMPDIR="$teacher_tmpdir"
mkdir -p "$teacher_tmpdir"
export CUDA_VISIBLE_DEVICES=${TEACHER_GPU:-0}
runtime=${TEACHER_PYTHON:-/home/jiaju/.micromamba/envs/viserdex/bin/python}
if [[ ! -x "$runtime" ]]; then
  runtime=${PYTHON_BIN:-python}
fi
output=${TEACHER_OUTPUT:-outputs/teacher_20260921}
checkpoint=${TEACHER_CHECKPOINT:-logs/rsl_rl/brainco_hand/2026-09-20_18-22-13_revo3_viserdex_cube_formal/model_500.pt}
report_tag=${TEACHER_REPORT_TAG:-model500}
load_run=${TEACHER_LOAD_RUN:-2026-09-20_18-22-13_revo3_viserdex_cube_formal}
load_checkpoint=${TEACHER_LOAD_CHECKPOINT:-model_500.pt}
run_name=${TEACHER_RUN_NAME:-revo3_teacher_hold02_s21}
num_envs=${TEACHER_NUM_ENVS:-2048}
max_iterations=${TEACHER_MAX_ITERATIONS:-1500}
hold_time_s=${TEACHER_HOLD_TIME_S:-0.2}
hold_reward_scale=${TEACHER_HOLD_PROGRESS_REWARD_SCALE:-2.0}
mkdir -p "$output"
common=(--headless --device cuda:0 --checkpoint "$checkpoint")
case "$stage" in
  smoke)
    timeout -k 30s 10m "$runtime" -u scripts/rsl_rl/evaluate_state_baseline.py "${common[@]}" \
      --mode repeated --success-tolerance 0.4 --episode-length-s 3 \
      --episodes-per-face 8 --num-envs 8 --seeds 101 --report "$output/smoke.json"
    ;;
  baseline)
    timeout -k 30s 20m "$runtime" -u scripts/rsl_rl/evaluate_state_baseline.py "${common[@]}" \
      --mode repeated --success-tolerance 0.4 --episode-length-s 30 \
      --episodes-per-face 128 --num-envs 128 --seeds 101 --report "$output/${report_tag}_repeated30s.json"
    ;;
  hold)
    # A separate process per face avoids reinitializing the Kit scene in one app.
    for face in 0 1 2 3 4 5; do
      timeout -k 30s 10m "$runtime" -u scripts/rsl_rl/evaluate_state_baseline.py "${common[@]}" \
        --mode hold --faces "$face" --episode-length-s 10 \
        --episodes-per-face 64 --num-envs 64 --seeds 101 \
        --report "$output/${report_tag}_hold_face${face}.json"
    done
    ;;
  train)
    timeout -k 30s 12h "$runtime" -u scripts/rsl_rl/train.py \
      --task BrainCo-Direct-Revo3-SemanticReorient-Cube-v0 --headless --device cuda:0 \
      --num_envs "$num_envs" --seed "${TEACHER_SEED:-42}" --max_iterations "$max_iterations" --resume \
      --load_run "$load_run" --checkpoint "$load_checkpoint" \
      --run_name "$run_name" \
      env.goal_hold_time_s="$hold_time_s" env.hold_progress_reward_scale="$hold_reward_scale" \
      env.success_tolerance=0.16 agent.save_interval=100
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
