#!/usr/bin/env bash
# Run from RevoLab root on the existing Isaac Lab host.
set -euo pipefail
stage=${1:?usage: run_teacher_stage.sh smoke|baseline|hold|train}
export PYTHONPATH="source/BrainCo_DexHand:scripts/rsl_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/jiaju/tmp
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${TEACHER_GPU:-0}
runtime=/home/jiaju/.micromamba/envs/viserdex/bin/python
output=outputs/teacher_20260921
checkpoint=${TEACHER_CHECKPOINT:-logs/rsl_rl/brainco_hand/2026-09-20_18-22-13_revo3_viserdex_cube_formal/model_500.pt}
report_tag=${TEACHER_REPORT_TAG:-model500}
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
      --num_envs 2048 --seed 42 --max_iterations 1500 --resume \
      --load_run 2026-09-20_18-22-13_revo3_viserdex_cube_formal --checkpoint model_500.pt \
      --run_name revo3_teacher_hold02_s21 \
      env.goal_hold_time_s=0.2 env.hold_progress_reward_scale=2.0 \
      env.success_tolerance=0.16 agent.save_interval=100
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
