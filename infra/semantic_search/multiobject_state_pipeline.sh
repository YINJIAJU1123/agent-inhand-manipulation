#!/usr/bin/env bash
set -euo pipefail
base=/mnt/data_nas/jiaju/inhand_a100/20260920_multiobject
code=${SEARCH_CODE:?Set pinned SEARCH_CODE}
run=${SEARCH_RUN_ROOT:?Set fresh SEARCH_RUN_ROOT}
seed=${SEARCH_SEED:?Set SEARCH_SEED}
iterations=${SEARCH_ITERATIONS:-3000}
num_envs=${SEARCH_ENVS:-768}
test ! -e "$run"
mkdir -p "$run/training/source" "$run/evaluation"
git -C "$code" diff --exit-code
test -z "$(git -C "$code" status --porcelain)"
git -C "$code" rev-parse HEAD > "$run/code_revision.txt"
checkpoint=/mnt/data_nas/jiaju/inhand_a100/20260916/training/shaping_continue/2026-09-16_18-33-32_a100_continue/model_998.pt
printf '%s  %s\n' 42905dcdae5bb99fb2ed55e9dbfe103579a20c90ff8cf00e0c6342cea1874bf5 "$checkpoint" | sha256sum -c -
cp -n "$checkpoint" "$run/training/source/model_998.pt"
stage() {
    local name=$1 limit=$2
    shift 2
    printf '%s START %s\n' "$(date -Is)" "$name"
    set +e
    timeout --kill-after=60s "$limit" bash "$code/infra/semantic_search/run_a100.sh" "$@" > "$run/$name.log" 2>&1
    local result=$?
    set -e
    printf '%s\n' "$result" > "$run/$name.exit_code"
    printf '%s END %s exit=%s\n' "$(date -Is)" "$name" "$result"
    (( result == 0 )) || exit "$result"
}
stage train 12h train.py --task BrainCo-Direct-Revo3-SemanticReorient-Cube-v0 \
    --num_envs "$num_envs" --max_iterations "$iterations" --seed "$seed" --object-split train \
    --resume --load_run source --checkpoint model_998.pt --experiment_name "$run/training" \
    --run_name multiobject --logger tensorboard \
    env.goal_hold_time_s=0.5 env.hold_progress_reward_scale=4.0 \
    env.max_consecutive_success=0 env.freeze_goal_for_episode=false agent.save_interval=100
expected_iteration=$((998 + iterations - 1))
trained=$(find "$run/training" -mindepth 2 -maxdepth 2 -name "model_${expected_iteration}.pt" ! -path '*/source/*')
test -s "$trained"
printf '%s\n' "$trained" > "$run/final_checkpoint.txt"
for split in train val; do
    for eval_seed in 1000 1001 1002; do
        for face in 0 1 2 3 4 5; do
            name="eval_${split}_seed${eval_seed}_face${face}"
            stage "$name" 10m evaluate_state_baseline.py --checkpoint "$trained" \
                --object-split "$split" --num-envs 48 --episodes-per-face 96 \
                --faces "$face" --seeds "$eval_seed" --report "$run/evaluation/$name.json"
            test -s "$run/evaluation/$name.json"
        done
    done
done
python3 "$code/scripts/rsl_rl/summarize_object_evals.py" "$run/evaluation"
printf '%s PIPELINE_COMPLETE\n' "$(date -Is)"
