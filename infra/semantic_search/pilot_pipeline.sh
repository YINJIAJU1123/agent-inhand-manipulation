#!/usr/bin/env bash
set -euo pipefail
root=/home/jiaju/DexManipulation/InHandManipulation5090
code=${SEARCH_CODE:-$root/project/semantic_search_v0}
run_root=${SEARCH_RUN_ROOT:?Set a fresh SEARCH_RUN_ROOT}
gpu=${SEARCH_GPU:-3}
iterations=${SEARCH_ITERATIONS:-500}
num_envs=${SEARCH_ENVS:-64}
seed=${SEARCH_SEED:-0}
mkdir -p "$run_root"
exec 9>"$root/search_gpu${gpu}.lock"
flock -n 9 || { echo 'Another search worker holds this GPU lock.' >&2; exit 2; }
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 500 )); then
    echo "GPU $gpu is occupied ($used MiB); refusing to start." >&2
    exit 3
fi
git -C "$code" rev-parse HEAD > "$run_root/code_revision.txt"
git -C "$code" diff --exit-code
run() {
    local name=$1
    shift
    printf '%s START %s\n' "$(date -Is)" "$name"
    set +e
    timeout --kill-after=30s 8h bash "$code/infra/semantic_search/run_5090.sh" \
        "$@" --output "$run_root/$name" > "$run_root/$name.log" 2>&1
    local status=$?
    set -e
    printf '%s\n' "$status" > "$run_root/$name.exit_code"
    printf '%s END %s exit=%s\n' "$(date -Is)" "$name" "$status"
    if (( status != 0 )); then exit "$status"; fi
    test -s "$run_root/$name/status.json"
}
run visible_train --mode train --initial visible --split train --num_envs "$num_envs" --iterations "$iterations" --seed "$seed"
for initial in visible hidden; do
    run "visible_controller_${initial}" --mode eval --initial "$initial" --split val --num_envs 8 --episodes 32 --seed 1000 --checkpoint "$run_root/visible_train/final.pt"
done
run scan_hidden --mode scan --initial hidden --split val --num_envs 8 --episodes 32 --seed 1000 --checkpoint "$run_root/visible_train/final.pt"
run mixed_train --mode train --initial mixed --split train --num_envs "$num_envs" --iterations "$iterations" --seed "$seed"
for initial in visible hidden; do
    run "mixed_controller_${initial}" --mode eval --initial "$initial" --split val --num_envs 8 --episodes 32 --seed 1000 --checkpoint "$run_root/mixed_train/final.pt"
done
printf '%s PIPELINE_COMPLETE\n' "$(date -Is)"
