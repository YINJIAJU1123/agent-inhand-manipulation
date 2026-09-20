#!/usr/bin/env bash
set -euo pipefail
root=/home/jiaju/DexManipulation/InHandManipulation5090
code=${SEARCH_CODE:?Set pinned SEARCH_CODE}
run=${SEARCH_RUN_ROOT:?Set fresh SEARCH_RUN_ROOT}
gpu=${SEARCH_GPU:-3}
seed=${SEARCH_SEED:-0}
iterations=${SEARCH_ITERATIONS:-3000}
num_envs=${SEARCH_ENVS:-96}
test ! -e "$run"
mkdir -p "$run"
exec 9>"$root/search_gpu${gpu}.lock"
printf '%s WAIT_GPU_LOCK gpu=%s\n' "$(date -Is)" "$gpu"
# The existing single-object pipeline owns this same lock. Never interrupt it.
flock -w 86400 9 || exit 2
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 500 )) || { echo "GPU$gpu occupied: $used MiB" >&2; exit 3; }
test -z "$(git -C "$code" status --porcelain)"
git -C "$code" rev-parse HEAD > "$run/code_revision.txt"
stage() {
    local name=$1
    shift
    printf '%s START %s\n' "$(date -Is)" "$name"
    set +e
    timeout --kill-after=30s 18h bash "$code/infra/semantic_search/run_5090.sh" \
        "$@" --output "$run/$name" > "$run/$name.log" 2>&1
    local result=$?
    set -e
    printf '%s\n' "$result" > "$run/$name.exit_code"
    printf '%s END %s exit=%s\n' "$(date -Is)" "$name" "$result"
    (( result == 0 )) || exit "$result"
    test -s "$run/$name/status.json"
}
stage smoke --mode smoke --object-split train --num_envs 6 --seed "$seed"
stage scale_smoke --mode train --object-split train --num_envs "$num_envs" --iterations 3 --seed "$seed"
if [[ ${SEARCH_VALIDATE_ONLY:-0} == 1 ]]; then
    printf '%s VALIDATION_COMPLETE\n' "$(date -Is)"
    exit 0
fi
for initial in visible mixed; do
    stage "${initial}_train" --mode train --object-split train --initial "$initial" \
        --split train --num_envs "$num_envs" --iterations "$iterations" --seed "$seed"
    checkpoint="$run/${initial}_train/final.pt"
    for objects in train val; do
        for visibility in visible hidden; do
            stage "${initial}_${objects}_${visibility}" --mode eval --object-split "$objects" \
                --initial "$visibility" --split val --num_envs 12 --episodes 96 \
                --seed 1000 --checkpoint "$checkpoint"
        done
        if [[ "$initial" == visible ]]; then
            stage "scan_${objects}_hidden" --mode scan --object-split "$objects" --initial hidden \
                --split val --num_envs 12 --episodes 96 --seed 1000 --checkpoint "$checkpoint"
        fi
    done
done
printf '%s PIPELINE_COMPLETE\n' "$(date -Is)"
