#!/usr/bin/env bash
set -euo pipefail
root=/home/jiaju/DexManipulation/InHandManipulation5090
gpu=${SEARCH_GPU:-3}
code="$root/project/semantic_search_v0"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$code/source/BrainCo_DexHand"
export CUDA_VISIBLE_DEVICES="$gpu"
export XDG_CACHE_HOME="$root/cache"
export TMPDIR="$root/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd "$code"
exec "$root/isaac51/bin/python" -u scripts/rsl_rl/run_semantic_search.py \
    --headless --device cuda:0 "$@" \
    --kit_args "--/renderer/activeGpu=$gpu --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
