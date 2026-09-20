#!/usr/bin/env bash
set -euo pipefail
base=/mnt/data_nas/jiaju/inhand_a100/20260920_multiobject
code=${SEARCH_CODE:-$base/code}
gpu=${SEARCH_GPU:?Set SEARCH_GPU}
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
script=${1:?Provide an rsl_rl Python script basename}
shift
case "$script" in train.py|evaluate_state_baseline.py) ;; *) exit 2 ;; esac
mkdir -p "$base/tmp_gpu$gpu"
exec 9>"$base/gpu${gpu}.lock"
flock -n 9 || exit 3
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 500 )) || { echo "GPU$gpu occupied: $used MiB" >&2; exit 4; }
source /mnt/workspace/xinyu/brainco_isaaclab_2_3_2_env/activate_brainco.sh
export CUDA_VISIBLE_DEVICES="$gpu" OMNI_KIT_ALLOW_ROOT=1 OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR="$base/tmp_gpu$gpu"
export PYTHONPATH="$code/source/BrainCo_DexHand:/mnt/workspace/xinyu/IsaacLab/source/isaaclab:/mnt/workspace/xinyu/IsaacLab/source/isaaclab_tasks:/mnt/workspace/xinyu/IsaacLab/source/isaaclab_assets:/mnt/workspace/xinyu/IsaacLab/source/isaaclab_rl:${PYTHONPATH:-}"
cd "$code"
exec python "scripts/rsl_rl/$script" --device cuda:0 --headless "$@"
