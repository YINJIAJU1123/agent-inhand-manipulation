#!/usr/bin/env bash
set -euo pipefail
root=/home/jiaju/DexManipulation/InHandManipulation5090
mkdir -p "$root/tools" "$root/logs"
if [[ ! -x "$root/tools/uv" ]]; then
    curl -fL --connect-timeout 10 --max-time 120 --retry 2 https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz -o "$root/tools/uv.tar.gz"
    tar -xzf "$root/tools/uv.tar.gz" -C "$root/tools" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
fi
uv="$root/tools/uv"
export UV_PYTHON_INSTALL_DIR="$root/tools/python"
python_bin="$root/tools/python311/bin/python3.11"
if [[ ! -x "$python_bin" ]]; then
    "$uv" python install 3.11
    python_bin=3.11
fi
[[ -x "$root/isaac51/bin/python" ]] || "$uv" venv --python "$python_bin" "$root/isaac51"
"$uv" pip install --python "$root/isaac51/bin/python" 'isaacsim[all,extscache]==5.1.0.0' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
"$uv" pip install --python "$root/isaac51/bin/python" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
[[ -d "$root/IsaacLab-2.3.2/source/isaaclab" ]] || git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$root/IsaacLab-2.3.2"
"$uv" pip install --python "$root/isaac51/bin/python" 'setuptools==80.9.0' wheel
"$uv" pip install --python "$root/isaac51/bin/python" --no-build-isolation -e "$root/IsaacLab-2.3.2/source/isaaclab" -e "$root/IsaacLab-2.3.2/source/isaaclab_tasks" -e "$root/IsaacLab-2.3.2/source/isaaclab_assets" -e "$root/IsaacLab-2.3.2/source/isaaclab_rl" 'rsl-rl-lib==3.1.1' pillow pytest tensorboard
"$uv" pip install --python "$root/isaac51/bin/python" 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128
"$root/isaac51/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
"$uv" pip freeze --python "$root/isaac51/bin/python" > "$root/logs/semantic_search_packages.txt"
echo BOOTSTRAP_COMPLETE
