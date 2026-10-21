#!/usr/bin/env bash
set -Eeuo pipefail
unset DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
python_bin=''
for candidate in python3 python /isaac-sim/python.sh /opt/isaacsim/python.sh; do
    if (command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]) && "$candidate" -c 'import isaaclab,torch' >/dev/null 2>&1; then
        python_bin="$candidate"
        break
    fi
done
[[ -n "$python_bin" ]]
exec "$python_bin" run_cloud.py
