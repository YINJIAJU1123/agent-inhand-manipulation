# Build candidate only: publication requires scripts/rsl_rl/smoke_visual_camera.py.
ARG ISAACSIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54
FROM ${ISAACSIM_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG ISAACLAB_REV=37ddf626871758333d6ed89cf64ad702aef127d0
ENV OMNI_KIT_ALLOW_ROOT=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ISAACLAB_PATH=/workspace/IsaacLab \
    ISAACSIM_PATH=/isaac-sim
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake libglib2.0-0 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab && \
    cd /workspace/IsaacLab && git checkout "${ISAACLAB_REV}" && \
    ln -s /isaac-sim /workspace/IsaacLab/_isaac_sim
COPY docker/constraints.txt /opt/vlmrotation/constraints.txt
RUN /isaac-sim/python.sh -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
RUN /isaac-sim/python.sh -m pip install setuptools==80.9.0 wheel==0.45.1 && \
    /isaac-sim/python.sh -m pip install --no-build-isolation -c /opt/vlmrotation/constraints.txt \
    -e /workspace/IsaacLab/source/isaaclab \
    -e /workspace/IsaacLab/source/isaaclab_assets \
    -e /workspace/IsaacLab/source/isaaclab_tasks \
    -e /workspace/IsaacLab/source/isaaclab_rl \
    rsl-rl-lib==3.1.1 tensorboardX termcolor omegaconf psutil lark
WORKDIR /workspace/RevoLab
COPY . /workspace/RevoLab
ARG SOURCE_REV=unknown
LABEL org.opencontainers.image.source="https://github.com/YINJIAJU1123/agent-inhand-manipulation" \
      org.opencontainers.image.revision="${SOURCE_REV}"
RUN /isaac-sim/python.sh -m pip install --no-deps -e /workspace/RevoLab/source/BrainCo_DexHand && \
    /isaac-sim/python.sh -m pip freeze > /opt/vlmrotation/installed-packages.txt && \
    printf '%s\n' "${SOURCE_REV}" > /opt/vlmrotation/source-revision.txt && \
    git -C /workspace/IsaacLab rev-parse HEAD > /opt/vlmrotation/isaaclab-revision.txt
# GPU drivers are supplied by NVIDIA Container Toolkit at runtime, never copied
# from a developer machine. Code and Python dependencies require no host mounts.
ENTRYPOINT ["/isaac-sim/python.sh", "-u"]
CMD ["scripts/rsl_rl/smoke_visual_camera.py", "--headless", "--output", "/tmp/vlmrotation-smoke"]
