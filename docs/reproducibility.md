# Reproducibility and publishing

## Code publication from the local computer

Keep `origin` pointing to upstream RevoLab and use a separate remote for this
research repository. A local GitHub SSH identity can push without installing
GitHub credentials on the GPU server:

```bash
git remote add agent-github git@github.com:YINJIAJU1123/agent-inhand-manipulation.git
git push agent-github main
```

If the remote already exists, use `git remote set-url agent-github ...` instead.
The working checkout and its Git history are the source of publication; a
source archive is only a transfer alternative, not a replacement for history.

## Environment status

The development LXD instance is named `VLMrotation` and was cloned from an
Ubuntu 22.04 environment. Its root filesystem occupies approximately 47 GB.
RevoLab, IsaacLab, RSL-RL, the Python interpreter and the Isaac Sim environment
are provided by host bind mounts. A root-filesystem export does **not** include
these dependencies and is not yet a self-contained reproducibility artifact.

Observed packages in the development Python environment:

| Package | Installed version |
| --- | --- |
| Python | 3.11 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab package | 0.54.3 |
| RSL-RL | 3.1.1 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| Gymnasium | 1.2.1 |
| NumPy | 1.26.0 |
| Hydra | 1.3.6 |
| transformers | 4.57.6 |

These are inventory values, not a tested installation lockfile. IsaacLab was
mounted as a source directory without its own Git metadata, so the exact
source revision still needs to be established. The RGB-D collector currently
fails during Isaac Sim RTX scene startup, before task execution. CUDA device
visibility alone does not establish that rendering works.

## Docker Hub publication

The local CLI is authenticated through GitHub browser authorization. The
`yinjiaju00/vlmrotation` repository was created as **private** on 2026-09-12.
Visibility can be changed separately when public release is intended.

An LXD export cannot be directly pushed as a Docker image. A Docker/OCI image
must first be built with its dependencies included, and its camera smoke test
must pass before it is tagged as validated. The intended repository is
`yinjiaju00/vlmrotation`; it has not yet received a validated image.

For a Docker account linked through GitHub, run `docker login` without a
username argument to start browser device authorization. A separate Docker
password is not required. Browser sign-in alone does not authenticate the CLI.

After an image exists and has passed validation, publish it with an immutable
version tag and record the returned digest:

```bash
docker login
docker tag vlmrotation:<validated-version> yinjiaju00/vlmrotation:<validated-version>
docker push yinjiaju00/vlmrotation:<validated-version>
```

Keep credentials, personal files, runtime logs, datasets and experiment
checkpoints outside the image build context. Preserve upstream license notices.
Record code revision, dependency pins, image digest and smoke-test results
together; a Dockerfile or LXD snapshot alone is not proof of reproducibility.

## Standalone Docker build candidate

`docker/Dockerfile` builds from the official full Isaac Sim 5.1.0 image, pinned
by digest. It includes released Isaac Lab v2.3.2 at revision
`37ddf626871758333d6ed89cf64ad702aef127d0` (package version 0.54.2), project
sources and assets. This is a newly pinned environment, not a byte-for-byte
copy of the source directory used in the old LXD instance (package 0.54.3).

```bash
docker build -f docker/Dockerfile --build-arg SOURCE_REV="$(git rev-parse HEAD)" \
  -t vlmrotation:local .
docker run --name vlmrotation-smoke --gpus all --shm-size=2g \
  -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all vlmrotation:local
docker cp vlmrotation-smoke:/tmp/vlmrotation-smoke ./outputs/camera-smoke
```

The Isaac Sim runtime is governed by NVIDIA's license included in the base
image; `ACCEPT_EULA=Y` records acceptance when running it. The host must provide
a compatible NVIDIA driver and NVIDIA Container Toolkit. No host Python or
IsaacLab installation is mounted. First startup may compile shaders for several
minutes. Internet access may still be needed for external IsaacLab assets.

The default command checks actual RGB-D output and repeated semantic goal
resets, then writes `rgb.png`, `depth.npy` and `report.json`. A sensor pass does
not prove policy performance, readable semantic markings or language grounding.
The current cube is unmarked, and its goal normal is world +Z, not the camera
view direction. These task changes and corrected-teacher evaluation must precede
semantic data collection.

The 2026-09-12 Docker sensor test produced 256×256 RGB-D, exercised all six
goal faces over 128 goal resets, and exited successfully on an RTX 5060 Laptop
GPU with driver 570.211.01. See `docs/validation/camera-20260912.json` and the
corresponding image. This is a local sensor test, not a 5090 training benchmark.
The complete Kit extension shutdown path segfaulted after successful rendering;
the smoke uses Isaac Sim's default fast shutdown on success and explicitly
returns a nonzero status for Python test failures before that shutdown.

`/opt/vlmrotation/installed-packages.txt` records resolved packages inside the
image; source revisions are stored beside it. Image contents exclude credentials,
experiment logs, datasets and checkpoints through an allowlisted build context.

`docker/Dockerfile.release` is a small overlay used when the dependency image
has already been built and tested locally. It copies only the current source,
reinstalls the editable project and updates its revision metadata; it does not
copy host Python or NVIDIA libraries. The complete from-scratch recipe remains
`docker/Dockerfile`.
