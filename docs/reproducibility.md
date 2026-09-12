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
