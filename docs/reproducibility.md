# Reproducibility and publishing

## Code

The working tree is ready to publish as `main`:

```bash
git remote add origin https://github.com/YINJIAJU1123/agent-inhand-manipulation.git
git push -u origin main
```

The same tree can be transferred without GitHub credentials with
`git archive --format=tar.gz HEAD`; this is useful on the 5090 host where the
LXD environment is running.

If the 5090 host cannot authenticate to GitHub, run the following on a local
computer that has GitHub CLI or an SSH key configured:

```bash
scp jiaju@10.3.100.20:/home/jiaju/agent-inhand-manipulation-src-latest.tar.gz .
mkdir agent-inhand-manipulation && tar -xzf agent-inhand-manipulation-src-latest.tar.gz -C agent-inhand-manipulation
cd agent-inhand-manipulation
git init && git add -A && git commit -m "initial VLM rotation research code"
git branch -M main
git remote add origin https://github.com/YINJIAJU1123/agent-inhand-manipulation.git
gh auth login                 # or configure a GitHub SSH key
git push -u origin main
```

## VLMrotation environment

The new LXD instance is named `VLMrotation`.  It is based on `Yin-handover-2204` and uses
the `handover` directory storage pool.  The root filesystem currently occupies
about 47 GB at
`/var/lib/lxd/storage-pools/handover/containers/VLMrotation`; the RevoLab,
IsaacLab and Python environments are separate bind mounts.  This explains why
an image export is large even though the code checkout is small.

To create a portable LXD image on the 5090 host:

```bash
lxc publish VLMrotation --alias vlmrotation-20260912
lxc image export vlmrotation-20260912 /path/to/vlmrotation-20260912
```

The exported image can then be placed in an online object store or a private
registry.  Docker Hub/GHCR/Hugging Face uploads require an account token; no
token is embedded in this repository.  Keep the large checkpoint and rollout
shards outside Git (for example in an object store or a dataset repository) and
record their URL and SHA256 in the experiment manifest.

For a Docker image that has already been loaded on the local computer, the
Docker Hub publication commands are:

```bash
docker login
docker tag vlmrotation:20260912 yinjiaju00/vlmrotation:20260912
docker push yinjiaju00/vlmrotation:20260912
```

Use a public repository for the free Docker Personal plan.  A private image
requires the single private repository included in that plan; a 47-GB Isaac
Sim image may still be impractical to upload, so publishing a Dockerfile plus
downloadable environment manifest is usually the better reproducibility
artifact.
