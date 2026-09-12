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

## VLMrotation environment

The new LXD instance is currently named `VLMrotation2` because the original
name was occupied during cloning.  It is based on `Yin-handover-2204` and uses
the `handover` directory storage pool.  The root filesystem currently occupies
about 47 GB at
`/var/lib/lxd/storage-pools/handover/containers/VLMrotation2`; the RevoLab,
IsaacLab and Python environments are separate bind mounts.  This explains why
an image export is large even though the code checkout is small.

To create a portable LXD image on the 5090 host:

```bash
lxc publish VLMrotation2 --alias vlmrotation-20260912
lxc image export vlmrotation-20260912 /path/to/vlmrotation-20260912
```

The exported image can then be placed in an online object store or a private
registry.  Docker Hub/GHCR/Hugging Face uploads require an account token; no
token is embedded in this repository.  Keep the large checkpoint and rollout
shards outside Git (for example in an object store or a dataset repository) and
record their URL and SHA256 in the experiment manifest.
