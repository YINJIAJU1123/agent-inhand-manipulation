# Visual student runbook

Run these commands from the repository root **inside the Isaac Lab container**
that contains the frozen teacher checkpoint. The host shell is not sufficient;
it needs the Isaac Sim camera runtime and PyTorch.

## Preconditions

```bash
export VISUAL_PYTHON=/opt/isaaclab-env/bin/python
# Use the actual mounted path inside the container if different.
export TEACHER_CHECKPOINT=/workspace/RevoLab/logs/rsl_rl/brainco_hand/2026-09-21_10-35-09_revo3_teacher_hold02_s21/model_1999.pt
sha256sum "$TEACHER_CHECKPOINT"
# Must equal:
# 193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707
```

The pipeline refuses to start when the checkpoint hash or action scale differs
from `configs/visual_student_freeze.json`.

## Full v1 run

```bash
export VISUAL_OUTPUT=/workspace/RevoLab/outputs/visual_student_v1
export VISUAL_NUM_ENVS=64
export VISUAL_EPISODES=387
export VISUAL_STRIDE=2
export VISUAL_TRAIN_SEEDS='0 1 2'
export VISUAL_MEMORY_MODES='plain evidence'

bash scripts/vision/run_visual_student_pipeline.sh all
```

The run creates one frozen-teacher rollout merge, one semantic feature cache,
six offline checkpoints/reports, and six live RGB-D reports. Use fresh output
directories for each code revision.

## Individual stages

```bash
bash scripts/vision/run_visual_student_pipeline.sh collect
bash scripts/vision/run_visual_student_pipeline.sh cache
bash scripts/vision/run_visual_student_pipeline.sh train
bash scripts/vision/run_visual_student_pipeline.sh eval
```

`train` reports action prediction only. `eval` is the paper-facing stage and
must use live camera frames through `--semantic-camera` or the versioned SigLIP
frontend. Cached feature means are intentionally rejected by the live evaluator
because they can leak the target face.

## Next search stage

The hidden-surface search task (`BrainCo-Direct-Revo3-SemanticSearch-Cube-v0`)
has a separate online PPO runner and protocol. It is not silently mixed into
this v1 visual reorientation result. Start it only after the v1 student path
has a reproducible live result and freeze a new search-specific manifest.
