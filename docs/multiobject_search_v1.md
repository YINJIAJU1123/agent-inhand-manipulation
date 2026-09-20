# Multi-object search and shared control preparation

## Material Passport

- Mode: run; procedural multi-object framework and bounded training deployment.
- Research variables: search/history strategy and object generalization.
- Visual training/distillation choices are shared settings, not an ablation axis.
- Evidence: catalog unit checks, actual RGB-D smoke artifacts, A100 state
  evaluation and PPO update smoke; long-run results are pending.

## Object protocol

`algo/agentic/object_catalog.py` defines `procedural_surfaces_v1`: six training,
three validation and three held-out test instances, spanning boxes, hexagonal
prisms and octagonal prisms. Validation/test instances have different dimensions
within these same families. This is instance/size generalization, not unseen
shape-category generalization or a real-world object benchmark.

Each object has six explicit planar marker frames. Hexagonal prisms use their
six side faces; boxes and octagonal prisms use four opposing sides plus two end
faces. All markers are 20 x 20 mm, 1 mm thick, with a 0.2 mm visual gap and no
collision. Object mass derives from true mesh volume at density 400 kg/m^3.
The physical mesh is convex and used directly with convex-hull collision.
Markers share the original six-color instruction grammar and independently
randomized layout partitions. Hand, camera, visual frontend, actions, rewards
and display thresholds are shared across methods.

Isaac's native MultiAssetSpawner assigns assets cyclically to slots, with
physics replication disabled. The actual PhysX view order is mapped back to
USD object identity and checked for balance. Object geometry stays fixed per
slot; pose, joints, target and marker layout reset per episode. Training env
count must be divisible by six; validation env count by three. Evaluation uses
equal per-slot quotas and exports object identity for reporting, never as an
actor input. A real-object mesh adapter can supply this same mesh/surface
description, but no automatic real-asset import or grasp generation is claimed.

## Fixed visual setting

The executable visual baseline retains the existing pooled RGB-D/color
features, GRU-256, MLP and PPO. No visual backbone or distillation comparison is
introduced. A100 state control preparation is separately labeled and is NOT
already connected to this visual policy as a teacher or weight initializer.
Any later agreed shared distillation stage must be connected and verified
explicitly before claiming the visual learner uses state pretraining.

The legacy single-cube path remains selectable by omitting `--object-split`;
its 28 mm markers and geometry are preserved. New runs use `--object-split
train|val|test`, with optional `--objects` restricted to that partition. Training
on held-out objects/layouts is rejected. Per-object training counts and held-out
evaluation are saved separately. Equal RNG seeds still do not implement paired
initial-state replay; this inherited pilot limitation remains.

## State control preparation

The state task retains the 158-dimensional full-state observation and adapts
target quaternions to each object's actual surface frame. It is a privileged
orientation/control task, not a language-search result. Shared initialization
is the existing `model_998.pt` (SHA256
`42905dcdae5bb99fb2ed55e9dbfe103579a20c90ff8cf00e0c6342cea1874bf5`).

The bounded A100 pipeline defaults to 768 envs, 16 steps/update, 3,000 additional
updates (36,864,000 transitions per run), seeds 0/1/2. These are independent
adaptation RNG seeds sharing one pretrained checkpoint, not independent
from-scratch pretraining seeds. It keeps the prior 0.5 s hold shaping, random
goal yaw and 21-action interface. Training has a 12 h timeout and must produce
the expected final iteration before evaluation starts.

Automatic evaluation covers six surfaces x three evaluation seeds x 96 episodes
for each of training and validation objects: 3,456 episodes per checkpoint.
Test objects remain reserved. Each scene gets a fresh process; results are
validated and aggregated only after the complete grid exists.

## Visual queue

`multiobject_visual_pipeline.sh` uses the same GPU lock as the existing pilot,
waits up to 24 h, and checks free memory before starting. It runs a six-object
render/reset smoke and a 96-env PPO scaling smoke first. Only successful stages
advance. Defaults: 3,000 updates x 32 steps x 96 envs = 9,216,000 transitions for
each of visible-controller and mixed-target training, from separate random
initializations. Direct PPO and scan reuse the visible controller for their
corresponding comparisons. Evaluation covers initially visible/hidden targets
on seen and unseen validation objects, with validation marker layouts.

This is an initial multi-object baseline budget, not evidence that learning has
converged. `SEARCH_VALIDATE_ONLY=1` stops after smokes. Every run must use a fresh
output directory and pinned clean source. No existing worker is terminated.
