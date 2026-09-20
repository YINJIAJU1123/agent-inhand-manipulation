# Random-layout semantic search v0

This is an engineering pilot for the agreed visible/hidden search task, not a
completed CVPR benchmark. It introduces a new task ID and leaves the state
teacher task/checkpoints unchanged.

## Material Passport

- mode: run
- task: randomized marker layout, visible/hidden resets, ordinary history PPO,
  and an image-only scan/recognize/hold baseline
- source baseline: `238df55f36b3e4e86ad6b3399ef0b3efd1e1b3dc`
- status: implementation; per-run status is recorded in the output directory
- evidence: actual simulator PNGs, smoke assertions, training checkpoints,
  heartbeat, held-out episode JSONL and summary JSON
- limitations: six-color grammar, fixed image encoder, cube only, pilot thresholds

## Task and information boundary

Each cube has six colored square patches. A fresh permutation assigns colors to
local faces at every reset. All 720 permutations are deterministically shuffled
with seed 20260920 and split into 576 train / 72 validation / 72 test layouts.
Language is explicitly restricted to `Show the <color> marker.`; the parser
extracts color identity, not local face identity. This is not an open-vocabulary
language model or a compositional shape benchmark.

Visible resets select a face with depth-tested visibility at least 0.65,
projected area at least 20 pixels, and observable color evidence. Hidden resets
select a back-facing face with visibility at most 0.02. Mixed training assigns
these two conditions with equal probability before rejection sampling. A scene
without an eligible target is reset, with a hard limit and rejection accounting.

Actor inputs are 16x16 pooled RGB-D, six image-derived color area/centroid
statistics, normalized actuated joint positions, velocities, previous commands,
and a six-dimensional instruction color token. The 1111-dimensional input
feeds a GRU-256 and MLP [256,128], outputting 21 joint commands. The fixed encoder
is deliberately inspectable for the first pilot. A pretrained or learned visual
encoder is a later stronger baseline, not part of this implementation.

The asymmetric critic additionally sees object pose, velocities and the target
world normal. Neither those quantities, hidden face index/layout, nor previous
reward are actor inputs. All observation modalities are identical for direct
PPO and the scan baseline's low-level controller. No human task-action
demonstrations or teacher imitation are used.

## Success and reward

49 surface samples per patch are projected into the actual camera. Their
predicted optical-axis depth is compared with rendered
`distance_to_image_plane`, rejecting occluded, offscreen and behind-camera
samples. The 4 mm tolerance is a pilot setting and needs sensitivity analysis.
Patch projected area uses a local planar approximation. Inspect rendered
examples before interpreting metrics as reading-quality guarantees.

Display requires visibility >=0.65, area >=20 pixels, facing angle <=65 degrees,
linear speed <=0.04 m/s and angular speed <=0.5 rad/s continuously for 1 s.
Terminal success requires this predicate at the final step, without a drop.
The horizon is 20 s; control frequency is 30 Hz. Early display followed by
failure is `ever_display`, not terminal success. Drop distance is inherited
from the original environment (0.24 m from nominal grasp center).

Reward is shared between all PPO controls: camera-facing alignment, depth-tested
visibility, qualified display and dwell, first successful dwell, object-center
distance, action magnitude/slew and drop penalty. The critic/reward use
privileged training supervision. There is no private exploration reward for a
proposed memory method.

Terminal metrics are copied before automatic reset. Images and language are
rebuilt after reset. GRU state is reset on each episode boundary. Episodes are
sampled with per-vector-slot quotas during evaluation, preventing fast failures
from dominating counts.

## Scan baseline

The low-level controller is the visible-task recurrent PPO checkpoint. Every
two seconds, a cyclic rule chooses another currently visible non-requested
color as an intermediate instruction, using only RGB color masks. Three
consecutive detections of the requested color switch to holding it; a sustained
loss returns to scanning. Controller recurrence is cleared when its temporary
instruction changes. The same joint limits, frequency and image pipeline apply.

This implements a simple scan/recognize/hold system without giving it object
pose or hidden target location. It does **not** guarantee complete surface
coverage; limited coverage is an explicit potential failure. We must not call
it an optimal geometric scanning planner. Its visible-controller training cost
must be reported alongside the system comparison. The exact same visible
checkpoint should also be evaluated directly on hidden tasks to isolate the
scan rule's contribution.

## Commands

All outputs must use fresh directories; overwrite is rejected. The 5090 wrapper
selects the physical rendering GPU explicitly and limits Kit CPU threads.

```bash
bash infra/semantic_search/run_5090.sh --mode smoke --num_envs 4 --output /path/smoke
bash infra/semantic_search/run_5090.sh --mode train --initial visible --num_envs 64 --iterations 300 --seed 0 --output /path/visible_seed0
bash infra/semantic_search/run_5090.sh --mode train --initial mixed --num_envs 64 --iterations 300 --seed 0 --output /path/mixed_seed0
bash infra/semantic_search/run_5090.sh --mode eval --initial hidden --split val --num_envs 16 --episodes 32 --checkpoint /path/mixed_seed0/final.pt --output /path/mixed_eval
bash infra/semantic_search/run_5090.sh --mode scan --initial hidden --split val --num_envs 16 --episodes 32 --checkpoint /path/visible_seed0/final.pt --output /path/scan_eval
```

Use validation layouts for pilot tuning; reserve test layouts for the frozen
protocol. Repeat training with independent seeds before making method claims.
Short update smokes validate optimizer/checkpoint correctness only. A completed
run does not imply the task was learned.

## Monitoring and reproducibility

Every run saves arguments, code revision/dirty status, checkpoint SHA256,
environment YAML, runner configuration and human demonstration/pretraining
disclosure. `heartbeat.json` records process ID, elapsed time and rollout steps.
`status.json` is created only after successful completion; `failure.json` stores
Python failures. External `timeout` bounds the worker. Preserve outer worker
exit codes too, because native failures can happen before Python setup.

Do not modify an active run's source/configuration. Pin a separate code snapshot
for any subsequent fix or experiment. Use a new run directory after a failure.
