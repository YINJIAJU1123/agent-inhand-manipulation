# Four experimental questions and the first visual/semantic dependency diagnostic

## Paper questions

These are research questions, not conclusions established by the current system.

| Question | Required evidence |
|---|---|
| RQ1: Does learning improve waiting and recovery beyond strong fixed control? | Matched comparison with tuned slow/hold, untrained initialization and relevant residual-control baselines; completion, drops, time and variability on unseen states. |
| RQ2: Do visual evidence and semantic goals determine useful waiting/recovery behavior? | Same-state input interventions, retrained modality ablations, changed target/appearance combinations, and behavior changes that improve closed-loop completion. |
| RQ3: Does the method support a genuinely asynchronous semantic agent? | Real VLM/Astra requests with physics running; measured latency, stale/corrected responses, goal revision and observation feedback; end-to-end metrics. |
| RQ4: Does the learned mechanism transfer and generalize on hardware? | Repeated real trials with strong controls, multiple object/contact/appearance conditions, unseen combinations and complete failure reporting. |

RQ2 input interventions alone do not establish the other questions. The V3
comparison gave 65/92 successes versus fixed slow 62/92, with more total drops
(14 versus 12), substantial repeat variance and very small learned residuals.
Its outcome predictor has not beaten a constant prior. These remain open issues.

## Registered first RQ2 experiment

Runner: `scripts/recovery_controller_v3_rq2/run_inputs.py`.
Evaluator: `scripts/agentic/evaluate_recovery_inputs.py`.
Interventions: `scripts/agentic/recovery_input_interventions.py`.
Tests: `tests/test_recovery_input_interventions.py`.

Run directory: `outputs/recovery_controller_v3/rq2_inputs_20261018`.
The finite local RTX 3090 runner was launched on 2026-10-18 at 18:48 China time,
PID 1087529. It generates new caches, performs zero-delay controls, and evaluates
all interventions. It creates no recurring automation and does not train or
modify the frozen student/controller.

- Two newly registered evaluation seeds: **83001, 83002**, 12 slots each.
  They differ from the V3 development evaluation seeds and all training seeds.
- Two delays: 6 and 20 seconds. Two repetitions with reversed method order.
- Five conditions: fixed slow; full V3; frozen explicit controller RGB-D
  history; mismatched explicit controller goal; both interventions together.
- Positive delays: 2 seeds x 2 delays x 2 repeats x 5 methods x 12 = 480
  nominal trials. Four zero-delay groups add 48, totaling **528**.
- Sixteen planned videos: slots 0/5 of full V3 and slow, first repetition,
  both seeds and both positive delays.
- Frozen V3 checkpoint:
  `outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt`,
  SHA-256 `cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042`.
- Each branch replays identical cached prefix actions in a fresh process and
  requires exact physical/control-state agreement and matching eligibility.
  All failures are retained; a seed whose zero-delay controls all fail stops
  the pipeline rather than being replaced.
- Existing success metric: error <=0.16 rad for six consecutive post-response
  control steps, no previous drop, within a 300-step recovery period.
- Aggregate only completed five-method blocks. Report repeat spread and all
  trial denominators. At most 24 distinct initial request states are involved;
  528 nominal trials do not imply 528 independent initial conditions.

### What is actually intervened on?

Only the **explicit inputs to the lightweight V3 policy** change:

1. `frozen_visual`: replace V3's first 400 observation dimensions (four compact
   RGB-D frames) with the history recorded at request time. Hand state, proposed
   student action, command history and clocks continue updating.
2. `shifted_goal`: feed V3 the next canonical goal embedding modulo six. The
   frozen student and success scorer retain the true delivered goal.
3. `frozen_visual_shifted_goal`: apply both interventions.

The student always receives live correct RGB-D and the correct language
embedding. Its proposals and the evolving proprioception/history can convey
visual/goal information indirectly to V3. Thus an unchanged result would mean
that these **direct channels** have not shown additional value; it would not
show that the entire system works without vision or language.

These are test-time interventions and can cause distribution shift. A performance
loss is not, by itself, proof that the original representation is optimally
learned. Proper no-modality baselines must later be retrained under matched
budgets. Do not describe freezing a history as removing all vision.

### Fixed-state probes

At one-second intervals during learned control, evaluate all four input
conditions and fixed slow on exactly the same observation, proposal and current
target. Record actual action differences, interpolation rates, raw policy means
and residual norms. These forward probes do not advance or branch the simulator;
they isolate network sensitivity from render-induced trajectory divergence.
Per-arm `input_probes.json` and `probe_observations.pt` retain the evidence.
Closed-loop execution then measures whether these differences matter physically.

Three focused tests passed on CPU/CUDA where applicable: only selected input
fields change; freezing the request history has zero effect at the request;
probes leave model weights intact and react to changed goal embeddings; invalid
variants are rejected. Runtime checkpoint and cache audits are separate gates.

### How the evidence guides optimization

- If direct-input action differences are negligible, inspect whether the policy
  relies on student proposals and whether the training states contain situations
  requiring different waiting/recovery actions. Increase informative paired data
  before increasing network size or claiming semantic decision learning.
- If differences are substantial but closed-loop benefits are absent, focus on
  action/outcome supervision and recovery transitions, especially 20-second waits.
- Retraining ablations and matched goal changes at identical mechanical states
  are follow-up experiments. They must keep the requested goal/scoring coherent;
  a deliberately wrong controller embedding is a diagnostic, not a new task.
- The calibrated recoverability head and active query/stop policy remain future
  work. This run uses already delivered canonical goals and simulated responses.

### Initial runtime check (18:53 China time)

Both newly registered caches and all 48 nominal zero-delay trials completed.
Seed 83001 has 11 eligible states and succeeds in 3/11 in each repetition;
seed 83002 has nine eligible states and succeeds in 7/9 and 6/9. This exposes
baseline student generalization failures before adding waiting. Preserve these
cases in analysis; do not replace the seed or attribute every failure to V3.
All physical-prefix audits so far have zero error and identical eligibility.
The first 20-second learned intervention branch has begun without runtime errors.
Positive-delay comparisons and useful visual/semantic dependence remain pending.

## Practical real-to-sim-to-real preparation

This section is a proposed collaboration plan, not hardware work already done.
The colleague's hand model, camera and depth availability are awaiting confirmation.
No hardware commands or messages to colleagues were sent.

### Current interface that hardware must match

The current checkpoint uses **RGB-D plus proprioception**, not RGB alone:

- Renderer 512 x 512, resized to 256 x 256 for the frozen frontend; student
  consumes an eight-observation history, updated at 15 Hz; control is 30 Hz.
- Frontend includes RGB grid samples, fixed six-color marker statistics and
  depth grids. The language embeddings are separate; this is not a generic
  pretrained image encoder that can automatically handle arbitrary textures.
- Depth is metric, sanitized/clipped to [0,2] metres then divided by two.
  Hardware must match the simulator's depth convention and RGB registration;
  missing real depth must be modeled, not silently replaced by ideal depth.
- Proprioception contains normalized 21-joint positions, scaled joint velocities,
  fingertip positions/quaternions/linear-angular velocities, and previous actions.
  Fingertip quantities require matching forward kinematics and coordinate frames.
- Output is 21 normalized joint targets with action scale 1.0. Physical hardware
  coupling, independent actuator count, joint order, limits, servo gains and
  command latency need explicit mapping. A different hand requires adaptation.
- The base object references the DexCube USD with density=400 kg/m^3. Its actual
  collision geometry/inertia must be inspected or exported; the separate 7-cm
  goal-debug primitive must not be mistaken for the contact asset.
- Colored patches and symbols are visual-only, with no collision geometry.
  Real printed/attached markers should avoid unintentionally changing contact
  geometry, mass distribution or friction; measure the assembled object's mass.

Source references: `scripts/agentic/student_features.py`,
`train_recovery_controller_v3.py`, `symbol_scene.py`,
`source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/inhand_manipulation_env.py`,
and the `brainco_hand_*_env_cfg.py` configurations.

### Recommended first object and digital twin

Start with a rigid, matte object close to the current cube geometry, with
replaceable high-contrast semantic labels. First vary semantics and appearances
on matched geometry; then add objects with different contact geometry. This
provides a tractable transfer baseline while preserving a path to generalization.

For each physical object, prepare:

| Asset | What to collect or build |
|---|---|
| Geometry | CAD or calibrated multi-view reconstruction; metric scale, edge rounding, seams, object coordinate frame and semantic surface normals. |
| Appearance | Texture/UV and material, all faces and occluded surfaces, several viewpoints and actual lab lighting. Keep a scale reference and exposure metadata. |
| Contact model | Simplified collision mesh/convex parts, measured mass, estimated centre of mass/inertia, friction range and fingertip material. |
| Semantics | Label images, their physical locations/orientations, face-to-instruction mapping, and held-out layouts/prompts. |
| Sensor calibration | RGB/depth intrinsics and registration, camera-to-hand extrinsics, crop/resize convention, timestamped frame latency and dropout behavior. |
| Hand characterization | Joint/actuator mapping, limits, measured response traces, coupling, target tracking and latency; contact behavior under the actual fingertip surfaces. |

Use a textured mesh for the first simple object. Consider 3D Gaussian Splatting
when real appearance is hard to reproduce; retain a separate physical collision
model. Photorealistic rendering does not identify friction or contact dynamics.

[ViserDex](https://rffr.leggedrobotics.com/works/viserdex/) demonstrates using
3DGS augmentation for monocular RGB pose estimation and then feeding estimated
poses to a student. Its visual-data construction is useful precedent; directly
feeding pose to our policy would change our present input contract and should
be evaluated as a separate baseline or newly trained architecture.
[RialTo](https://real-to-sim-to-real.github.io/RialTo/) provides a precedent for
constructing digital twins from real scenes and improving policies in simulation;
its demonstrated tasks do not guarantee transfer for this hand.

### Calibration and adaptation sequence

1. Measure the real hand, object and camera before freezing a deployment setup.
   Capture timestamped stationary and moving observations with partial finger
   occlusion, not only isolated turntable object photographs.
2. Recreate the sensor framing and object appearance in simulation. Verify
   channel order, undistortion, metric depth, white balance and exposure. Current
   fixed-color features make these details especially consequential.
3. Calibrate contact and servo behavior from short real motion/hold traces.
   Randomize mass, friction, gains, latency and visual nuisance factors over
   measured plausible ranges. Keep semantic label identities coherent: random
   color/layout changes must update labels when they change the task meaning.
4. Use real images to assess frontend error and, if needed, fine-tune/retrain the
   visual student under the versioned observation contract. A pure-RGB hardware
   setup needs a separately trained/distilled RGB-only student; setting depth to
   zero at deployment is not a validated substitute. A learned image encoder is
   an option after measuring the current frontend's domain gap.
5. Deploy the frozen-student baseline first, then compare the waiting/recovery
   controller under the same hardware and response latencies. Keep tuning data
   separate from final test trials and retain all failures.
6. Measure object orientation independently for evaluation. An external tracker
   or a pose-estimation system can help; quantify its error and occlusion failure.
   Do not feed evaluation-only measurements to the policy. If visual tags appear
   in the policy camera, disclose them as part of the task/observation setup.

[FoundationPose](https://nvlabs.github.io/FoundationPose/) is a possible evaluation
or pose-baseline component with CAD/reference data. Its official implementation
uses RGB-D for pose estimation; it should not be advertised as a drop-in RGB-only
solution. We have not integrated it or tested it under this hand's occlusion.

The most useful first package from the hardware colleague is the hand/SDK model,
camera model and synchronized RGB(-D), joint/command logs, one object's CAD or
multi-view photos with measured scale/mass, and semantic label layout. These
inputs let us build a measured digital twin and estimate the actual transfer gap.
