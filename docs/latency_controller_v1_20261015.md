# Latency controller v1 — continuous-physics verification waits

## Scope

This experiment trains a small PPO supervisor around the **frozen RGB-D student**.
It simulates waiting for verification of the **last already-delivered goal**.
The response reaffirms that goal. No live Astra calls are made, and this version
has no new-goal handling, learned response-validity test, or learned hold primitive.
It is a first controller experiment, not evidence of a complete asynchronous agent.

Physics runs at 120 Hz, joint commands at 30 Hz, fresh camera frames at 15 Hz,
and mode decisions at 2 Hz. Simulated delay is measured in physical simulation
steps; no Python sleep substitutes for physical evolution.

## Actions and information boundary

The frozen student's outputs are normalized **joint position targets**.

- Continue: use the student's target.
- Slow: interpolate 15% from current commanded target toward the student's target.
- Hold: latch the commanded joint target on entry and maintain it until leaving hold.

The learned actor receives compact RGB-D statistics/grids and their temporal
change, measured robot proprioception, the student proposal, current/latched
joint targets, previous mode, and elapsed request age. It does not receive the
true remaining delay, object pose/velocity, scorer error, or a hidden target label.
Goal conditioning is inherited from the frozen student with its already-delivered
canonical goal embedding. A separate training-only critic additionally receives
orientation error, object distance, and object linear/angular velocity.

The actor has two hidden layers (128, 64). PPO uses 5 update epochs, clipped
ratio 0.2, decision-step gamma 0.995, GAE lambda 0.95, and entropy weight 0.02.
The objective includes survival, orientation progress, a small target-change
penalty, drops, and the outcome of resuming the frozen student after the wait.
A prescribed hold target is not described as a learned stabilizing policy.

## Protocol

Training: 24 environments, 12 batches, seeds 51000–51011, delays cycling through
2/8/16 seconds. All six delivered goals are balanced. Every episode first runs
the student for 30/60/90 steps to reach an intermediate manipulation state.
After waiting, the frozen student receives up to 300 control steps to recover.
The initial smoke test used 12 environments and a shorter recovery window; it
is excluded from formal results and not used to select a checkpoint.

Evaluation: independent seeds 62001/62002, 12 environments per seed, unseen
4/12-second waits. Compare continue, slow, hold, and the final PPO controller.
Each mode starts from a shared cached request state: joint positions/velocities,
object root state, command targets, previous actions, the RGB-D observation and
student history. Every branch resets and reconstructs this state, with a numeric
restoration check. Hidden contact-solver warm-start memory is not serialized.
Request-state proprioception is audited across methods after reconstruction.
Training runs locally on RTX 3090; evaluation runs on an otherwise idle RTX 5090.
All evaluated methods use RTX 5090 on the same host (separate free GPUs).

Success requires orientation error <= 0.16 radians for six consecutive control
steps **after** the response, with no preceding drop. Successful slots stop
being scored. Prefix failures, wait drops, and recovery drops are reported
separately. Primary rates condition on reaching the request alive; unconditional
counts remain available in the trial files. This differs from prior Astra reach
benchmarks and must not be directly compared with their percentages.

## Files

- `scripts/agentic/latency_controller_v1.py`: sensor-only actor, training critic,
  position-target primitives, PPO update.
- `scripts/agentic/train_latency_controller_v1.py`: continuous-physics rollout,
  terminal recovery credit, training, matched evaluation, provenance.
- `tests/test_latency_controller_v1.py`: target interpolation/hold latching and
  actor-gradient checks without Isaac Sim.
- `outputs/latency_controller_v1/train_v2_20261015/`: formal first training run;
  checkpoint, manifest, source snapshot, training history and individual trials.

## Training outcome and evaluation audit

The fixed 12-iteration training run completed in 1,075 seconds: 288 nominal
training episodes, 265 surviving request starts and 4,048 PPO decisions.
The actor has 59,523 parameters. The final checkpoint is
`outputs/latency_controller_v1/train_v2_20261015/controller_v1_final.pt`, SHA256
`a385b770d84fefb91a6f3024b045aabcc3b329204d17c71f3dffbea99f1d26f0`.
It was selected by the predeclared final iteration, not by test performance.

The original `baseline_20261015` results are **exploratory, not a strict paired
comparison**: despite reset differences <= 1.2e-7, request poses/proprioception
had diverged substantially during the student prefix. Those figures must not
be used as evidence of a learned-controller gain. The corrected evaluation uses
`request_cache`, generated once, and all modes reconstruct the same request states.
The cache retains 23/24 request starts (the excluded prefix failure is fixed for
all methods). There are 46 eligible cases per mode across the two delay values.

Corrected baseline and learned evaluations completed in
`baseline_cached_20261015` and `learned_cached_20261015`: 192 nominal trials,
184 eligible scored trials (46 per mode). All 12 cross-mode request-state audits
have **zero** maximum difference in the saved object/proprioception state, and
all eligibility masks match.

| Policy during wait | Recovery successes | Success rate | Wait drops | Recovery drops |
|---|---:|---:|---:|---:|
| Continue student | 32/46 | 69.6% | 6 | 1 |
| Always slow | 31/46 | 67.4% | 1 | 5 |
| Always hold target | 31/46 | 67.4% | 0 | 7 |
| Learned supervisor | **34/46** | **73.9%** | **1** | **5** |

These are conditional on reaching the frozen request state alive. Including the
one prefix failure repeated at two delay values gives 34/48 = 70.8% for learned.
The 46 cases are 23 request states crossed with two delays, not 46 independent
initial states. No statistically established improvement is claimed.

The learned actor spends 4.08% of waiting steps continuing, 95.78% slowing,
and 0.14% holding. Ten cases have more than an initial entry switch. Thus it
is mostly a slowdown policy and has not learned a mature three-mode strategy.
Its gain over fixed slowing is only three additional successes, with the same
wait/recovery drop counts. It reduces waiting drops compared with continued
student execution, but five failures occur after the hard handoff back to the
student. This motivates learning/smoothing the handoff and improving the hold
primitive before claiming a robust agent-waiting controller.

By delay, learned succeeds on 18/23 four-second cases and 16/23 twelve-second
cases. Continue scores 17/23 and 15/23; slow 18/23 and 13/23; hold 15/23 and 16/23.

The final checkpoint, full per-case reports, decision probabilities, state
caches/checksums and eight validated videos are saved locally. The checkpoint
is also backed up on `brainco@10.3.100.21` at
`/home/brainco/agent_inhand_astra_20261015/outputs/latency_controller_v1/checkpoints/controller_v1_final.pt`.
No live API calls were used in this experiment.

- Consolidated results: `outputs/latency_controller_v1/paired_summary_20261015.json`.
- Successful 12-second wait and recovery:
  `outputs/latency_controller_v1/learned_cached_20261015/62001_12_learned_case0.mp4`.
- Survives the wait, drops during recovery:
  `outputs/latency_controller_v1/learned_cached_20261015/62002_12_learned_case5.mp4`.

 Early interrupted launch folders are not completed
experiments. Cache-generation trial summaries are placeholders, not task results.

### Limits of this first version

- This is a simulated delay replay, not live asynchronous Astra integration or
  a demonstrated real-time end-to-end controller.
- Training uses three coupled prefix/delay groups (30/60/90 steps with 2/8/16 s).
  A later training run should independently randomize these dimensions.
- Only two held-out reset seeds are used; this is an engineering pilot.
- The actor learns mode selection; stable holding itself is a prescribed target.
- Changed goals, obsolete responses, perturbations, unseen objects and much
  longer waits are not yet covered.

### Reproduction

Use the existing Isaac Lab Python environment, accept the already-authorized
Isaac Sim EULA, and enable headless camera rendering. Main commands:

```bash
# Point this at the installed Isaac Lab Python environment.
export ISAAC_PYTHON=/home/yin/Grasping/Grasppose/env_isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMP_NUM_THREADS=4
export PYTHONPATH=outputs/vlm_dependencies:outputs/learning_dependencies:outputs/video_dependencies
"$ISAAC_PYTHON" scripts/agentic/train_latency_controller_v1.py --headless --device cuda:0 \
  --num-envs 24 --iterations 12 --train-delays 2 8 16 --skip-eval --output NEW_TRAIN_DIR
"$ISAAC_PYTHON" scripts/agentic/train_latency_controller_v1.py --headless --device cuda:0 \
  --num-envs 12 --eval-only --eval-seeds 62001 62002 --eval-delays 4 \
  --eval-modes continue --request-cache NEW_CACHE_DIR --cache-only --output NEW_CACHE_RUN
"$ISAAC_PYTHON" scripts/agentic/train_latency_controller_v1.py --headless --device cuda:0 \
  --num-envs 12 --eval-only --eval-seeds 62001 62002 --eval-delays 4 12 \
  --eval-modes continue slow hold --request-cache NEW_CACHE_DIR --output NEW_BASELINE_DIR
"$ISAAC_PYTHON" scripts/agentic/train_latency_controller_v1.py --headless --device cuda:0 \
  --num-envs 12 --eval-only --eval-seeds 62001 62002 --eval-delays 4 12 \
  --eval-modes learned --checkpoint FINAL_CHECKPOINT --request-cache NEW_CACHE_DIR \
  --record-slots 0 5 --output NEW_LEARNED_DIR
```

For a masked nonzero physical GPU, use the same physical index in
`CUDA_VISIBLE_DEVICES` and `--kit_args=--/renderer/activeGpu=INDEX`, while the
logical physics device remains `cuda:0`. Use fresh output directories.


## Expanded paired diagnostic prepared on 2026-10-17

The new evaluator is `scripts/latency_counterfactual/evaluate.py`, with a finite
runner in `scripts/latency_counterfactual/run.py` and a seed-cluster analysis in
`scripts/latency_counterfactual/analyze.py`. It leaves the running local v7
experiment and all of its frozen sources unchanged.

The predeclared formal grid uses 12 independent reset seeds (81001–81012),
12 environments per seed, six fixed physical delays (0, 0.5, 2, 6, 12, 20 s),
and four waiting policies (continue, slow, held position target, learned v1).
This is 144 nominal request states and 3,456 nominal rollouts; repeated delays
are not independent request states. Prefix failures are fixed by the shared
cache and reported separately. The student runs a balanced 1/2/3-second prefix
before caching. Every branch reconstructs the same mechanical state, command
targets, student histories and visual features. A zero-delay control must agree
across all four modes. Hidden contact-solver warm starts remain a limitation.

Visual processing matches local v7: 512-pixel RGB-D rendering resized to 256,
15 Hz camera updates, 30 Hz commands and 2 Hz mode selection. The response
reaffirms an already delivered goal. This isolates waiting behavior without
live API, grounding or reply-content variation; unknown-goal cold starts and
changed goals are outside this diagnostic. Success retains the controller
pilot criterion of six consecutive post-response steps within 0.16 radians,
with up to 300 recovery steps. Wait survival, recovery drops, success given
wait survival, recovery time, mode selections and handoff states are recorded.

An excluded 48-rollout smoke (seed 80999, 6 environments, delays 0/6 s, four
modes, 90-step recovery) precedes the formal run. Videos are recorded for the
smoke and for slots 0/5 of seed 81001 at delays 6/20 s. The finite runner waits
up to 12 hours for a 5090 with less than 512 MiB usage and less than 5% load
for three consecutive checks; it starts only one simulator and uses no
recurring automation. Complete seeds are aggregated with paired comparisons
and confidence intervals clustered by reset seed.

Deployment status at preparation: the 8 GPUs on 10.3.100.20 and those visible
through port 22031 were busy. No connection to 10.3.100.21 was made because the
user had previously excluded it. Upload of the 76.75 MiB package to 20 was
rejected by automatic approval review pending explicit authorization of that
payload and destination. No remote experiment has started at this point.
The local package manifest is
`outputs/latency_counterfactual/v2_20261017/deployment_manifest.json`.

### Zero-delay diagnostic on 2026-10-17

After the local 720-episode v7 run finished, a local engineering smoke was
started while remote payload authorization remained pending. It failed the
zero-delay negative control: all four modes applied the same student at zero
wait, but their recovery outcomes differed. This smoke is excluded and the
3,456-rollout formal experiment has not started.

A short trace audit found equal measured initial mechanics and equal initial
actions, followed by different mechanics after the first physics step when
reusing a simulation process. Fresh processes eliminated that first-step
difference, but fresh images later differed and caused trajectory divergence.
Additional renderer diagnostics are recorded under
`outputs/latency_counterfactual/`. The previous deployment archive is marked
stale and must not be deployed without regenerating and validating it.

This also limits interpretation of the older cached 46-case pilot: it audited
measured request states, but did not establish equal subsequent hidden solver
or rendering state. Preserve its raw results as exploratory evidence; do not
use them as a strict counterfactual controller-gain claim. The separate v7
720-episode live-agent run does not use this mid-trajectory restore helper.

### Corrected v3 protocol and local launch (2026-10-17)

The earlier v2 plan and failed engineering traces above are superseded by this
v3 implementation. Student, language adapter, learned controller and action
scale remain frozen; these changes repair the evaluation harness, not the
trained controller.

- Every cache and every policy/delay arm runs in a fresh Isaac Sim process.
- The cache now includes the exact commanded prefix actions. Each arm starts
  from the same seeded initial environment and replays those actions. It then
  checks joint positions/velocities, object root state, command targets and
  action buffers against the cache with **zero tolerance**. It does not write
  an intermediate physical state back into the simulation. This preserves
  contact history generated by the prefix instead of transplanting measured
  state into an unrelated solver history.
- Each arm receives the identical cached student observation history at the
  request time. Subsequent RGB-D frames come from that arm's evolving scene.
  Images are never replayed after the policies diverge.
- RTX sampling diagnostics identified residual RGB differences even with
  unchanged depth and mechanics. The validated controlled renderer uses the
  installed Isaac Lab performance preset, AA off, fixed environment light,
  no direct lighting/shadows, no tone-map dithering, no auto-exposure or film
  grain, and explicit simulation-based render frame/time counters. This
  produces a flatter appearance than live v7. It is a separate controlled
  benchmark; its rates must **not** be pooled with live-v7 results or taken as
  evidence of unchanged performance under live-v7 lighting.
- Two independent 90-step zero-delay trajectories under this renderer matched
  exactly (`ambient_audit_20261017`). Prefix replay separately reproduced all
  eight audited mechanical/control fields with max error 0.0. The runner adds
  a stronger per-seed gate: four independently executed zero-delay arms must
  have identical complete trajectory hashes and outcomes before any delayed
  arms for that seed are accepted. A mismatch halts the run; it is never
  silently discarded or retried to select a favorable result.
- The nominal grid remains 12 seeds × 12 slots × 6 delays × 4 methods = 3,456
  rollouts. Only 144 nominal request states are independent of delay/method;
  prefix failures and eligible denominators are explicit. The 48-rollout
  engineering smoke is excluded. Per-seed clustered intervals remain primary
  for uncertainty; the fixed-delay result is not a live Astra/VLM benchmark.

A finite local runner was launched at
`outputs/latency_counterfactual/formal_v3_local_20261017` (PID 785990; empty queue-only preflight archived separately).
It first waits for the local RTX 3090 to be free of compute jobs and then runs
its smoke gate before starting the formal grid. `progress.json`, per-arm logs,
`negative_control.json`, `prefix_replay_audit.json`, and `summary.json` provide
status and audit evidence. `summary.json` only includes fully completed seeds.
The source, configuration, asset and weight hashes are recorded in
`protocol.json`. This is a finite experiment process, not a scheduled task.
No package was uploaded to remote hosts and no process was launched on host 21.

Launch validation: all 8 smoke arms / 48 nominal rollouts completed; all four
zero-delay full trajectory hashes match, and all eight prefix replay audits
have max error 0.0. The runner advanced to formal seed 81001. See
`formal_v3_local_20261017/launch_validation.json` for the timestamped evidence.
No new formal success-rate claim is made at launch.


### v3 validity failure and rendering diagnosis (2026-10-17)

The v3 local run was stopped at 2,844 / 3,456 completed nominal rollouts
(237 / 288 arms). Nine seeds were complete: 108 nominal request states,
88 eligible states and 20 prefix failures. All four zero-delay trajectories
matched exactly, but each arm achieved **0 / 88 task successes** and had
65 recovery drops. Equal failing trajectories do not validate task performance.
The launch gate checked repeatability but omitted a baseline-performance gate;
this was an evaluation-design error. No replacement formal run has started.

Evidence is retained in `formal_v3_local_20261017/validity_review.json`; the
unchanged numerical summary describes this compromised rendering domain only.
For example, 20-second waiting survival was 17/88 (continue), 61/88 (slow),
73/88 (hold), and 59/88 (learned). These are not task success rates, and do not
establish a learned-controller benefit in the original visual domain.

Two bounded diagnostics used seed 81001, 12 environments, identical frozen
weights, goals, prefix length, 300-step recovery and a six-step success rule.
Both ran continuously from reset with **no intermediate state overwrite**:

| Renderer | Prefix survivors | Post-prefix reach | Six-step success | Recovery drops |
|---|---:|---:|---:|---:|
| Original/default | 12/12 | 11/12 | 11/12 | 1/12 |
| Formal ambient profile | 12/12 | 1/12 | 0/12 | 10/12 |

The render-profile change alone is sufficient to reproduce the collapse on
this seed. The entire render bundle was toggled; this does not isolate a single
lighting switch, and one seed is not a new overall success estimate. Matching
fields and limitations are recorded in
`outputs/latency_counterfactual/validity_diagnostic_20261017/comparison.json`.
The diagnostic source is `scripts/latency_counterfactual/evaluate_validity.py`.

The next valid evaluation must retain original visual appearance and first
check baseline performance on separate preflight seeds. It must quantify
renderer repeat variance, rather than change appearance solely to obtain
bit-identical trajectories. Exact mechanical prefix replay can still be used;
single-step reach and sustained success must be reported separately.
The previous 720-episode live-v7 results used the original renderer and remain
a separate experiment. No training, policy-weight update or live API call
was performed during these diagnostics.


### Original-render two-seed restart (2026-10-17)

At the user's request, restore the original renderer and cap the diagnostic
at seeds 81001 and 81002. No extra smoke seed is added. The finite local run
is `outputs/latency_counterfactual/original_v4_two_seed_20261017` (initial
runner PID 906178), launched with `scripts/latency_counterfactual/run_original.py`.
It uses 12 environments per seed, delays 0/0.5/2/6/12/20 s, and
continue/slow/hold/learned modes: 576 nominal rollouts but only 24 nominal
request states. The frozen student, adapter and controller are unchanged.
No live Astra/VLM request is made; this isolates known-goal waiting behavior.

No lighting, renderer preset, anti-aliasing, dithering or render-clock override
is applied. Each arm still starts a fresh simulator and replays the exact
cached prefix commands. Eight mechanical/control fields must match exactly
at the request. Subsequent images are rendered from the branch's own state.
Full trajectory hashes are retained, but image-noise differences are measured
as nuisance variation instead of being removed by changing appearance.

Before any delayed arm starts, both seeds must pass a continue zero-delay
check, then all four zero-delay repeats. Predeclared engineering checks:
at least 6 eligible states per seed; at least 50% six-step success in each
zero-delay arm; at most 25 percentage points success-rate spread among the
four identical-control repeats. Failure halts the whole batch with evidence
retained; no silent retries or seed exclusions are allowed. These thresholds
are regression checks, not significance tests.

Primary success remains six consecutive post-response control steps with
orientation error <=0.16 rad and no prior drop, within 300 recovery steps.
Single-step reach is now also recorded as a secondary metric. Both seeds
were selected from the old plan and seed 81001 was used in the renderer
diagnostic; these are not held-out validation seeds. Two-seed reports omit
bootstrap intervals and make no statistical-significance or generalization
claim. Do not pool them with the retired ambient-only run or live-v7 results.

Videos are recorded for seed 81001 slots 0/5 at delays 0/6/20 s. Protocol and
source/asset/weight hashes are frozen in `protocol.json`; the corresponding
files are archived locally in `frozen_sources_and_weights.tar.gz`. Monitor
`progress.json`, `baseline_checks.json`, `launch_validation.json`, per-seed
`negative_control.json`, and `summary.json`. This is a finite process, not a
scheduled task, and host 21 is not used.


Two-seed baseline outcome and explicit diagnostic amendment:
seed 81001 had 12 eligible states, 10 single-step reaches and 6 sustained
successes; seed 81002 had 10 eligible states (2 prefix failures), 6 reaches
and 4 sustained successes. The original runner stopped on the second seed's
40% sustained rate, below the predeclared 50% engineering threshold.
The stop and threshold failure remain recorded; they are not relabelled as
passing validation.

To fulfill the user's requested small two-seed comparison,
`continue_original_diagnostic.py` completes the same preplanned 576-rollout
grid as an explicitly exploratory diagnostic. The 24 existing trials are
retained, no seed is changed, no trial is rerun or excluded, and evaluation
code, render configuration, weights, actions and success criteria remain
frozen. `diagnostic_amendment.json` records the change after observing the
baselines and the original protocol hash. The initial 50% performance and
25-point repeat-spread thresholds are still reported as engineering warnings.
Mechanical replay, source integrity, render equality, completeness and
runtime checks remain hard stops; all-zero repeated baseline performance
or fewer than six eligible requests also stops the diagnostic.

Use `analyze_original_diagnostic.py` for this amended run. Completion is
labelled `completed_diagnostic`, with per-seed engineering failures and no
bootstrap significance claim. The initial runner log, source archive and
progress snapshot before continuation are retained.


### Two-seed original-render completion review (2026-10-18)

All 48 arms / 576 nominal rollouts finished at 2026-10-17 23:29:22 China
time. Both seeds are retained. There are 24 nominal request states, of
which 22 survive the common prefix. All 48 physical prefix-replay audits
have max error 0.0; frozen source/asset/weight hashes still match. No
per-arm runtime failure was recorded. The runner has exited, and 24
nonempty videos were saved for seed 81001 slots 0/5 at delays 0/6/20 s.

For controller comparison, positive delays (0.5/2/6/12/20 s) are reported
separately from zero-delay identical-control repetitions. Each strategy
has 120 nominal cases, 110 eligible cases and 10 repeated prefix failures:

| Strategy | Sustained success | Wait drops | Recovery drops |
|---|---:|---:|---:|
| Continue | 70/110 (63.6%) | 10 | 8 |
| Slow | 71/110 (64.5%) | 4 | 13 |
| Hold | 68/110 (61.8%) | 0 | 26 |
| Learned | 71/110 (64.5%) | 2 | 18 |

These 110 cases repeat 22 physical request states across five delays; they
are not 110 independent initial states. The learned policy chose slow
on 24,564 / 26,034 active waiting control steps (94.35%), continue on
1,470 (5.65%), and hold on zero steps. It has not demonstrated a task
success advantage over fixed slow. Reduced observed waiting drops are
a preliminary trend, not a validated causal gain.

Zero-delay success counts, although the control law is identical, were
10/22, 17/22, 13/22 and 18/22 across the four labelled repeats. Per-seed
spreads were 41.67 and 40 percentage points, failing the retained repeat
gate. This large nuisance variation prevents attributing the roughly
one-percentage-point positive-delay differences to the waiting policy.
The complete run remains `completed_diagnostic`, with engineering warnings
retained. No new training or live Astra/VLM inference occurred.

The useful next diagnostic is original-render repeat variability and the
transition from waiting back to student execution, using the same two
seeds. The 26 post-wait drops under fixed hold make recovery behavior a
concrete target for inspection; they do not prove a specific handoff bug.
Machine-readable interpretation and positive-delay counts are saved in
`original_v4_two_seed_20261017/completion_review.json`.


### Additional matched-state waiting repetitions (2026-10-18)

The user requested more testing of the low waiting-drop observation. A finite
local replication was launched at
`outputs/latency_counterfactual/repeated_waits_v5_20261018` (PID 961416), using
`scripts/latency_counterfactual/run_repeated_waits.py`. It retains the same
two seeds 81001/81002 and exact original request caches, with two additional
repetitions at delays 0/6/20 s and all four strategies: 576 nominal rollouts.
There are still only 24 nominal request states, 22 eligible; these repetitions
measure variability on those states and do not establish generalization.

Original rendering, evaluator code, scene assets, all weights, action scale
and six-step success are unchanged. Cache copies and frozen files are hashed.
All arm processes must reproduce the eight audited physical/control fields
exactly and use the original render settings. New outputs are separate from
the completed v4 diagnostic; prior poor outcomes and engineering warnings
remain preserved. No training or live Astra/VLM call is involved.

Execution order is fixed in advance by schedule seed 181026. Zero-delay
groups run first. Within each phase groups are shuffled, and repeat 2
reverses repeat 1's mode order for every seed/delay to balance each mode's
serial position. All repeated zero-delay results are reported as identical-
control repetitions; outcome variability is not labelled a policy effect.
All-zero success across the zero-delay repetitions of either seed, physical
or file-integrity mismatches, wrong rendering, incomplete trials or runtime
errors stop the finite run. No outcome-based retries or exclusions occur.

Primary outputs separate waiting drops, recovery drops, and final sustained
success, with recovery-drop conditional rates also reported. Summaries only
include completed four-strategy blocks. Fresh results and previous results
at the matching delays are shown separately, by seed/repetition/delay; the
old short/long-delay mixture is not pooled with this focused long-wait run.
No confidence intervals or significance claims are made from two seeds.

Videos are recorded for slots 0/5, both seeds, both positive delays in
repeat 1. The planned analysis asks whether the observed waiting-drop
reduction replicates, whether learned beats fixed slow/hold, and how much
post-wait dropping offsets the survival benefit. Current evidence supports
a useful waiting-stability observation, not learned-policy superiority.
Launch verification: the first fresh-process prefix replay passed all eight
fields with max error 0.0, and the first zero-delay repetition is running.
The mode-order schedule and balanced-summary handling were checked locally.


### Repeated waiting evaluation completed (2026-10-18)

The additional original-render repetitions completed at 02:00:57 China time:
48/48 arms, 576/576 nominal rollouts, and 32 video files. All 48 mechanical
prefix audits had max error 0.0, frozen source/asset/weight hashes still match,
and no per-arm runtime failure was recorded. The finite runner has exited.

Fresh positive-delay results (6/20 s, two repetitions, 88 eligible cases per
strategy) are: continue 15 waiting drops, 2 recovery drops, 57 successes;
slow 3 waiting drops, 11 recovery drops, 58 successes; hold 0 waiting drops,
20 recovery drops, 54 successes; learned 2 waiting drops, 11 recovery drops,
60 successes. The low waiting-drop observation repeated, but fixed slow
and hold are strong alternatives. Learned exceeds slow by only one fewer
waiting drop and two more task successes in this batch.

Zero-delay aggregate successes were 28/44, 30/44, 30/44 and 34/44 for the
four labels even though they apply identical control. Per-seed/repetition
zero-control spreads were 25, 50, 30 and 41.67 percentage points. The same
22 eligible initial request states were reused throughout; neither the
repeated trials nor the small method differences establish generalization
or a learned-controller superiority claim. The original success threshold
and evaluation code remain unchanged.

The recovery problem also repeats: fixed hold never dropped during waiting
but dropped 20 times after execution resumed; learned dropped 11 times after
resuming. Further work should inspect the recovery transition together with
repeat variability. The machine-readable review is
`repeated_waits_v5_20261018/completion_review.json`; fresh and prior results
remain separate in `summary.json`.
