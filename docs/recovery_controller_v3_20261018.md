# Waiting and recovery V3: variance diagnosis and optimization

## Status

The user approved continued diagnosis and optimization after the V2 completion
policy failed to beat fixed slow (30/46 versus 34/46 eligible positive-delay
trials). Those V2 results and checkpoints are retained. V3 is a development
experiment. The completed comparison is summarized below: V3 has a small
aggregate advantage, but a reliable advantage over fixed slow is not established.

The finite V3 runner started at **2026-10-18 14:25:54 China time**, PID 1070193,
on the available local RTX 3090. Its progress is saved in
`outputs/recovery_controller_v3/pilot_20261018/progress.json`. It runs initial
zero-delay controls, training, then repeated comparisons, with finite stage
timeouts and failure reporting. This is not a recurring scheduled task.

## 1. Isolating the repeat variance

Five fresh simulator processes used the same seed-82002 request cache, 12
eligible states and 300 control steps: two original-render closed-loop runs,
one replay of the first run's recorded actions, and two diagnostic runs with
a 3x3 Gaussian RGB filter. Each process exactly replayed the physical prefix.

- Original closed-loop repetitions first differ at control step **2** (zero
  based). Before that step, physical state and proprioception still match
  exactly. The new student feature differs by at most 0.004902 and its action
  by 0.004917; the physical trajectories then diverge.
- With the first run's **recorded actions replayed**, all 300 before/after
  physical states, actions and proprioception match exactly, despite fresh
  render differences. Success and drop outcomes also match case by case.
- Both original closed-loop runs achieve 6/12, but **four cases have different
  success outcomes**. Equal aggregate rates therefore do not imply stable
  trajectories.
- The Gaussian diagnostic gives 8/12 in both repetitions, with two case-level
  disagreements. It still diverges at step 2 and does not reduce the first
  action discrepancy. This is insufficient evidence of a robust fix; the
  filter is **not deployed** in V3 training or comparison.

The evidence isolates image-dependent action variation as a source of closed-
loop divergence in this test. It does not prove universal simulator determinism.
Recorded-action playback is an open-loop diagnostic, not a policy performance
baseline. The concatenated physical-state errors mix positions and velocities;
their maximum must not be interpreted as an object displacement.

Evidence: `outputs/recovery_controller_v3/variance_20261018/comparison.json`,
per-arm `trace.pt`, RGB frames and completion files. Reproduction scripts:
`scripts/recovery_controller_v3/diagnose_variance.py` and `run_variance.py`.

## 2. Optimization changes

V2 used a shared feature encoder for actor, critic and outcome prediction.
Auxiliary/value losses could move policy features outside the actor-only PPO
clipping objective. V2 also recorded large approximate KL updates. This is a
plausible optimization problem, not a proven sole cause of the negative result.

V3 addresses it as follows:

1. Independent policy, value and outcome encoders and optimizers. Critic or
   outcome updates cannot change actor parameters.
2. Value loss normalized by the terminal success reward scale of 12.
3. Exact old/new Gaussian policy KL over the rollout, limited to **0.02**.
   A violating step restores both policy weights and optimizer state.
4. Zero-mean action initialization reproduces **fixed slow during waiting and
   immediate frozen-student execution after the response**. Slowing the rejoin
   must be learned; a two-second slow ramp is no longer imposed at initialization.
   Residual influence still vanishes within two seconds. The residual bound
   is 0.02 and initial exploration standard deviation is exp(-3).
5. Forward the original student action directly at full speed, avoiding a GPU
   subtract/add rounding difference. Exact CPU and CUDA equality is tested.
6. Episode-uniform outcome replay with entire reset seeds held out for
   validation (`seed % 5 == 0`), preventing adjacent frames from the same
   episode appearing in both fitting and validation. Report Brier scores
   against a training-frequency prior, not only classification accuracy.

The outcome labels come from changing behavior policies. This is an empirical
rollout-outcome predictor, **not fixed-policy counterfactual recoverability or a
calibrated safety guarantee**. It does not yet control agent call timing. The
pilot changes several components together; a gain would require later ablations
before attribution to a particular learning mechanism.

Actor inputs remain four compact RGB-D frames, robot proprioception, student
proposal, current/latched/previous targets, elapsed request/response clocks and
the delivered 768-dimensional language embedding. Ground-truth object state,
orientation error, face ID and remaining latency are excluded from the actor.
The privileged critic may use object-state quantities during training.

## 3. Verification before launch

Four focused regression tests pass, covering exact CPU/CUDA baseline commands,
action bounds, critic/outcome isolation, excessive-KL rollback, and whole-seed
outcome-data separation. The two-iteration simulator smoke completed with finite
weights and nonzero actor updates; exact KL was 0.01975 and 0.01990. Replay
training ran on the second iteration. These are plumbing checks, not matched
performance estimates.

Smoke artifacts are under `outputs/recovery_controller_v3/smoke_20261018`.
The smoke preceded the exact-resume arithmetic correction; its original source
is retained in `source_before_exact_resume_fix`. The final correction passed
CPU/CUDA equality tests before the main pilot was frozen.

Implementation: `scripts/agentic/recovery_controller_v3.py`,
`train_recovery_controller_v3.py`, `scripts/recovery_controller_v3/run_pilot.py`,
and `tests/test_recovery_controller_v3.py`.

## 4. Registered finite comparison

- Reuse evaluation seeds **82001 and 82002**, with 24 nominal cached states,
  23 eligible states. These are previously examined development seeds.
- Two fresh-process zero-delay frozen-student repetitions per seed: 48 nominal
  trials. Retain variation and stop if all controls fail for either seed.
- Train V3 for 32 iterations, 384 nominal episodes; reset seeds 920000–920031,
  delays 2/6/12 seconds. Use the final checkpoint without eval-based selection.
- Compare fixed slow, hold+smooth, the saved V2 completion policy, and V3 at
  delays 6/20 seconds, both seeds and two repetitions: 384 nominal trials.
  Total evaluation budget is **432 trials**.
- Randomize method order within each seed/delay group and reverse it in the
  second repetition. Aggregate only complete four-method comparison groups.
- Record slots 0/5 for slow, hold+smooth and V3 in positive-delay repetition 1:
  24 planned videos. Preserve individual successes, waiting drops, recovery
  drops, predictor outputs and learned residual magnitudes.
- Replay identical prefix commands in fresh processes and require eight
  physical/control fields to agree exactly. Frozen source, asset, cache,
  student, adapter and model hashes are checked and archived.

The original RGB-D frontend, rendering, frozen student and action scale remain
in use. Success remains orientation error <=0.16 rad for six consecutive
post-response steps, no prior drop, within 300 recovery control steps.
Physics runs throughout the wait. Goals are already delivered and a simulated
response reaffirms them; **this pilot does not invoke live Astra or VLM APIs**.

Repetitions quantify variability on reused states, not 432 independent starting
conditions. Review per-goal and per-seed outcomes and repeat spread before
concluding that V3 improves completion. Broader objects and live agent decision
learning follow only after a reproducible benefit over strong controls.


## 5. Completion review — 2026-10-18 17:22 China time

The runner completed at **16:12:06 China time** and exited. Training finished
all 32 iterations (384 nominal episodes, 357 eligible), taking 39.2 minutes.
All 432 planned evaluation trials and 36 physical-prefix audits completed,
with no failure files, zero prefix errors and identical eligibility. All 189
frozen source/asset/model/cache hashes and the final checkpoint hash match.
There are 24 nonempty video files. The final checkpoint SHA-256 is
`cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042`.

Each method has 92 eligible positive-delay trials (6/20 seconds, two repeats),
reusing 23 distinct eligible request states. The separate 48 nominal zero-delay
trials are excluded from this table.

| Method | Sustained success | Waiting drops | Recovery drops | Timeouts |
|---|---:|---:|---:|---:|
| Fixed slow | 62/92 (67.4%) | 3 | 9 | 18 |
| Hold + smooth | 50/92 (54.3%) | 8 | 11 | 23 |
| V2 completion | 61/92 (66.3%) | 6 | 9 | 16 |
| V3 completion | 65/92 (70.7%) | 4 | 10 | 13 |

V3's aggregate advantage over slow is **three successes / 3.3 percentage
points**, with more drops and fewer timeouts. This does not establish improved
waiting safety. Case-matched descriptive counts are 17 V3-only successes versus
14 slow-only successes, with 48 successes and 13 failures shared by both.
The repeated states must not be treated as independent samples.

The advantage is not consistent across conditions:

| Group | Fixed slow | V3 |
|---|---:|---:|
| Delay 6 seconds | 31/46 (67.4%) | 35/46 (76.1%) |
| Delay 20 seconds | 31/46 (67.4%) | 30/46 (65.2%) |
| Seed 82001 | 31/44 (70.5%) | 38/44 (86.4%) |
| Seed 82002 | 31/48 (64.6%) | 27/48 (56.3%) |
| Repeat 1 | 28/46 (60.9%) | 32/46 (69.6%) |
| Repeat 2 | 34/46 (73.9%) | 33/46 (71.7%) |

Zero-delay controls are 10/11 in both repetitions for seed 82001, and 6/12
versus 9/12 for seed 82002. Visual closed-loop repeat variability therefore
remains material relative to the aggregate advantage.

### Learning diagnostics

- Maximum accepted/final exact KL over training is **0.0199921**; seven
  iterations exercised rollback. The requested optimizer bound worked.
- V3's active waiting interpolation rate averages **0.1551**, versus fixed
  slow's 0.15. Recovery rate averages **0.99894**. Mean residual L2 norms
  are 0.000933 during waiting and 0.000577 during recovery. The learned
  behavior stays close to its strong initialization. These are decision-
  weighted descriptions, not a separate ablation of the initialization.
- The final outcome replay validation Brier is **0.2380**, versus **0.2193**
  for a constant prediction based on the training frequency (lower is better).
  The outcome head has not demonstrated reliable predictive value. On the
  development evaluation, Brier is 0.2337 at the request and 0.2482 at arrival;
  arrival predictions remain insufficiently discriminative. Do not use this
  head as a trusted query/stop confidence signal yet.

### Interpretation and next optimization target

The training mechanics now enforce the intended policy-update bound, and the
completed comparison gives a small aggregate improvement. It does **not** yet
support a robust learned-controller advantage, an improvement in drop safety,
or a paper-level recoverability-learning claim. The near-baseline action rates
also leave open how much of the gain comes from the changed initialization.

Prioritize the 20-second and seed-82002 failures, especially transitions into
recovery, and improve the outcome-learning targets before enlarging the same
evaluation. Any follow-up comparison should retain the strong slow baseline,
matched initial states and repeated measurements. A separate initialization
ablation and unseen-state validation are needed before attributing an eventual
gain to the learned mechanism. This completed pilot still uses known goals and
simulated responses; it is not a new live-Astra result.

Machine-readable review: `outputs/recovery_controller_v3/pilot_20261018/completion_review.json`.
Raw results: `summary.json`, `train/training.json`, and per-arm files under `eval/`.
