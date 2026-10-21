# Goal-conditioned waiting and recovery controller — first pilot

## Research hypothesis and scope

The user approved beginning this direction on 2026-10-18. The hypothesis is that
waiting actions should preserve the ability to complete the delivered semantic
goal after reasoning finishes. The preceding v5 diagnostic motivates this:
fixed hold had zero waiting drops but 20 recovery drops in 88 repeated eligible
cases. Those cases reused 22 request states, and zero-delay outcome variability
exceeded the observed learned-versus-slow difference. Neither learned-policy
superiority nor a specific transition failure mechanism is established.

This first pilot learns through waiting **and the first two seconds after the
response**. It then hands control to the frozen student for the rest of the
10-second recovery window. All six canonical goals are already delivered;
a simulated response reaffirms the goal. There are no live Astra/VLM calls,
newly grounded goals, learned query decisions, active observations or new objects
in this experiment. These require later experiments after the mechanism works.

## Model and training

- Four-frame compact RGB-D history, robot proprioception, student proposal,
  current/latched/previous targets, elapsed request/response clocks, and a
  normalized 768-dimensional delivered language embedding.
- A learned sensor/goal representation feeds continuous speed and 21 bounded
  joint-command residuals. Decisions update at 15 Hz; physics is 120 Hz and
  control 30 Hz. Residual amplitude is bounded by 0.04 in normalized action
  units and smoothly vanishes over the two-second rejoin. Action scale is 1.
- Zero-initialized actor means exactly implement fixed slow followed by a linear
  speed ramp. Training samples small Gaussian perturbations; deterministic
  evaluation uses the policy mean. Commands remain in [-1, 1].
- PPO updates both waiting and recovery decisions. The completion objective uses
  success +12, drop -8, horizon failure -3, orientation progress weight 0.15 and
  command-change penalty 0.002. Drop/success metrics come from the environment's
  pre-reset metrics. Terminal continuation outcomes are attached to the final
  residual decision, treated as a macro transition through frozen execution.
- A privileged critic may see orientation error, object distance and velocities.
  None of these, nor target face IDs or remaining latency, enter the actor.
- An auxiliary outcome head learns the realized complete-task outcome from the
  policy's own rollouts, weighted equally per eligible episode. This is an
  **on-policy completion predictor**, not a counterfactual expert recoverability
  probability or a calibrated safety guarantee. Held-out Brier diagnostics are
  reported; the head does not yet schedule agent calls.

The matched `wait_survival` ablation uses the same initialization, seeds,
parametrization, two-second bridge and training budget, but rewards only surviving
waiting (+2) and penalizes waiting drops (-8), retaining the action penalty.
It has no completion/progress reward or outcome-head loss. This compares the
complete objective package with survival training; it does not isolate the
auxiliary loss alone. That separate ablation is deferred.

## Frozen protocol and first finite run

Implementation:
- `scripts/agentic/recovery_controller_v2.py`
- `scripts/agentic/train_recovery_controller_v2.py`
- `scripts/recovery_controller_v2/run_pilot.py`
- `tests/test_recovery_controller_v2.py`

The standalone simulator smoke used seed 919900, 12 environments and a 2-second
wait. It completed one PPO update and saved its checkpoint: 10 eligible requests,
5 sustained successes, 1 waiting drop and 1 recovery drop. This tests execution
and learning plumbing, not policy quality. Four CPU tests cover handoff/hold
semantics, action bounds, delayed credit/dead-slot masking and actor learning
without privileged inference inputs.

The finite pilot is planned at
`outputs/recovery_controller_v2/pilot_20261018` on the available local RTX 3090:

1. Generate prefix caches for evaluation seeds **82001 and 82002**, 12 envs each.
2. Two fresh-process zero-delay continue repetitions per seed. Preserve their
   variation; stop if every eligible zero-delay trial fails for either seed.
3. Completion training: 32 iterations / 384 nominal episodes, reset seeds
   920000–920031, cycling delays 2/6/12 seconds.
4. Survival training: identical 32-iteration budget and seed/delay schedule.
5. Use the final checkpoints, with no evaluation-based checkpoint selection.
   Run learned and survival zero-delay bypass controls per evaluation seed.
6. Evaluate delays 6 and 20 seconds with continue, slow, hold, slow+smooth,
   hold+smooth, learned completion and learned survival: 336 nominal positive
   delay trials, plus 96 zero-delay repeated controls = **432 evaluation trials**.

Evaluation uses at most 24 distinct prefix states. Training and evaluation seeds
are disjoint. Evaluation order is randomized once in the saved protocol; all
trials, including failures, are retained. All arms replay identical cached prefix
commands in fresh simulators and audit eight physical/control fields for exact
agreement. Cached visual histories initialize each branch; subsequent observations
come from the branch's current physical state. No intermediate physical-state
restoration is used. The original RTX appearance and its measured variability
are retained. All zero-delay modes bypass both learned actions and the bridge.

Success remains <=0.16 rad for six consecutive post-response steps without prior
drop, within 300 recovery steps. Waiting drops, recovery drops and sustained
success are separate. Results are aggregated only for complete seven-method
positive-delay blocks. Zero-delay variation accompanies every comparison; no
significance or generalization claim is made from two seeds. Videos are recorded
for slots 0/5 of learned completion and both smooth fixed baselines at positive
delays. The process has finite watchdogs, source/asset/weight snapshots, immutable
cache/checkpoint hashes, and status files. It is not a scheduled task.

## Advancement criterion

First inspect repeat variation, per-goal failures and transition videos. Advance
to broader object/semantic variation and live agent scheduling only if complete-
task training offers a reproducible benefit over the strong fixed smooth controls.
If it does not, retain the outcome and diagnose training/execution instead of
labelling the pilot successful. Survival alone is not the primary task metric.

## Launch record

The finite runner was started on 2026-10-18 at approximately 03:26 China time,
PID 1011984. The smoke checkpoint contains 221,614 parameters, all finite;
its initially zero actor weights and bias both changed after the PPO update.
Current status is recorded in `outputs/recovery_controller_v2/pilot_20261018/progress.json`;
training curves and checkpoints are stored separately under each objective.
Comparative results are pending.

## Completion review (2026-10-18)

The finite run completed at **05:49:18 China time**. Both objectives completed
32 updates (384 nominal training episodes each). All 432 planned evaluation
trials completed without runtime failures. There are 24 nonempty videos and 36
exact mechanical-prefix audits (maximum error 0.0); source, asset, frozen model
and evaluated checkpoint hashes remained unchanged. The runner has exited.

Positive-delay results only (6 and 20 seconds; 46 eligible cases per method,
reusing **23 distinct eligible request states**):

| Method | Sustained successes | Wait drops | Recovery drops |
|---|---:|---:|---:|
| Continue | 28/46 (60.9%) | 11 | 0 |
| Slow | 34/46 (73.9%) | 2 | 6 |
| Hold | 27/46 (58.7%) | 4 | 10 |
| Slow + smooth | 26/46 (56.5%) | 2 | 9 |
| Hold + smooth | 32/46 (69.6%) | 4 | 5 |
| Learned completion | 30/46 (65.2%) | 4 | 6 |
| Learned waiting survival | 30/46 (65.2%) | 2 | 2 |

The complete-task objective has **not demonstrated an advantage** over survival
training or fixed slow. Smoothing has mixed observed effects, so the pilot does
not establish that a universally smooth rejoin improves this task. Aggregate
training rates are not matched-state comparisons: render variation also caused
different prefix eligibility between the two training runs.

Both learned policies still use waiting interpolation rates near fixed slow:
active-decision mean 0.1601 for completion and 0.1581 for survival, versus 0.15
for the fixed baseline. This describes the rate component only; joint residuals
were not recorded in the decision logs and are not claimed to be zero.

The completion predictor separates eventual successes/failures only weakly:
mean prediction 0.534 versus 0.510 at the request, and 0.538 versus 0.517 at
response arrival. Brier scores are 0.234 and 0.231. A post-hoc constant forecast
using the corresponding training outcome frequency scores 0.231 and 0.211 on
these same samples. The head is not ready to drive agent scheduling.

Zero-delay identical-control repeats remain variable: seed 82001 spans
9–11/11 successes; seed 82002 spans 2–8/12. The latter is a 50-percentage-point
spread despite exact request mechanics and identical control laws. This limits
causal interpretation of all small method differences. No statistical or
new-object generalization claim is made, and no live agent inference occurred.

Machine-readable review: `outputs/recovery_controller_v2/pilot_20261018/completion_review.json`.
The next priorities are diagnosing zero-delay variability, improving the outcome
learning signal/optimization, and examining conditional handoff behavior before
moving to broader live-agent experiments. This pilot remains a useful negative
result, not evidence that the proposed learned method has succeeded.
