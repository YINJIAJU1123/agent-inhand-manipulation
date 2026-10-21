# Recovery training factor ablation, 2026-10-19

## Why this run exists

The completed V4 paired pilot scored 117/168 eligible cases (69.64%), compared
with frozen V3 at 126/168 (75.00%) and fixed slow control at 113/168 (67.26%).
V4 scored 63/84 on original goals and 54/84 on opposite goals. V3 scored 61/84
and 65/84 respectively. The pilot contains only 21 distinct eligible request
states. Repeated goals, delays and repeats are correlated, not independent scenes.

V4 changed goal coverage, adaptive training-state selection and duplicated
hard-state samples together. Duplication also changed the number of optimization
minibatches. Those results do not isolate the cause of the regression.

Completed evidence: `outputs/recovery_controller_v4/paired_20261018/completion_review.json`.
Frozen V3 remains the reference controller.

## Four training arms

| Arm | Correct-goal pairing | Failure loss weights |
| --- | --- | --- |
| `uniform_original` | Off | Off |
| `paired_only` | On | Off |
| `weighted_only` | Off | On |
| `paired_weighted` | On | On |

All four arms start from the same V3 checkpoint and use the same architecture,
reward, action scale, training-state sequence, delay sequence and optimizer rules.
The policy sees RGB-D features, proprioception, action history, clocks and the
correct goal embedding. Ground-truth pose errors are used for scoring, training
the critic and diagnostics, but are not added to actor inputs.

With pairing off, each cached physical request state is rolled out six times
under its original correct goal, using independent action noise. With pairing
on, the six draws cover the six correct goals once each. The actual goal branch
and stochastic draw ID are separate fields. Noise is matched by draw ID across
factor cells. The student, controller and scorer all receive the selected correct
goal. Request-state mechanics and observation histories must match exactly.

The eight training groups use the state and delay order from the preceding V4
training records. These are training-only records, not evaluation outcomes.
Slot weights are frozen from those same records and shared across both pairing
conditions. Weight-on applies explicit scalar weights to PPO and critic losses.
It does not duplicate trajectories or add optimization epochs. The outcome head
uses unweighted episode replay and has no gradient path to the policy.

This design controls adaptive state selection rather than estimating its effect.
The legacy V4 checkpoint is a separate reference, not the factorial both-on cell.
Its adaptive sampling and duplicate-sample optimizer implementation differ.

### Shared prefix and budget

The first four groups have uniform weights. Within each pairing condition,
weight-off and weight-on therefore share exactly the same checkpoints, optimizer
state, replay and rollouts through group four. They branch at group five.

- Each logical arm: 8 updates and 576 nominal episode slots.
- Total logical training: 32 updates and 2,304 slots.
- After sharing the identical prefix: 24 physical updates and 1,728 fresh slots.
- Optimizer rule: 3 epochs, minibatch size 256, the existing learning rates and
  exact mean KL guard of 0.02 with rollback.
- Valid decision counts and accepted optimizer steps can differ across arms
  because horizons and KL stops differ. Both are recorded.

Training seeds are 930100, 930101, 930102 and 930103. The exact order, source
record hashes and per-slot weights are in the machine-readable protocol.

## Validation registered before training

The final eighth checkpoint from each arm is evaluated. There is no selection of
an intermediate checkpoint based on these evaluation results.

- Two new evaluation seeds: 85001 and 85002.
- Original correct goal and opposite correct goal, from the same cached state.
- Simulated response delays of 6 and 20 seconds.
- Two repeats, with reversed method order in the second repeat.
- Seven methods: fixed slow, frozen V3, legacy V4 and the four trained arms.
- 1,344 positive-delay slots plus 96 zero-delay controls = 1,440 nominal slots.
- Ineligible prefix failures remain recorded. Success uses eligible cases as the
  denominator, with counts reported explicitly.
- Up to 40 videos, selecting the first eligible slot from each evaluation seed
  before inspecting method outcomes.

Success is full orientation error at most 0.16 radians for six consecutive
post-response control steps, without a previous drop, within a 300-step recovery
window. Physics continues throughout the wait. The controller operates during
the wait and the first 60 recovery steps, then hands over to the frozen student.

This is a controller diagnostic with known canonical goals and simulated delays.
It does not measure live Astra or VLM grounding. Two evaluation seeds and one
training realization per branch are insufficient for strong significance claims.

## How the regression will be located

The report computes the effect of pairing separately with weights off and on,
and the effect of weighting separately with pairing off and on. The interaction
is reported, as are both evaluation seeds and goal branches. Partial comparisons
include only groups completed by all seven methods.

The collector logs pose error at request, response and handoff, wait/recovery
drops, non-drop failures, per-phase command rate and a one-second diagnostic
trace. Response and handoff means include still-active cases only and are
explicitly selection-conditioned.

Recovery changes will be based on the completed contrasts:

1. If weighting hurts both pairing conditions, remove weighting first.
2. If pairing hurts both weighting conditions, inspect goal-switch coverage and
   reachability under the frozen student.
3. If all continued-training arms regress, test anchoring the actor to V3 to
   limit global speed and residual drift.
4. If failures concentrate after response, separate waiting and recovery updates
   and inspect the handoff schedule.

No automatic baseline replacement is performed.

## Implementation and verification

Entry points:

- `scripts/recovery_factorial/run.py`
- `scripts/agentic/collect_recovery_factorial.py`
- `scripts/recovery_factorial/update.py`
- `scripts/recovery_factorial/weighted_ppo.py`
- `scripts/recovery_factorial/factorial_analysis.py`

Four focused unit tests passed. They check independent factor assignment,
unchanged unique sample count, exact equivalence to the V3 update for uniform
weights, weighted-objective gradients and KL rollback. A separate synthetic
CLI check passed two updates, including checkpoint save and optimizer resume.
The synthetic data reside under `/tmp` and are not experimental evidence.

The actual Isaac Lab smoke collector completed 12 nominal slots. Exact physical
prefix and goal-switch checks passed. Its outcomes are only an execution check
and are excluded from the factorial comparison.

## Run artifacts

Run root: `outputs/recovery_factorial/ablation_20261019`.

- `protocol.json`: registered protocol and frozen source/weight hashes.
- `frozen_sources_and_weights.tar.gz`: source and weight snapshot.
- `progress.json`: current finite runner stage and completed budgets.
- `training.json`: all logical arm histories, including shared-prefix references.
- `models/`: final trained controllers after training completes.
- `summary.json`: balanced validation contrasts as groups complete.
- `diagnosis.json`: diagnostic summary after all registered evaluation finishes.
- `logs/`: subprocess commands and logs.

The finite local RTX 3090 runner was launched on 2026-10-19 at approximately
00:26 Asia/Shanghai. Training completed at 03:58 and validation at 07:30 on
2026-10-19. All 290 stages completed. The runner stops on a
failed stage, altered frozen source/weight/cache, mismatched physical replay or
stage timeout, and retains the failed outputs for review. It does not silently
retry or exclude seeds.


## Completed results

All 1,440 nominal evaluation slots finished. Each positive-delay method has 192
eligible cases, drawn from 24 distinct request states. The 96 zero-delay controls
are reported separately. The 120 exact physical replay audits passed, all 212
frozen source/asset/weight entries, six caches and 28 model files matched, and 40
nonempty videos were saved.

| Method | Successes | Success rate | Wait drops | Recovery drops | Non-drop failures |
| --- | --- | --- | --- | --- | --- |
| Fixed slow | 125/192 | 65.10% | 5 | 23 | 39 |
| Frozen V3 | 138/192 | 71.88% | 9 | 8 | 37 |
| Legacy V4 | 135/192 | 70.31% | 8 | 17 | 32 |
| Neither factor, continued training | 140/192 | 72.92% | 5 | 23 | 24 |
| Pairing only | 128/192 | 66.67% | 6 | 17 | 41 |
| Weighting only | 129/192 | 67.19% | 4 | 24 | 35 |
| Pairing and weighting | 124/192 | 64.58% | 7 | 23 | 38 |

With weights off, pairing costs 6.25 percentage points. With pairing off,
weighting costs 5.73 points. Their effects remain negative at the other factor
level, and all four effect directions agree across the two seeds. Effects vary
by delay and repeat, so they should not be described as universally harmful.
There is only one training realization per branch.

The unpaired, unweighted continuation gains two successes over V3 but increases
recovery drops from 8 to 23. V3 has one drop before the two-second handoff and
seven afterwards. The continued model has eleven before handoff and twelve
afterwards. The weighted-only model has twelve in each phase. This shows where
the observed drops occur. It does not establish whether the recovery actor or
the state produced by waiting caused the drop.

Frozen V3 remains the baseline. No new checkpoint has been promoted. Completion
review: `outputs/recovery_factorial/ablation_20261019/completion_review.json`.

A follow-up crosses frozen V3 and `uniform_original` actors across waiting and
recovery. It reruns both same-checkpoint controls and tests the two mixed
combinations. See [the phase crossover protocol](recovery_phase_crossover_20261019.md).
