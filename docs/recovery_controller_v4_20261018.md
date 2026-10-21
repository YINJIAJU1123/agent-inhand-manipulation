# Correct-goal pairing and failure-state curriculum, 18 October 2026

## Motivation and evidence boundary

The completed v3 direct-input diagnostic achieved 61/80 successes with full inputs,
50/80 with fixed slowing, 48/80 with frozen controller vision, 57/80 with a shifted
controller goal, and 55/80 with both interventions. There were 20 distinct eligible
request states and two reset seeds. Full v3 had 10 drops against nine for slowing.
Useful direct semantic dependence remains unresolved. Complete prior evidence is
in `outputs/recovery_controller_v3/rq2_inputs_20261018/completion_review.json`.

## New experiment

Output: `outputs/recovery_controller_v4/paired_20261018`.
Launcher: `scripts/recovery_controller_v4/run_paired.py`.
No scheduled monitor is created. This is a finite local GPU process that runs
training and then validation, and stops on an infrastructure or audit failure.

The first pilot retains the v3 network, action parametrization, loss, PPO KL guard,
frozen student, adapter, action scale, renderer and success rule. It starts from
v3 checkpoint SHA-256
`cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042`.
Optimizer state is reset once for the new training stage, then carried between
updates. The final pre-registered eighth update is evaluated, with no selection
using evaluation outcomes.

### Correct-goal intervention

A fresh simulator first replays an identical cached command prefix. After the
request state is reached, only the target orientation and delivered goal are
changed. All six branches use the same mechanical state, proprioception, RGB-D
history and symbol layout at that instant. Student, controller and scorer all
receive the selected correct goal. Changing a goal does not advance simulation.
Branch zero preserves the prefix goal, and branch one selects its opposing face.
Across six branches every slot sees every canonical face exactly once.

The current goal is supplied before the wait and reaffirmed by the simulated
response. This does not expose an unknown future response, and is not a live VLM
experiment. Changing the goal may legitimately change the student proposal.
The actor does not receive the goal face id, target quaternion, object pose,
orientation error or remaining delay. Ground-truth geometry is used only by the
training critic/reward and independent scorer.

### Training and failure sampling

- Four training seeds: 930100, 930101, 930102, 930103, with 12 slots each.
- Eight updates, each containing all six goal branches collected under the same
  pre-update policy. There are 576 nominal training slots, with correlated reuse
  of up to 48 physical request states.
- Delays by group: 2, 6, 12, 6, 12, 6, 2, 12 seconds.
- The first four groups cover each training cache once. The remaining groups
  sample training caches using a 50% uniform and 50% failure-weighted mixture.
- A physical slot with at least two failed goals or any drop in its previous
  training group receives multiplicity two in the next PPO update. This weight
  is fixed before collection and applies equally to all six goals. Prefix failures
  remain in the record and never enter the valid-decision loss.
- All trajectories are freshly sampled with the current policy. Failed trajectories
  identify request states to revisit, not stale off-policy action labels.
- This deliberately changes the training distribution. The first pilot combines
  paired goals and failure sampling, so it cannot isolate either component's effect.
- There is no reward or loss that merely forces different actions for different
  goals. Useful differences must be supported by correct-goal task outcomes.

### Validation

Two disjoint seeds, 84001 and 84002, each with 12 slots. Each request state is
branched into the original and opposing correct goals, evaluated at 6 and 20
seconds, with two repeats. Fixed slowing, frozen v3, and final v4 share the same
state caches and task goals. Method order is reversed on the second repeat.
This gives 576 nominal positive-delay slots and 96 zero-delay controls, or 672
in total. Every registered goal and failure is retained, including hard goals.
The two seed clusters and shared states limit independence.

Success requires full orientation error at most 0.16 radians for six consecutive
post-response control steps, with no prior drop, within 300 post-response steps.
The learned controller acts through the wait and first 60 recovery steps. The
frozen student then continues. Physics runs at 120 Hz, command updates at 30 Hz,
and visual/controller updates at 15 Hz.

Request probes forward all six correct goals on identical sensor observations,
including recomputing the student's corresponding proposal. These probes measure
immediate command dependence, not physical success or isolated dependence on the
controller's explicit goal channel. A separately trained goal-free controller
remains a future ablation.

## Validation before launch

Four unit tests passed, including CUDA goal permutations, all-six-goal coverage,
opposing-face mapping, rejection of mixed-policy groups, episode-preserving
trajectory concatenation, equal hard-slot weights across goals, and rejection of
physical mutation during target switching.

A separate real simulator smoke test used seed 930099, outside training and
validation. Original and opposing goals had identical physical and request-history
SHA-256 values and zero simulated steps during the switch. Both exported on-policy
rollouts successfully. Smoke outcomes are engineering checks, not validation data.

Source/assets/weights, the protocol, all cache hashes, per-branch goal audits,
selected training weights, optimizer statistics, raw outcomes and video outputs
are retained beneath the output directory. Training sources are frozen for this
run. See `progress.json`, `training.json`, and eventually `summary.json` for status.
