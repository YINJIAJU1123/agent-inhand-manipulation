# Waiting and recovery phase crossover, 2026-10-19

The completed training factorial experiment found that unpaired, unweighted
continuation scored 140/192 versus frozen V3 at 138/192, but had 23 recovery drops
versus eight. Eleven versus one of those drops occurred during the first two
seconds after the response. Waiting changes the state in which recovery begins,
so drop timing alone does not identify which phase caused the regression.

This follow-up exchanges only the frozen actor selected for each phase.

| Arm | Waiting actor | Recovery actor |
| --- | --- | --- |
| `v3_v3` | V3 | V3 |
| `v3_new` | V3 | Unpaired unweighted continuation |
| `new_v3` | Unpaired unweighted continuation | V3 |
| `new_new` | Unpaired unweighted continuation | Unpaired unweighted continuation |

RGB-D history, proprioception, the correct goal, action mapping, two-second
recovery bridge and frozen student remain the same. No state is reset at the
phase switch. No parameters are trained or overwritten. Both actors have the
same architecture. The outcome head does not select actions or stop episodes.

## Registered evaluation

- Previously observed seeds 85001 and 85002. This is exploratory attribution,
  not a new held-out performance test.
- Original and opposite correct goals, waits of 6 and 20 seconds, two repeats.
- Four combinations, 64 fresh simulator arms and 768 nominal episode slots.
- Both same-checkpoint controls are rerun. Previous evaluation results are not
  substituted for controls. Previous zero-delay controls remain separate.
- Method order is randomized and reversed in repeat two.
- Slot zero videos are recorded for repeat one, chosen before outcomes.
- Full orientation within 0.16 radians for six consecutive post-response steps,
  no previous drop, within a 300-step recovery window.
- Exact request-state physical/history identity is checked against the preceding
  experiment. Frozen source, cache and checkpoint hashes are checked each arm.
- Stop on an error, no silent retries or exclusion of seeds.

The analysis compares recovery actors separately under each waiting actor and
waiting actors separately under each recovery actor. A mixed combination is not
automatically promoted based on these already observed seeds. A later change
requires a fresh validation set within the two-seed limit.

## Verification and status

Real frozen checkpoint and observation tests passed: loading preserves the torch
random state, using V3 in both phases exactly preserves its action distribution,
the actor switches at the response boundary, and recovery parameters are frozen.
The finite RTX 3090 runner was launched on 2026-10-19. Results are pending.

Output: `outputs/recovery_phase_crossover/phase_20261019`.

- `protocol.json`: exact schedule, checkpoint identities and frozen inputs.
- `progress.json`: current finite batch progress.
- `summary.json`: contrasts over completed balanced groups.
- `eval/`: full per-case records, phase-labelled decisions and videos.

Entry point: `scripts/recovery_phase_crossover/run.py`.
