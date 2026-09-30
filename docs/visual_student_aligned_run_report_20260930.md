# Aligned visual student run (2026-09-30)

## Frozen contracts

- Teacher task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Visual task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Teacher checkpoint: `model_1999.pt`
- Teacher SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Action scale: `1.0` (normalized environment command; no external multiplier)
- Goal yaw: `0.0`
- Camera protocol: semantic RGB features, 256x256 RGB, vision stride 2
- Student split: fixed episode split for the DAgger round-2 aggregate, 675 train / 169 validation episodes

## Environment alignment correction

The first visual collection used a primitive cuboid while the frozen state teacher used the USD DexCube. This changed contact geometry and made the old visual-student result unsuitable as a teacher-to-student verdict. The visual task now reuses the exact object configuration from the state teacher, including the USD asset and rigid-body parameters.

The previous v2 data and live results are retained as diagnostic evidence only. They must not be used as the main CVPR result.

## Corrected teacher collection

The corrected formal collection requested 387 episodes and completed 390 vectorized episodes. The quality filter found 382 successful episodes and 8 drops; there were no timeouts.

- Usable episodes: `382/390 = 97.95%`
- Drop rate: `8/390 = 2.05%`
- Filtered samples: `6,554`
- Captured batches: `118`
- Structural audit: zero batch mismatches, duplicate keys, noncontiguous steps, stride mismatches, and terminal-count mismatches
- RGB/depth camera health: passed

Only the 382 success-only episodes were used for student training. The resulting cache has 6,554 samples with 36-D semantic RGB features, 6-D language features, and frozen checkpoint/action metadata.

## Student fit

Each condition was trained for 20 epochs with the fixed episode split. Final validation MSE:

| Memory mode | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Plain | 0.142652 | 0.145276 | 0.142359 |
| Evidence | 0.143710 | 0.142486 | 0.145242 |

## Closed-loop evaluation

The live evaluation uses the same camera task, semantic feature encoder, history clock, action scale, and goal-yaw contract as collection. The evaluator history update was corrected to advance only on captured frames and to pad a new episode with its first observation, matching the offline sequence dataset. The first live run also exposed that the direct environment already consumes normalized commands; the previous `6.0` multiplier was removed and the frozen action contract is now `1.0`.

## Closed-loop results and diagnosis

The first corrected live run used the old `6.0` multiplier and produced only 1–4/72 successes with 39–57% drops. The direct environment already consumes normalized `[-1,1]` commands, so this multiplier was removed. With normalized actions, drops fell to 7–22%, but success remained about 2.8% on the semantic baseline.

A spatial RGB grid and normalized depth grid were then added, and the previous-action portion of proprioception was clamped to the same normalized contract. A 24-episode depth ablation reached 3/24 (12.5%) in one all-episode split, but was not stable enough to freeze as a final result.

The next experiment is DAgger: the visual policy drives the environment while the frozen teacher labels the visited states. The first 200-episode DAgger shard contains 22,912 labeled samples; its own student-visited state distribution is substantially different from the teacher-only demonstrations. A plain DAgger retrain reached 7/24 (29.2%) in the first closed-loop check, confirming covariate shift as a real bottleneck. The DAgger dataset and collector are now the basis for the next iteration.

## DAgger round 2 quick verification

The second round used `dagger_plain_seed1.pt` as the behavior policy while the
frozen teacher labeled every visited state. It completed 200 requested
episodes and produced 23,872 aligned RGB-D samples. The aggregate cache combines
the corrected teacher cache, round 1, and round 2:

- `53,338` samples and `844` episode groups
- spatial RGB-D feature width `1,060` (16x16 RGB and depth grids)
- normalized 21-D action contract, `action_scale=1.0`
- aggregate cache SHA256:
  `7907b6f98b1cde77e7d1dbe5343813a5684cbe706168c18ee16311cfa28b0ca7`

Two representative models were trained for 40 epochs on the fixed split. The
quick closed-loop check used 24 episodes per model, four vectorized environments,
the same camera/task protocol, and no external action multiplier:

| Model | Success | Drop | Checkpoint SHA256 |
| --- | ---: | ---: | --- |
| DAgger round 2 plain | 14/24 (58.33%) | 8/24 (33.33%) | `7e242e572dec13e3e8ab203cee8d32273a8fb24b23394bb459916433e8531f0b` |
| DAgger round 2 evidence | 12/24 (50.00%) | 10/24 (41.67%) | `21f1e3b9db1d48618cdfe53543182ebc47e232a5b834988ac13610556ba929a6` |

This is a clear improvement over the first DAgger quick check (7/24 plain and
5/24 evidence), but the drop rates and the small evaluation sample do not yet
support calling the visual student stable. The next gate is a larger paired
evaluation of the plain model, followed by another DAgger round only if the
failure cases still show systematic covariate shift.

## DAgger rounds 3 and 4

The 96-episode check on round 2 exposed that its 24-episode result was
optimistic: plain reached 36/96 (37.50%) with a 45.83% drop rate. Round 3
used the round-2 plain policy for another 240-episode teacher-labeled shard,
then reached 50/96 (52.08%) with a 39.58% drop rate.

Round 4 used the round-3 plain policy. Its behavior rollout contained 19,648
samples from 240 episodes, with 165 teacher-success episodes and 38 drops.
After merging all four DAgger rounds with the corrected teacher cache, the
aggregate contains 95,002 samples across 1,386 episode groups. The quick
24-episode check was 16/24 for plain and 21/24 for evidence. The larger
96-episode confirmation was:

| Model | Success | Drop |
| --- | ---: | ---: |
| DAgger round 4 plain | 55/96 (57.29%) | 34/96 (35.42%) |
| DAgger round 4 evidence | 73/96 (76.04%) | 18/96 (18.75%) |

The evidence model is now the best visual-student candidate. Its remaining
errors are concentrated on target faces 4 and 5 (69.2% and 60.0% success in
this split), while faces 0--3 are between 76.2% and 100%. This points to a
target-face and late-orientation coverage gap rather than a general camera or
action-contract failure. The model is improved but is not yet at the frozen
teacher's 90%+ stability gate.

## Targeted face-4/5 supplementation and unified candidate

The round-4 evidence model was used for a targeted DAgger shard that records
only target faces 4 and 5. It added 15,043 samples from 240 episodes (face 4:
103/113 teacher-success episodes; face 5: 84/127). The resulting unified cache
contains 110,045 samples across 1,635 episode groups.

A unified evidence model trained on this cache reached the following 96-episode
check under the frozen RGB-D protocol:

- Overall: `79/96 = 82.29%` success, `15/96 = 15.63%` drops
- Face 0: 91.67% success
- Face 1: 86.36% success
- Face 2: 66.67% success
- Face 3: 72.73% success
- Face 4: 85.71% success
- Face 5: 93.75% success

The targeted data fixed the face-5 gap and kept face 4 high. Face 2 and face 3
now set the remaining lower bound. This is strong enough to start the larger
training and paired evaluation phase, but it is still below the frozen teacher
gate and should not yet be reported as final student stability.
