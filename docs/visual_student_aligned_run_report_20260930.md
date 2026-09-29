# Aligned visual student run (2026-09-30)

## Frozen contracts

- Teacher task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Visual task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Teacher checkpoint: `model_1999.pt`
- Teacher SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Action scale: `1.0` (normalized environment command; no external multiplier)
- Goal yaw: `0.0`
- Camera protocol: semantic RGB features, 256x256 RGB, vision stride 2
- Student split: fixed episode split, 306 train / 76 validation episodes

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
