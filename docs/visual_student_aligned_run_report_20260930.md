# Aligned visual student run (2026-09-30)

## Frozen contracts

- Teacher task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Visual task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Teacher checkpoint: `model_1999.pt`
- Teacher SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Action scale: `6.0`
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

The live evaluation uses the same camera task, semantic feature encoder, history clock, action scale, and goal-yaw contract as collection. The evaluator history update was corrected to advance only on captured frames and to pad a new episode with its first observation, matching the offline sequence dataset.

The 72-episode per-checkpoint results will be recorded here after all six frozen student checkpoints finish.
