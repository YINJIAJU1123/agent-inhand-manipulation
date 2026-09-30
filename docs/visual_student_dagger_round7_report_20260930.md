# VisERDex-style visual student: DAgger round 7 (2026-09-30)

## Candidate and frozen protocol

This run uses the frozen Revo3 teacher and the VisERDex-style visual-language student contract:

- Teacher task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Student task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Teacher checkpoint: `model_1999.pt`
- Teacher SHA-256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- RGB-D semantic frontend: 16x16 RGB grid + 16x16 normalized depth grid + six marker slots
- Vision stride: 2 control steps
- History: 8 captured observations
- Action scale: `1.0` normalized command
- Evaluation: 96 episodes, four vectorized environments, 600-step horizon

The student observes RGB-D, language face instruction, hand proprioception and observation history. It does not receive object pose, target-face IDs, goal quaternion or orientation error.

## DAgger round 7

Round 7 targeted faces 2 and 3 with the best round-6 evidence policy. It added 15,155 labeled samples. The merged cache contains:

- 139,507 samples
- 2,128 episode groups
- 1,702 train episodes / 426 validation episodes
- cache SHA-256: `ab723f07c4d6b12607ac23aae1e7e392d5359cbb266526772fb0b78dc91ffbf0`

Three independent evidence seeds were trained for 40 epochs with the same episode split.

| Seed | Success | Drop |
|---:|---:|---:|
| 1 | 80/96 = 83.33% | 13.54% |
| 3 | 85/96 = 86.46% | 9.38% |
| 5 | 80/96 = 83.33% | 11.46% |

Mean success is **84.38%** with a standard deviation of **1.47 percentage points**. The seed-3 checkpoint is the current candidate:

- `outputs/visual_student_v3/checkpoints/dagger_target23_r7_evidence_seed3.pt`
- SHA-256: `6fe27ea3372669c7c9eced3deb47ffaa980a39ee90999cff9cd8cc87ca812717`

## Round 8 diagnostic

A follow-up targeted faces 1, 3 and 5. Seed-3 reached 88.54%, but the three seeds were 82.29%, 88.54% and 77.08% (mean 82.64%, standard deviation 4.67 points). This round is retained as diagnostic evidence and is not selected as the final candidate because it increased seed variance.

## Decision

Round 7 is the current **stable VisERDex-style Revo3 visual-language student candidate**. It is ready for paired protocol evaluation, video recording and later real-camera integration. Its mean success is below the frozen teacher's 90%+ stability gate, so the student should be reported as a strong stable candidate rather than as teacher-level final performance.
