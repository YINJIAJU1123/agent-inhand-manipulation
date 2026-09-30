# Long-horizon visual student evaluation (2026-10-01)

## Protocol

The long-horizon check uses the frozen visual protocol:

- Student task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Teacher task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Teacher checkpoint SHA-256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- RGB-D semantic frontend: 16x16 RGB grid + 16x16 normalized depth grid + six marker slots
- Vision stride: 2
- Action scale: 1.0
- Student evaluation: 192 episodes, four environments, 600-step horizon
- Teacher comparison: 96 episodes, four environments, seed 123

The reset seed and episode budget are aligned for comparison. The teacher and student still use separate task observation interfaces, so this is a same-seed paired-style control check rather than an exact trajectory-by-trajectory pairing.

## Round-8 evidence student

Round 8 added targeted DAgger data for faces 1, 3 and 5. The seed-3 checkpoint is the best long-horizon candidate:

- Checkpoint: `outputs/visual_student_v3/checkpoints/dagger_target135_r8_evidence_seed3.pt`
- SHA-256: `9aaa9d54c1a0a69d31a0dbdac51552bb466b090d5659f79e83ceebbc0e0fe732`

| Training seed | Episodes | Success | Drop |
|---:|---:|---:|---:|
| 1 | 192 | 80.73% | 17.71% |
| 3 | 192 | **85.94%** | **10.42%** |
| 5 | 192 | 82.29% | 14.58% |

Mean success is **82.99%** with a standard deviation of approximately **2.6 percentage points**. Seed 3 is the current long-horizon deployment candidate.

Seed-3 per-face success:

- face 0: 88.89%
- face 1: 85.71%
- face 2: 82.86%
- face 3: 87.10%
- face 4: 95.45%
- face 5: 78.79%

## Teacher comparison

The frozen state teacher reached **98.96% success** and **1.04% drop** on 96 episodes with seed 123. The visual student remains below the teacher gate, but the long-horizon seed spread is now small enough to support a reproducible student baseline and real-camera integration work.

## Decision

Use round-8 seed-3 as the current VisERDex-style Revo3 visual-language student candidate for videos, deployment wiring and real-camera tests. Report the 192-episode mean and per-face results in experiments; keep the teacher result as the privileged upper-bound reference.
