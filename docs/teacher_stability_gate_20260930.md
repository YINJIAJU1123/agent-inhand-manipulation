# Teacher stability gate (2026-09-30)

## Frozen checkpoint and protocol

- Checkpoint: `model_1999.pt`
- SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Mode: state baseline v2, fixed goal yaw 0, hold evaluation
- Faces: 0–5
- Seeds: 0, 1, 2
- Quota: 16 episodes per face per seed, 288 episodes total

## Gate result

| Metric | Result |
| --- | ---: |
| Instant reach | 272/288 = 94.44% |
| Continuous hold | 272/288 = 94.44% |
| Held at end | 261/288 = **90.63%** |
| Drop | 14/288 = **4.86%** |

Held-at-end success by seed was 85.42%, 91.67%, and 94.79%. By target face it was 95.83%, 87.50%, 91.67%, 87.50%, 89.58%, and 91.67% for faces 0–5. No face collapsed and all isolated evaluation blocks completed.

This passes the working teacher gate for visual data generation: held-at-end success is above 90%, drop is below 5%, and per-face performance is bounded. The teacher is now stable enough to freeze for the visual student stage.

## Camera-task alignment fix

The first camera collection used a primitive cuboid while the frozen teacher was trained on the USD DexCube. The two objects had different collision/contact behavior. After changing `BrainCoHandVisualSemanticReorientEnvCfg.object_cfg` to reuse the exact state-teacher USD object configuration, a 32-episode camera smoke produced 32 successes and 1 drop, with zero rollout audit failures. Formal visual collection has now been regenerated with this aligned configuration: 382/390 episodes succeeded, with 8 drops and no timeouts or structural audit failures. Previous v2 visual-student data is retained only as a diagnostic run.

The per-face/seed gate report is stored in `outputs/teacher_gate_20260930/teacher_gate_hold_fixedyaw.json`.
