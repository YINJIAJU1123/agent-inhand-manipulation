# Visual student data stability report (2026-09-29)

## Frozen contract

- Freeze ID: `viserdex_revo3_visual_student_v1`
- Teacher task: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Teacher checkpoint: `model_1999.pt`
- Teacher checkpoint SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Camera task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- RGB-D: 256x256, capture stride 2
- Action dimension: 21, bounded student output, action scale 6.0
- Live protocol: 72 episodes, 4 environments, horizon 600

## Collection and audit

The corrected collector was run on the free physical GPU 2, remapped to logical `cuda:0` for Isaac Sim. The raw shard contains 388 complete quality records for 387 requested episodes (the extra record is the final vectorized slot), 450 observed vector episodes, 159 captured batches, and 10,176 samples.

| Check | Raw result | Filtered result |
| --- | ---: | ---: |
| Batch shape mismatches | 0 | 0 |
| Duplicate sample keys | 0 | 0 |
| Non-contiguous episode steps | 0 | 0 |
| Stride mismatches | 0 | 0 |
| Complete episodes with terminal count != 1 | 0 | 0 |
| Low RGB variance frames | 0 | 0 |
| Low valid-depth frames | 0 | 0 |
| Bounded student actions | 100% | 100% |

The raw quality records contain 301 successful episodes, 86 drops, and 1 timeout. The `success_only` quality policy selected 301 complete episodes and 7,724 samples. The filtered shard has 301/301 success records, no in-progress episodes, and passes the same audit with zero structural failures.

Raw shard SHA256: `a6270ab62b9ee2db09e404f4da876d54af9a3454690f6f8fe6ec8733d9b886af`  
Filtered shard SHA256: `ff342312834edfd7dfa79d24826c45c62da36aa2c90f6c38facdb1fbc489e283`  
Feature cache SHA256: `5f29b9abee201fa054cb9bb2822ad022d4485748d5fbf9187e42bf1dfa9680e2`

The deterministic episode split contains 241 train episodes and 60 validation episodes. All six training runs used this same split, freeze manifest, checkpoint, action scale, history length 8, and 20 epochs.

## Training and live evaluation

Final validation MSE was 0.1493–0.1517 across the six runs. Every checkpoint completed the 72-episode closed-loop protocol and emitted a complete report.

| Student | Seed | Validation MSE | Live success | Live drop |
| --- | ---: | ---: | ---: | ---: |
| plain | 0 | 0.1504 | 1.39% (1/72) | 25.00% |
| plain | 1 | 0.1514 | 1.39% (1/72) | 26.39% |
| plain | 2 | 0.1493 | 1.39% (1/72) | 19.44% |
| evidence | 0 | 0.1517 | 1.39% (1/72) | 27.78% |
| evidence | 1 | 0.1511 | 1.39% (1/72) | 36.11% |
| evidence | 2 | 0.1486 | 1.39% (1/72) | 27.78% |

## Interpretation

The collection, filtering, split, training, and live-evaluation pipeline is now reproducible and stable: no episode alignment, camera, terminal, batch, or process failures were observed, and all six frozen-protocol evals finished. The closed-loop task success remains low and seed-invariant at 1/72, so this checkpoint is a stable engineering baseline rather than a CVPR result. The next performance iteration should replace the deterministic 36-D marker statistics with a versioned frozen image encoder and retain this audit/split/evaluation contract unchanged.

Compact reports and checkpoints are stored under `outputs/visual_student_v2/`; the multi-gigabyte raw rollout shard remains on the GPU host.
