# Visual student run report (2026-09-26)

## Frozen contract

- Freeze ID: `viserdex_revo3_visual_student_v1`
- Teacher: `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0`
- Checkpoint: `/home/jiaju/src/RevoLab/logs/rsl_rl/brainco_hand/2026-09-21_10-35-09_revo3_teacher_hold02_s21/model_1999.pt`
- Checkpoint SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Camera task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Camera: RGB-D, 256x256, control stride 2, action scale 6.0
- Live evaluation: 72 episodes per checkpoint, 4 environments, horizon 600

The camera task keeps the teacher's 158-D state observation. The earlier 244-D VisERDex task was rejected because it cannot load the frozen `model_1999.pt`; the runnable defaults and manifest now use the compatible camera semantic task.

## Run evidence

- Camera smoke: passed; RGB shape `[256, 256, 3]`, valid depth fraction `0.6452`.
- Full collection: 388 completed episodes, 10,176 aligned samples, stride 2.
- Feature cache: 36-D deterministic semantic RGB marker statistics plus 6-D instruction one-hot.
- Trained checkpoints: plain/evidence memory modes, seeds 0/1/2.

## Offline validation

| Student | Seed | MSE | MAE |
| --- | ---: | ---: | ---: |
| plain | 0 | 0.13979 | 0.20841 |
| plain | 1 | 0.13755 | 0.20754 |
| plain | 2 | 0.14070 | 0.21203 |
| evidence | 0 | 0.14290 | 0.21262 |
| evidence | 1 | 0.14082 | 0.20986 |
| evidence | 2 | 0.14240 | 0.21490 |

## Closed-loop evaluation

| Student | Seed | Episodes | Terminal success | Drop |
| --- | ---: | ---: | ---: | ---: |
| plain | 0 | 72 | 1.39% | 22.22% |
| plain | 1 | 72 | 1.39% | 19.44% |
| plain | 2 | 72 | 1.39% | 25.00% |
| evidence | 0 | 72 | 4.17% | 20.83% |
| evidence | 1 | 72 | 1.39% | 13.89% |
| evidence | 2 | 72 | 2.78% | 19.44% |

These are deterministic semantic-camera smoke baselines, not a paper result and not a frozen SigLIP result. The next paper-facing step is replacing the fallback feature cache with a versioned frozen image encoder, then repeating the same live protocol.

Remote artifacts remain under `/home/jiaju/src/RevoLab/outputs/visual_student_v1/full/`; small checkpoints and JSON reports were copied to `outputs/visual_student_v1/` locally. The multi-gigabyte rollout shard stays on the GPU host.
