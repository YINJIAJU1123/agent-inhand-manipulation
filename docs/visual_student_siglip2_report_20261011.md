# SigLIP2 + RGB-D Student v1

**Date:** 2026-10-11  
**Freeze:** `viserdex_revo3_visual_student_v1`  
**Purpose:** first controlled language-conditioned RGB-D student experiment.

## Frozen protocol

- Teacher/action contract: frozen Revo3 teacher, 21-D normalized action, `action_scale=1.0`.
- Student input: the existing 1060-D RGB-D semantic frontend plus proprioception and 8-step history.
- Student memory: evidence GRU; only the student GRU and action head are trained.
- Language encoder: frozen `google/siglip2-base-patch16-224`, 768-D normalized text embedding.
- Training split: the existing episode-level DAgger split, regenerated only with the new cache SHA; no sample or episode reshuffle.
- Data: 152,657 samples from the current DAgger aggregate.
- Instruction boundary: the historical aggregate did not contain raw instruction strings. This run reconstructs six canonical commands from `target_face`: `show the red marker`, `show the green marker`, `show the blue marker`, `show the yellow marker`, `show the magenta marker`, and `show the cyan marker`. This is a canonical language-conditioning experiment, not yet a paraphrase/open-vocabulary claim.

## Training and offline validation

| Seed | Validation MSE | Validation MAE | Out-of-bounds actions |
|---:|---:|---:|---:|
| 1 | 0.1750 | 0.2353 | 0 |
| 3 | 0.1797 | 0.2393 | 0 |
| 5 | 0.1676 | 0.2284 | 0 |

All checkpoints carry the frozen ID, action scale, `language_mode=vlm`, and the SigLIP2 language dimension. Seed5 is copied to `outputs/visual_student_v3/` as the first local checkpoint for follow-up evaluation.

## Closed-loop status

The live evaluator was launched with the same RGB-D grid (`16x16` RGB and `16x16` depth), `vision_stride=2`, `action_scale=1.0`, and SigLIP2 text path. Isaac Lab initialized successfully and loaded the model. The remote host was concurrently running several queue jobs and its camera/physics step became too slow; the 24-episode run and a 20-step smoke were stopped after confirming initialization but before a valid episode report was written. Therefore this document does **not** claim a SigLIP2 closed-loop success rate.

The next valid gate is a clean single-GPU live run on an otherwise idle device, followed by the fixed 2-target/24-episode protocol. Until that gate is complete, use the offline numbers only to verify the language-conditioned action regression path.
