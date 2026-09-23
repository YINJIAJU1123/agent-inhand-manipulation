# Visual-language student status

Updated: 2026-09-23

The frozen state-teacher source for this stage is `model_750.pt` from
`2026-09-22_09-31-56_viserdex_teacher_v5_scaled` (SHA-256
`d263d7c65f405d7eb385c782b61b91ebd14afd5be45e6608c8a97d027d3c69cd`).  The
camera collector uses the matching task
`BrainCo-Direct-Revo3-VisERDexTeacherVisual-Cube-v0`, preserves the teacher's
244-dimensional state interface, and fixes the language rollout yaw to zero so
the instruction names an observable face without also hiding an in-plane yaw.

## Data and checkpoints

Six fixed-face shards were collected on `brainco` with the frozen teacher,
`stride=2`, and `cuda:0`.  The merge contains 387 completed episodes and
10,880 aligned RGB-D/action/proprioception samples.  The merged data and all
student checkpoints are under
`/home/jiaju/src/RevoLab/outputs/viserdex_v5_visual_rollouts_fixedyaw/`.

The first offline baseline uses 12-dimensional global and center-crop RGB
statistics.  The second deterministic fallback uses 36 dimensions: the same
12 global values plus per-color-marker mass, image centroid, and spatial
spread for the six rendered face markers.  Both use a six-dimensional face
instruction vector.  The semantic-feature language student reached best
validation MSE 0.12969 (epoch 28); the matched no-language control reached
0.12358 (epoch 29).  These are behavior-cloning diagnostics, not success
metrics.

The intended SigLIP2 cache has not been produced yet: the remote host resets
the Hugging Face connection while fetching
`google/siglip2-base-patch16-224`.  The deterministic caches are therefore
kept as pipeline smoke baselines and must not be described as SigLIP results.

## Closed-loop diagnostics

All rows below use 24 episodes, four parallel environments, a 250-step
evaluator horizon, the matched visual task, and RGB refresh every two control
steps.  The evaluator reports terminal pose success and object drops; it is a
student diagnostic and is not the frozen teacher's repeated-success metric.

| Student input | Success | Drop |
| --- | ---: | ---: |
| 12-D RGB statistics + language | 4.17% | 4.17% |
| 12-D RGB statistics, language zeroed | 0% | 12.50% |
| 36-D marker-aware RGB + language | 0% | 8.33% |
| 36-D marker-aware RGB, language zeroed | 0% | 4.17% |

The marker-aware representation did not recover closed-loop success.  The
current limiting issue is behavior-cloning distribution shift and the bounded
action projection: 76.6% of raw teacher action elements in the merged replay
were outside `[-1,1]`, while the student contract clips actions to that range.
The student checkpoints are useful for interface and failure analysis, but
none is a final paper result and no PPO fine-tuning should be reported until
the action contract is re-aligned and a real frozen image encoder is cached.

Reports copied into `outputs/viserdex_revo3_cube/` include the rollout feature
summaries, training curves, oracle-feature diagnostics, and both live RGB
comparisons.
