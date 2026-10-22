# Current local policy

Snapshot: 2026-10-22. Weights are byte-identical to the student used in the latest
Astra experiments. Repository cleanup does not select a new training candidate.

| Field | Value |
| --- | --- |
| Architecture | VisualLanguageStudent, evidence memory |
| History | 8 frames |
| RGB-D feature dimension | 1060 |
| Goal dimension | 768 |
| Proprioception dimension | 128 |
| Output dimension | 21 |
| Output range / external scale | `[-1, 1]` / `1.0` |
| Control / visual update rate | 30 Hz / 15 Hz |
| Frozen contract | `viserdex_revo3_visual_student_v1` |
| Student bytes | 4,060,601 |
| Student SHA256 | `28f62f3e1f8590cd57479a89aff5d36f054b6c49d19e544a7820d72d10f8d6f7` |

The RGB-D frontend is `scripts/agentic/student_features.py`. The environment
constructs measured proprioception. Ground-truth object pose, target quaternion,
layout, scoring flags and reward are excluded from the actor's input.

`configs/visual_student_freeze.json` retains the action contract and historical
teacher provenance. The teacher checkpoint is not needed for student inference.
DAgger collection requires a separately provided teacher and dataset.

Published companion assets:

- `siglip2_text_adapter.pt`: text-only classifier and canonical goal embeddings.
- `siglip2_text_cache.pt`: exact instruction embeddings for frozen templates.
- `waiting_controller.pt`: the V3 controller used in the current agent.
- `pose_verifier.pt`: frozen dense RGB-D conditional pose evidence.
- `semantic_reader.pt`: the current RGB sticker reader and semantic-memory input.

Their checksums and sizes are in `configs/runtime_assets.json`. The pose verifier
and semantic reader are experimental evidence modules. High confidence is not
ground-truth correctness and their integration has not established a robust
full-task success improvement.

The student is VisERDex-style teacher-to-visual-student control. It is not an
upstream VisERDex checkpoint. This release covers simulation with the provided
cube/hand assets. Real-hand performance has not been established by these tests.
