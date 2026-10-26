# Agentic In-hand Reorientation

Language-guided Revo3 in-hand reorientation in Isaac Lab. This snapshot contains
our VisERDex-style RGB-D reference policy, its VLM/Astra supervisors, and the
fixed action-adapted student used in the current paper. It is a research implementation built on BrainCo RevoLab, not the
official VisERDex release.

## Start here

| Component | Entry point |
| --- | --- |
| RGB-D student and text goals | `scripts/run_student.py` |
| Reference Astra runtime (22 October snapshot) | `scripts/run_astra.py` |
| Fine-tuned VLM latent baseline | `scripts/run_vlm.py` |
| VLM inference service | `scripts/agentic/latent_service_v7.py` |
| Astra request bridge | `scripts/agentic/astra_bridge_v6.py` |
| Paper action-only component | `scripts/agentic/adapted_actor.py` |
| Check the adapted action interface | `scripts/check_adapted_actor.py` |
| Check all published weights | `scripts/check_runtime.py` |

The local policy reads RGB-D features, an eight-frame observation history,
robot proprioception, and a 768-dimensional goal representation. It outputs
21 normalized joint commands in `[-1, 1]`, with action scale **1.0**.
Control runs at 30 Hz and the student camera pipeline at 15 Hz.

Astra outputs structured inspection, execution, verification and stop commands.
Its selected color is mapped to the student's frozen goal embedding. The
fine-tuned small VLM instead predicts a continuous goal embedding. Neither
supervisor receives the simulator's hidden target or object pose.

The default simulation wrappers retain the 22 October reference runtime. The
26 October paper action component is supplied separately and does not replace
those defaults. See [paper integration](docs/paper_action_adaptation.md) for its
control boundary and reproduction scope.

## Setup

Use an existing Isaac Lab Python environment with camera support. Install the
local extension from the repository root:

```bash
python -m pip install -e source/BrainCo_DexHand
python -m pip install -r requirements-runtime.txt
python scripts/check_runtime.py
```

PyTorch and Isaac Sim should come from your Isaac Lab installation. The lightweight
check above runs on CPU and makes no network requests. Set `ISAACLAB_PYTHON` if
launching the wrappers from a different Python environment.

### Student

The published text cache supports the frozen evaluation instruction templates,
so this example does not download SigLIP2:

```bash
python scripts/run_student.py \
  --episodes 12 --num_envs 4 --seed 101 --max-steps 600 \
  --report outputs/student/report.json
```

For new free-form text, use the text encoder and classifier in
`scripts/vision/text_grounding.py`. The exact-text cache deliberately rejects
unknown strings.

### Reference Astra runtime

First check the complete simulation path with scripted responses:

```bash
python scripts/run_astra.py --offline-smoke --max-steps 60 \
  --output outputs/astra_smoke
```

For real inference, configure the loopback bridge using the
[supervisor guide](docs/supervisors.md), then run:

```bash
python scripts/run_astra.py --seed 110101 --max-steps 3600 \
  --record-video --output outputs/astra_live
```

Physics keeps advancing while requests are pending. The current configuration
uses fixed-green startup, the frozen waiting controller, RGB semantic memory,
and a visual pose verifier. These are development components, not a guarantee
of success. Output directories include reports, action traces and optional videos.

### Fine-tuned VLM

Start `latent_service_v7.py` with a separately supplied Qwen goal-head and LoRA
checkpoint, then run `scripts/run_vlm.py --output outputs/vlm_eval`. See
[supervisors](docs/supervisors.md) for the service command and checkpoint layout.
The large Qwen backbone and VLM adapter are not bundled with this student release.

## Weights and results

Two student files have distinct roles:

- [`viserdex_rgbd_student.pt`](checkpoints/student/viserdex_rgbd_student.pt) is the frozen reference, also retained for feedback and stopping.
- [`adapted_action_student.pt`](checkpoints/student/adapted_action_student.pt) is the paper's mixed-data actor, SHA256 beginning `b02d8083`. Only its action output is consumed.

All seven inference assets have exact SHA256 values in
[`configs/runtime_assets.json`](configs/runtime_assets.json).
See the [model card](docs/model_card.md) and [action adaptation](docs/paper_action_adaptation.md).

The complete paper development study gives Astra reference/adapted **21/36 and
28/36** completions. On prospective new resets, the raw counts are **27/36 and
28/36**, while both are **27/35** in the predeclared matched analysis. The
adaptation gain did not replicate in that second comparison. These results use
the later research evaluator, not this snapshot's default simulation wrappers.
The [paper repository](https://github.com/YINJIAJU1123/CVPR-AgenticRotation)
contains the full cohorts, uncertainty and failure analysis. The historical
4/12 results remain identified in [status](docs/status.md).

## Layout

- `source/BrainCo_DexHand`: local policy and Revo3 cube environments.
- `scripts/vision`: student training, DAgger collection and language routing.
- `scripts/agentic`: supervisor models, transports and asynchronous execution.
- `scripts/astra_*`, `scripts/semantic_completion_*`: dependencies of the current agent.
- `checkpoints`: current inference assets only.
- `deploy/revo3`: hardware deployment scaffolding, separate from validated simulation.

HORA, bimanual handover, unrelated robot assets, superseded experiment runners
and old checkpoints have been removed from this snapshot. Historical research
code remains in Git history. Credentials, local datasets, reports and videos
are not tracked.

## Attribution

Based on [BrainCo RevoLab](https://github.com/BrainCoTech/RevoLab) and Isaac Lab.
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
