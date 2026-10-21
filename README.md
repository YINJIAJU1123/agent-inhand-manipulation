# Agent In-hand Manipulation

Research fork of [BrainCoTech/RevoLab](https://github.com/BrainCoTech/RevoLab)
for goal-conditioned Revo3 reorientation and visual-language control.
Upstream assets and license notices are retained.

## Research status — 2026-10-22

The [current Astra retention study](docs/astra_retention_20261022.md) tracks
completion, observation stability, and learned action selection. The latest
matched live-agent diagnostic completed **7/18 tasks (38.9%)** with corrected
collision-clearance initialization, versus 5/18 with the original initialization.
Success requires correct semantic binding, full orientation error at most
0.16 radians, no drop, and a correct autonomous stop. Physics continues during
Astra inference. These are development cases, not evidence of 80% success.

Twelve recorded sensor-triggered states completed three six-second physical
interventions each. Mean time inside the selected-color pose tolerance was
87.1% for holding, 48.8% for continued student execution, and 35.8% for slowing.
All intervention states matched exactly. These local pose-retention measurements
include two incorrect semantic bindings and are not full-agent success rates.
A separate eight-state observation experiment completed 24 matched branches.
Holding retained the selected observation pose for 70.6% of six seconds, versus
34.9% for continued execution. This also measures local pose retention only.
The first cloud action-value candidate failed its held-out checks and was not
deployed. Local control-data collection continues, and a native RGB sticker
localization and symbol-reading pipeline is collecting data on one cloud GPU.
Neither learned candidate has been validated in live Astra tasks.
See the [results overview](docs/experiment_results_overview_20261021.md)
for earlier protocols and their separate denominators.

### Earlier asynchronous comparison and completion learning

The earlier [training and evaluation record](docs/training_evaluation_status_20261019.md)
separates complete semantic-agent tasks from known-goal controller diagnostics.
In the earlier 192-episode live Astra comparison, continued execution achieved
27/48 successes (56.25%), fixed slowing 25/48 (52.08%), the V3 waiting controller
25/48 (52.08%), and the newer temporal controller 22/48 (45.83%). Physics advanced
while model requests were pending. These are development results on two repeated
seeds. The newer controller has not demonstrated an overall success advantage.

The [completion evidence study](docs/semantic_completion_ablation_20261020.md)
adds goal-conditioned completion learning, first-stop diagnostics and a dense
96-by-96 RGB-D branch while keeping the visual student frozen. The latest
collection contains 14,062 training frames from 192 trajectories and 3,717
validation frames from 48 trajectories. Both matched training runs completed.
On the same 24 recorded audit trajectories, compact features gave 12 correct
first-stop proposals, 4 incorrect proposals and 8 cases without a proposal.
Dense RGB-D gave 18 correct, 3 incorrect and 3 without a proposal. Its 18/21
proposal precision (85.7%) remains below the predefined 90% admission criterion.
This is conditional completion evidence, not a new full-agent success rate.
No completion candidate has replaced the baseline.

The proposals did not change the recorded student's actions. Of the 18 correct
dense proposals, 9 trajectories were still inside the pose tolerance one second
later under continued execution. This is a pose-retention diagnostic, not a
drop rate or a measured effect of a hold controller. The current retention study above measures interventions and prepares learned
coordination; an online success improvement remains unverified.

Code entry points:

- `scripts/astra_retention_v1/run_learning_pipeline.py`: independent counterfactual data and action-value training.
- `scripts/astra_retention_online_v1/`: request-scoped integration and local physics probes.
- `scripts/astra_retention_audit_v1/`: audit of states that meet the actual runtime activation condition.
- `infra/astra_retention_cloud_20261022/`: bounded cloud collection and training.

- `scripts/semantic_completion_v3/collect_run.py`: bounded dense RGB-D collection.
- `scripts/semantic_completion_v3/train_run.py`: frozen-feature caching and matched compact/dense training.
- `scripts/semantic_completion_v2/run_comparison.py`: shared-trajectory diagnostics for the earlier V2 completion heads.
- `scripts/astra_repeated_eval_v1/`: repeated live Astra evaluation and analysis.
- `scripts/recovery_controller_v3/` and `scripts/recovery_factorial/`: waiting/recovery experiments.

This source snapshot includes experiment protocols and tests. Local datasets,
trained weights, raw reports and videos remain under the ignored `outputs/`
directory. Running the experiments requires those assets and a configured Isaac
Lab environment. Historical runners include machine-specific paths, including
the Python executable in `scripts/semantic_completion_v1/common.py`; configure
them for the target machine before launching. The 0.16-radian full-quaternion
completion criterion and the frozen policy contracts are unchanged.

## Earlier research milestones

Language baseline (2026-10-14): a frozen RGB-D student is driven by a trained
six-color goal classifier on frozen SigLIP2 text features. On two evaluation
seeds, held-out English instruction templates achieve 600/768 closed-loop
successes (78.13%). See the [text-grounding v2 report](docs/visual_student_language_v2_report_20261014.md)
for the protocol, controls, per-episode evidence, and limits of this result.
The earlier [v1 report](docs/visual_student_language_v1_report_20261014.md) has been
corrected to identify its rule-based color routing.

A separate [VLM agent symbol-task pilot](docs/agentic_symbol_results_20261014.md)
now tests active inspection, evidence memory, and event-triggered supervision.
Across 48 held-out pilot episodes, periodic re-observation succeeds in 6/12,
once-only grounding in 2/12, and the event/memory variant in 2/12 (oracle: 11/12).
The added agent mechanisms have not outperformed the periodic baseline; visual
verification remains unreliable. These are new-task results, not the language
baseline above. See the [method and reproduction guide](docs/agentic_symbol_method_20261014.md).

The [900-episode frozen evaluation](docs/agentic_symbol_large_eval_20261014.md)
is complete: periodic inspection succeeds in 88/300 (29.33%), once-only in
14/300 (4.67%), and oracle in 249/300 (83.00%). The VLM inventories visible
symbols and code selects the fixed star target; these results do not establish
multi-instruction language grounding.

The separate [semantic agent v3](docs/agentic_semantic_v3_20261014.md) introduces
actual VLM language parsing for six symbol targets, visual grounding, active
inspection, and execution feedback. The first 54-episode validation gives
4/18 successes for feedback, 4/18 for periodic observation, and 15/18 for
oracle. Feedback uses 66 versus 86 VLM calls but has not improved success;
visual symbol-to-sticker binding remains the main observed bottleneck.
These results are separate from the frozen star benchmark.

The [two-track v4 prototype](docs/agentic_two_track_v4_20261014.md) now adds
Qwen3-VL-4B QLoRA with a directly executed continuous goal latent and an Astra
tool supervisor. The small development test completed on 2026-10-15 gives
4/12 latent successes versus 12/12 paired oracle successes. Astra's interface
was initially awaiting credentials; authorized real image/tool calls now work
through the [v6 subscription evaluation](docs/astra_subscription_eval_v6_20261015.md). This version is
not yet a stable multi-symbol agent or a demonstrated improvement over v3.

A [900-episode paired comparison](docs/agentic_baselines_v5_large_20261015.md)
was launched on 2026-10-15: the symbolic/rule baseline, learned latent baseline,
and oracle each receive 300 episodes across 25 new seeds and two instruction
forms. Results are pending; this is a system comparison, not an isolated
finetuning ablation.

The [Astra v6 evaluation](docs/astra_subscription_eval_v6_20261015.md) adds
60 closed-loop episodes on a remote RTX 5090 using app-specific ChatGPT OAuth,
two concurrent model requests, and six calls per episode. The finite runner
saves request usage, videos, and report files under
`outputs/agentic_v4/astra_eval_20261015`; simulated hand rollouts and image/tool
requests completed with 31/60 successes. Those rollouts paused physics during inference.

The [lightweight waiting controller v1](docs/latency_controller_v1_20261015.md)
has now been trained with continuous physics during simulated verification delays.
On 46 shared-request-state cases it recovers successfully in 34 cases (73.9%),
versus 32 for continued student execution and 31 for fixed slowing/holding.
This is a small pilot, mostly learning slowdown; it is not yet a live asynchronous
Astra controller or evidence of a statistically established improvement.

The [live asynchronous v7 evaluation](docs/async_agent_eval_v7_20261016.md)
compares Astra and the fine-tuned continuous-latent VLM, each with continued
execution, fixed slowing, or the learned waiting controller. The registered
large protocol contains 720 episodes with physics advancing during requests;
read `outputs/async_v7/` run reports for completion and results.
The initial remote batch stopped after HTTPS transport errors. The current
local RTX 3090 run is `outputs/async_v7/local_20261016`, managed by
`run_local_async_v7.py` with independent backend failure handling.

The [goal-conditioned waiting and recovery pilot v2](docs/recovery_controller_v2_20261018.md)
trains continuous residual actions through both the reasoning wait and a two-second
rejoin, with a task-completion objective and a matched waiting-survival ablation.
The finite local pilot includes two evaluation seeds, strong fixed smooth controls,
432 diagnostic evaluation trials, and explicit zero-delay variability checks.
Both 32-update training runs and all 432 evaluation trials completed. On the
46 eligible positive-delay cases per method, completion learning achieved
30/46 (65.2%), equal to survival learning and below fixed slow at 34/46
(73.9%). Large zero-delay repeat variation persists; learned superiority
has not been demonstrated. See the linked completion review for details.

The [V3 variance diagnosis and optimization](docs/recovery_controller_v3_20261018.md)
completed all 32 training updates and 432 evaluation trials. Recorded-action
playback reproduced all 300 physical steps exactly, while new RGB-D observations
caused closed-loop actions to diverge. V3 separates policy/value/outcome learning,
limits policy KL with rollback, and initializes at fixed slow with immediate
student recovery. V3 achieved 65/92 (70.7%) eligible positive-delay successes,
versus fixed slow at 62/92 (67.4%). The small advantage reverses on the second
seed and repeat, and 20-second waits remain weak. Reliable superiority is not
yet established. Twenty-four videos were saved; live Astra/VLM calls remain
outside this diagnostic.

The [RQ2 input-dependency experiment and hardware preparation plan](docs/recovery_rq2_and_transfer_20261018.md)
defines four paper questions and starts a finite five-condition comparison on two
new evaluation seeds. It probes V3's direct visual/goal dependence while keeping
the frozen student inputs correct, with same-state action sensitivity probes and
528 planned evaluation trials. Results are pending. The hardware plan explicitly
accounts for the current RGB-D frontend, contact dynamics and sensor calibration.

Earlier project status is recorded in [progress_20260926.md](docs/progress_20260926.md).
Camera setup and environment reproduction are documented in
[camera validation](docs/camera_placement.md) and [reproduction instructions](docs/reproducibility.md).

The frozen visual-student contract is recorded in [visual_student_protocol_v1.md](docs/visual_student_protocol_v1.md), with the machine-readable manifest at [configs/visual_student_freeze.json](configs/visual_student_freeze.json).

The semantic goal reset was corrected on 2026-09-12: target tokens now clear
previous face bits, and side-face rotations map the named face to world +Z.
Earlier evaluation reports belong to the previous goal implementation and must
be re-evaluated before using its checkpoint for a labeled visual dataset.

Isaac Lab environments, robot assets, and pretrained checkpoints for BrainCo dexterous manipulation tasks. Including:

- Revo3 in-hand repose
- Revo3 in-hand reorientation
- Revo3 HORA/RMA in-hand ball and cylinder rotation
- Revo3 right-hand lift
- RevoTron dynamic handover

## Included Tasks

| Robot | Framework | Public task ID | Checkpoint | Demo |
| --- | --- | --- | --- | --- |
| Revo3 | Direct | `BrainCo-Direct-Revo3-Repose-Cube-v0` | `checkpoints/BrainCo-Direct-Revo3-Repose-Cube-v0.pt` | <img src="image/BrainCo-Direct-Revo3-Repose-Cube-v0.gif" width="320"/> |
| Revo3 | Direct | `BrainCo-Direct-Revo3-SemanticReorient-Cube-v0` | trained locally | goal-face-conditioned teacher for language/vision student |
| Revo3 | Direct | `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0` | data-collection config | fixed RGB-D camera; use `capture_camera()` for visual student data |
| Revo3 | Direct | `BrainCo-Direct-Revo3-Reorient-Cylinder-v0` | `checkpoints/BrainCo-Direct-Revo3-Reorient-Cylinder-v0.pt` | <img src="image/BrainCo-Direct-Revo3-Reorient-Cylinder-v0.gif" width="320"/> |
| Revo3 | Dexsuite | `BrainCo-Dexsuite-Revo3-Right-Lift-v0` | `checkpoints/BrainCo-Dexsuite-Revo3-Right-Lift-v0.pt` | <img src="image/BrainCo-Dexsuite-Revo3-Right-Lift-v0.gif" width="320"/> |
| Revo3 | HORA | `BrainCo-Direct-Revo3-HoraRotate-Ball-v0` | `checkpoints/hora/revo3_right_ball_stage1_best.pth` | <img src="image/BrainCo-Direct-Revo3-HoraRotate-Ball-v0.gif" width="320"/> |
| Revo3 | HORA | `BrainCo-Direct-Revo3-HoraRotate-Cylinder-v0` | `checkpoints/hora/revo3_right_cylinder_stage1_best.pth` | <img src="image/BrainCo-Direct-Revo3-HoraRotate-Cylinder-v0.gif" width="320"/> |
| RevoTron | Dynamic Handover | `BrainCo-Dynamic-Handover-Revo3-Cube-v0` | `checkpoints/dynamic_handover/revotron_handover.pth` | <img src="image/BrainCo-Dynamic-Handover-Revo3-Cube-v0.gif" width="320"/> |

## Repository Layout

```text
BrainCo-IsaacLab/
├── assets/                 # Robot assets, object meshes, and grasp caches
│   ├── usd/                # USD assets grouped by task family
│   └── urdf/               # URDF assets and meshes
├── checkpoints/            # Pretrained checkpoints
├── image/                  # Task demo GIFs
├── scripts/rsl_rl          # RSL-RL training and evaluation scripts
├── scripts/rl_games        # RL-Games training and evaluation scripts
├── scripts/hora            # HORA PPO / ProprioAdapt training and export scripts
├── deploy/revo3            # Revo3 sim-to-real deployment package
└── source/BrainCo_DexHand/ # Isaac Lab extension package
```

## Requirements

- Python 3.10+
- A working [Isaac Lab](https://github.com/isaac-sim/IsaacLab) installation

This repository is distributed as an Isaac Lab extension package, not as a standalone simulator fork.

## Reproducible Docker image

A validated amd64 image (Isaac Sim 5.1.0, Isaac Lab 2.3.2 and this source
overlay) is available at
`yinjiaju00/vlmrotation:20260913-3041a21`. The repository is currently private;
authenticate first, then run the RGB-D smoke test with a GPU:

```bash
docker login
docker pull yinjiaju00/vlmrotation:20260913-3041a21
docker run --rm --gpus all yinjiaju00/vlmrotation:20260913-3041a21 \
  scripts/rsl_rl/smoke_visual_camera.py --headless --output /tmp/vlmrotation-smoke
```

The image digest and build pins are recorded in
[docs/reproducibility.md](docs/reproducibility.md).

## Installation

1. Install Isaac Lab by following the official [Isaac Lab](https://github.com/isaac-sim/IsaacLab) setup for your target version.
2. Clone this repository into your workspace.
3. Install the BrainCo extension:

```bash
cd source/BrainCo_DexHand
pip install -e .
```

## Downloading Checkpoints

We provide a script to easily download all the pretrained checkpoints from our OSS server. Run the following command from the repository root:

```bash
./scripts/download-checkpoints.sh
```

This will download the `*.pt` files directly into the repository's `checkpoints/` directory.

## In-hand Manipulation

The in-hand manipulation environments use RSL-RL. The environments register when `BrainCo_DexHand` is importable in your Isaac Lab Python environment.

Check registration:

```bash
python -c "import BrainCo_DexHand"
```

Train:

```bash
python  scripts/rsl_rl/train.py --task BrainCo-Direct-Revo3-Repose-Cube-v0 --num_envs 8192 --headless
python  scripts/rsl_rl/train.py --task BrainCo-Direct-Revo3-Reorient-Cylinder-v0 --num_envs 4096 --headless

# Language-conditioned privileged teacher (structured goal; preserves 21-DoF action)
./scripts/rsl_rl/train_language_teacher.sh

# Collect RGB-D trajectories with a semantic state-teacher checkpoint
python scripts/rsl_rl/collect_visual_rollouts.py \
  --task BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0 \
  --checkpoint logs/rsl_rl/semantic_reorient/<run>/model_250.pt \
  --episodes 100 --num_envs 64 --headless --output data/visual_rollouts.pt
```

The teacher's language contract is deliberately small at this stage: the six
face ids are encoded as a one-hot goal vector and rendered into instructions
such as `show the red marker`.  This keeps the RL problem stable while making
the goal interface identical for the later CLIP/SigLIP visual student.  The
contract and templates live in
`source/BrainCo_DexHand/BrainCo_DexHand/algo/agentic/language_goal.py`.

The visual task uses an Isaac primitive cube and a fixed 256x256 RGB-D
`TiledCamera` at `/World/envs/env_*/SemanticCamera`.  The camera is an elevated
front eye-to-hand view at `(0, -0.72, 0.95)` m, tilted down by 32.59 degrees;
the exact rationale and field of view are documented in
[`docs/camera_placement.md`](docs/camera_placement.md).  The sensor is
intentionally kept out of the default PPO observation so existing state-teacher
checkpoints remain compatible; custom collectors should call
`env.capture_camera()` after each step and fuse the frames with a
vision-language encoder. Isaac Lab's app launcher must be started with
`--enable_cameras` for RGB/depth rendering.

Evaluate:

```bash
python  scripts/rsl_rl/play.py --task BrainCo-Direct-Revo3-Repose-Cube-v0 --checkpoint checkpoints/BrainCo-Direct-Revo3-Repose-Cube-v0.pt --num_envs 1
python  scripts/rsl_rl/play.py --task BrainCo-Direct-Revo3-Reorient-Cylinder-v0 --checkpoint checkpoints/BrainCo-Direct-Revo3-Reorient-Cylinder-v0.pt --num_envs 1

# Per-face semantic evaluation (writes a JSON report)
python scripts/rsl_rl/evaluate_semantic_reorient.py \
  --task BrainCo-Direct-Revo3-SemanticReorient-Cube-v0 \
  --checkpoint logs/rsl_rl/semantic_reorient/<run>/model_250.pt \
  --num_envs 256 --episodes 100 --headless --report eval_semantic.json
```

## Grasping

The grasping task uses the Dexsuite Revo3 environment with RSL-RL.

Train:

```bash
python  scripts/rsl_rl/train.py --task BrainCo-Dexsuite-Revo3-Right-Lift-v0 --num_envs 4096 --headless
```

Evaluate:

```bash
python  scripts/rsl_rl/play.py --task BrainCo-Dexsuite-Revo3-Right-Lift-Play-v0 --checkpoint checkpoints/BrainCo-Dexsuite-Revo3-Right-Lift-v0.pt --num_envs 1
```

## HORA Training and Export

The HORA rotation tasks are integrated as a separate training path because their PPO, ProprioAdapt Stage 2 adapter, and checkpoint format are not RSL-RL-compatible. This integration is adapted from [HORA](https://github.com/HaozhiQi/hora).

Stage 1 PPO teacher training:

```bash
python scripts/hora/train.py --algo PPO --task ball --num_envs 16384 --headless
python scripts/hora/train.py --algo PPO --task cylinder --num_envs 16384 --headless
```

Stage 2 ProprioAdapt training from a Stage 1 checkpoint:

```bash
python scripts/hora/train.py --algo ProprioAdapt --checkpoint outputs/hora/revo3_right/run_ball/stage1_nn/best.pth --task ball --num_envs 16384 --headless
python scripts/hora/train.py --algo ProprioAdapt --checkpoint outputs/hora/revo3_right/run_cylinder/stage1_nn/best.pth --task cylinder --num_envs 16384 --headless
```

Evaluate a checkpoint:

```bash
python scripts/hora/train.py --algo PPO --task ball --num_envs 32 --test --checkpoint checkpoints/hora/revo3_right_ball_stage1_best.pth
python scripts/hora/train.py --algo PPO --task cylinder --num_envs 32 --test --checkpoint checkpoints/hora/revo3_right_cylinder_stage1_best.pth
```

Export a Stage 2 checkpoint to ONNX:

```bash
python scripts/hora/export_onnx.py \
  --stage stage2 \
  --checkpoint outputs/hora/revo3_right/run_ball/stage2_nn/model_best.ckpt \
  --output ball_hora_stage2.onnx
```

## Dynamic Handover

Dynamic handover uses the RevoTron asset and RL-Games. This task is adapted from [dynamic_handover](https://github.com/cypypccpy/dynamic_handover). Run from the repository root with the Isaac Lab Python environment activated.

Train:

```bash
python scripts/rl_games/train.py \
  --task BrainCo-Dynamic-Handover-Revo3-Cube-v0 \
  --num_envs 4096 \
  --headless
```

Evaluate:

```bash
python scripts/rl_games/play.py \
  --task BrainCo-Dynamic-Handover-Revo3-Cube-Play-v0 \
  --num_envs 1 \
  --checkpoint checkpoints/dynamic_handover/revotron_handover.pth
```

To manually switch toss direction during play, pass a command file:

```bash
python scripts/rl_games/play.py \
  --task BrainCo-Dynamic-Handover-Revo3-Cube-Play-v0 \
  --num_envs 1 \
  --checkpoint checkpoints/dynamic_handover/revotron_handover.pth \
  --command-file /tmp/handover_command.txt
```

Then update the command from another terminal:

```bash
echo 'right_throw' > /tmp/handover_command.txt
echo 'left_throw' > /tmp/handover_command.txt
```

## Sim-to-real Deployment

Below is a sim-to-real deployment of `BrainCo-Direct-Revo3-HoraRotate-Cylinder-v0`. More tasks' sim-to-real deployments will be released in the future.

<p align="center">
  <img src="image/sim2real.gif" alt="Revo3 sim-to-real deployment demo" width="360"/>
</p>

The Revo3 deployment runtime lives in `deploy/revo3`. It is a lightweight, ROS-free package for running exported ONNX policies on the real Revo3 hand through the Revo3 Python SDK.

The deploy loop is closed-loop: it reads measured joint positions from hardware, builds the policy observation history, runs ONNX inference, maps policy-order targets to SDK motor order, and sends MIT commands back to the hand. It is not a fixed trajectory replay tool.

Install the deploy package:

```bash
cd deploy/revo3
pip install -e .
```

Install the hardware extra when running on the real hand:

```bash
pip install -e ".[hardware]"
```

The runtime uses:

- `policy.onnx`: exported policy network
- `policy.yaml`: policy I/O contract and metadata
- `config/revo3_right.yaml`: joint order, joint limits, SDK settings, MIT gains, and sim-to-real offsets

Run a dry-run policy loop without sending commands:

```bash
python scripts/run_policy.py \
  --onnx artifacts/repose_cube/policy.onnx \
  --policy artifacts/repose_cube/policy.yaml \
  --profile config/revo3_right.yaml \
  --dry-run
```

See `deploy/revo3/README.md` for export and hardware run commands.

## Notes

- The Revo3 tasks IDs follow the `BrainCo-<framework>-<robot>-<task>-v0` naming convention.
- Checkpoints are provided for reproducibility and evaluation.
- Dynamic handover play supports command-file throw direction overrides: `right_throw` and `left_throw`.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).

Some files retain upstream Isaac Lab copyright and SPDX headers and remain
subject to their original notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Paired-goal recovery pilot (2026-10-18)

See [the training and validation protocol](docs/recovery_controller_v4_20261018.md) for correct-goal pairing, train-only failure sampling and the two-seed evaluation.

### Recovery training factor ablation (2026-10-19)

See [the controlled four-arm protocol](docs/recovery_factorial_20261019.md) for separate tests of correct-goal pairing and failure weighting, followed by two-seed validation.
