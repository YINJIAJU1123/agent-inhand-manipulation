# Astra subscription evaluation v6 — 2026-10-15

## Purpose and scope

Add a real image-conditioned Astra tool supervisor to the existing semantic symbol reorientation benchmark. This is a separate run from the frozen rule/latent/oracle v5 experiment. No v5 critical source file was modified.

The public agent inputs are the instruction, RGB view and RGB-derived crop, elapsed control steps, previous skill/state, and image/proprioception changes. Ground-truth symbols, layout, target color, pose, orientation error, and scored success are not passed to the model. The output is one bounded observation/execution/verification/stop/abstention tool call. The existing frozen RGB-D student executes the selected color skill.

## Protocol

- Model: `gpt-6-astra`, standard service, reasoning effort `low`.
- Subscription authentication: app-specific Sign in with ChatGPT OAuth; no Platform key fallback.
- Seeds 35001–35005, canonical and paraphrased instructions, six symbols: 60 episodes / 30 initial physical scenarios.
- Six total calls and 600 simulation steps per episode, same score threshold and student/action scale as v5.
- Full per-episode input/output history over HTTP; `store:false`, `stream:true`, namespaced tools.
- Two concurrent HTTP requests; simulator steps only after decisions for the observed timestep are ready. Different episodes have separate model histories.
- Whole experiment cap: 360 attempted requests and 3,000,000 reported tokens. Token cap is checked before admission; in-flight calls may finish above it. Missing usage on failed requests is not treated as known zero usage.
- Invalid tools, broken streams, authorization errors, usage limits, or simulator failures stop the run. No silent policy replacement or repeated API retries.
- Record videos for both templates of seed 35001; save per-call token usage, tool arguments, image hashes, and timings.

Primary success requires correct semantic binding, execution of that goal, orientation error <= 0.16 radians, and no drop. The scorer terminates on reach. Stop declarations and false stops are recorded separately; this protocol does not establish autonomous successful stopping.

## Resources and credentials

Remote simulator: `brainco@10.3.100.21`, physical GPU 7 (RTX 5090), isolated project `/home/brainco/agent_inhand_astra_20261015`. IsaacSim is 5.1 on both local and remote hosts. The remote runner records IsaacLab and PyTorch versions and hashes source/assets/checkpoints. GPU and software differences still need to be considered before interpreting cross-system comparisons. Physical GPU 7 is masked with `CUDA_VISIBLE_DEVICES=7` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so physics uses logical `cuda:0`; Kit rendering is explicitly set to physical GPU 7.

OAuth credentials remain in the local app credential directory outside the repository. The simulator reaches a loopback-only bridge through SSH reverse forwarding using a separate short-lived experiment key. The local launcher owns the bridge/tunnel, mirrors reports/logs, and downloads videos when the finite run ends. No scheduled task is created.

Remote outputs: `/home/brainco/agent_inhand_astra_20261015/outputs/astra_eval_20261015`.
Local mirror: `outputs/agentic_v4/astra_eval_20261015`.

Compare only the matching v5 subset and verify saved initial states/layouts; do not compare a partial Astra sample to all v5 episodes as if they were matched. Report infrastructure interruptions separately from completed robot episodes.

## Verification before rollout

- Actual OAuth catalog and Astra text completion succeeded.
- Two actual image/tool calls on a development frame succeeded, including replay of the first call and measured feedback. These are integration checks, not robot success measurements.
- Transport tests cover streamed tool extraction with empty final output, cross-episode history rejection, failed-stream handling, and concurrent call-cap enforcement.
- Two startup failures were preserved in `astra_eval_20261015_startup{1,2}_failed`. The underlying issue was a shared `/tmp/isaaclab/logs` directory owned by another user; no rollout API requests occurred. Startup now uses a private writable TMPDIR.
- A third startup was terminated before any API calls because USDRT rejected physics on `cuda:7`. Its output and source snapshot are preserved in `astra_eval_20261015_startup3_failed`. The physical/logical GPU mapping above addresses this restriction; the renderer log confirms GPU 7 is active.

## Entry points

- `scripts/agentic/chatgpt_plan_smoke.py`: consent and minimum connectivity check.
- `scripts/agentic/astra_subscription_v6.py`: credential refresh, history, SSE parsing, tool validation, request ledger.
- `scripts/agentic/astra_bridge_v6.py`: bounded loopback bridge and remote client.
- `scripts/agentic/evaluate_astra_v6.py`: simulator and concurrent supervisor decisions.
- `scripts/agentic/run_astra_v6.py`: frozen finite experiment and aggregate reports.
- `scripts/agentic/launch_remote_astra_v6.py`: local bridge/SSH/process ownership and result synchronization.
- `scripts/tests/test_astra_subscription_v6.py`: offline transport tests.

## Live verification (2026-10-15 06:09 UTC)

The remote simulator passed initialization and real Astra tool requests have begun.
The first seed's object position, object quaternion, and symbol layouts match the
v5 oracle initial state exactly; maximum proprioception difference is 1.19e-7.
This verifies starting-state correspondence, not cross-hardware trajectory equivalence.
The shared installation still emits nonfatal cache/config permission warnings.
All three offline transport tests pass. Full robot results remain pending.

## First completed batch (2026-10-15 06:14 UTC)

Seed 35001 / canonical completed all six episodes: 3/6 reached the primary
success criterion, 6/6 ended with correct semantic binding, zero drops, and
three false stop declarations. All three failures were premature stop requests.
The batch used 23 successful model calls and 53,893 reported tokens
(52,738 input / 1,155 output). This is one seed cluster, not a stable success estimate.
The runner automatically advanced to the paraphrase batch.

The completed report and six continuous videos are mirrored locally under
`outputs/agentic_v4/astra_eval_20261015/seed35001_offset0_canonical_astra/`.
Success uses the scorer's <= 0.16 rad full-orientation criterion; the VLM's
visual completion judgment can disagree with this stricter metric. Do not
change prompts or stop logic during this frozen batch.
