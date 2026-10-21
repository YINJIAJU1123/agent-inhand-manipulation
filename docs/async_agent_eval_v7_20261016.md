# Live asynchronous Astra / fine-tuned VLM evaluation v7

## Research question

When language/vision inference takes time, can a small sensor-only supervisor protect an ongoing in-hand manipulation task without freezing physics?

The frozen student, text adapter, fine-tuned Qwen3-VL-4B LoRA/continuous goal head, and lightweight waiting supervisor are reused. This evaluation does not retrain them. It is a new asynchronous benchmark; synchronous v5/v6 success rates are not directly comparable.

## Registered comparison

- Two high-level backends: real Astra through the existing Sign in with ChatGPT bridge, and the trained continuous-latent VLM.
- Three waiting policies per backend: continue the last delivered skill, fixed 0.15 interpolation from the current commanded joint target, learned continue/slow/hold supervisor.
- Ten new seeds 73001–73010, canonical and paraphrased language, six symbol goals: 120 episodes per arm, 720 total.
- Six requests maximum per episode; 3,600 control steps / 120 physical seconds maximum. Completed and failed episodes stop scoring immediately. No auto-reset trajectory is counted as a retry.
- A separate 12-episode live smoke check precedes the experiment and is excluded from its statistics.
- Astra evaluation cap: 2,160 requests, plus at most 36 smoke requests. Stop new requests once the usage ledger reaches 6,000,000 tokens. In-flight requests can exceed the token stop threshold; this is not a hard token reservation.
- Rotate waiting-policy execution order across seeds. Each six-episode batch starts a fresh simulator. Audit matching initial joint positions/velocities, object root state, commanded targets, and symbol layouts, numerical tolerance 1e-6.
- Real service sampling/latency is independently observed in each arm, not identical counterfactual latency. Report latency distributions and matched initial-case outcomes; uncertainty is bootstrapped by reset seed, not by treating all correlated cases as independent.

## Non-pausing implementation

`AsyncAgent` owns one future per episode. Worker threads perform HTTP inference and mutate only their private high-level agent. The physics thread uses a delivered snapshot, never partial worker state. It only reads a result after `future.done()`.

A response becomes available after both the real inference finishes and the simulator advances `ceil(measured request wall time × 30)` control steps from request submission. This includes local queue and network time. This timing correction prevents a slow renderer from giving the robot artificially short physical exposure to a long API call. It is **not a claim of 30 Hz wall-clock throughput**. Requests are real, not prerecorded; physics continues during both inference and delivery alignment.

- Physics 120 Hz, joint targets 30 Hz, fresh camera 15 Hz, lightweight supervisor 2 Hz.
- Main camera 512×512, frozen student/compact waiting features resized to 256×256.
- Before the first delivered skill, all arms latch the initial commanded joint targets. No ground-truth goal is provided for cold-start control.
- During later pending requests, continue the last delivered skill with the assigned waiting policy. A delivered inspection skill is valid control input even when semantic grounding remains unknown.
- Use the fine-tuned VLM's continuous embedding directly; nearest skill ID is only a diagnostic/scoring label.
- Start the returned action's duration at delivery, not at the now-stale request timestamp.
- Stop and abstain are applied at response delivery. Completion is scored against the state then, not the request image.
- Drop/termination permanently closes the episode. Late responses are audited and discarded, never applied to a reset episode.
- After all scored episodes end, outstanding inference is drained only for accounting. This is outside the simulated trial.

## Scoring and limits

Primary success follows v5/v6: correctly delivered semantic binding, execute/verify state, orientation error ≤0.16 rad, and no drop. The evaluator terminates successful slots; autonomous stop correctness is reported separately. This differs from the waiting-controller training pilot's six-frame recovery hold requirement.

Record total success/drop, drops while pending, cold-start duration, waiting-mode usage, false stops, actual request latency, physical steps while pending, discarded late responses, and API token usage. Initial-state audits do not serialize hidden contact-solver caches; fresh processes minimize shared-history confounds.

The waiting supervisor was trained for delayed verification of an already delivered goal. Inspection, changing goals, and cold-start stability are outside its original training distribution. Initial holding and immediate response handoff are prescribed mechanisms. A gain here must be measured against fixed slowing; do not assume the trained gate is superior.

## Artifacts and execution

- Evaluator: `scripts/agentic/evaluate_async_v7.py`
- Nonblocking response lifecycle: `scripts/agentic/async_agent_v7.py`
- Dedicated latent GPU process: `scripts/agentic/latent_service_v7.py`
- Finite remote experiment: `scripts/agentic/run_async_v7.py`
- Local bridge/service owner: `scripts/agentic/launch_async_v7.py`
- Paired summary and seed-cluster confidence intervals: `scripts/agentic/analyze_async_v7.py`
- Regression checks: `tests/test_async_agent_v7.py`

Local outputs are under `outputs/async_v7/`; remote outputs under `/home/brainco/agent_inhand_astra_20261015/outputs/async_v7/`. Each run stores source hashes/snapshots, manifests, reports, request events, progress, and videos for smoke and seed 73001 (both language templates, all arms). OAuth credentials remain local; only a run-scoped bridge key travels to the remote host through SSH.

This is a finite evaluation runner, not a scheduled automation. Status and final results must be read from the run artifacts, not inferred from this protocol document.

## Live smoke completed; large experiment launched

Run `smoke_20261016a`, seed 72999, canonical instructions, learned waiting policy:

| Backend | Success | Drop | False stop | Timeout | Requests |
|---|---:|---:|---:|---:|---:|
| Astra | 3/6 | 2 | 1 | 0 | 24 |
| Fine-tuned latent VLM | 3/6 | 1 | 0 | 2 | 26 |

No inference errors. Astra delivered 23 responses and discarded one after episode termination; each delivered request advanced at least 144 physical control steps while pending. Request latency median/P95/max: 6.44/9.18/15.70 seconds. Latent VLM delivered 25, discarded one, minimum six pending control steps, latency 0.48/0.98/1.32 seconds. All tested initial joint positions/velocities, object root states and commanded targets matched exactly between backends. These small smoke outcomes are excluded from the formal comparison.

Formal run: `outputs/async_v7/large_20261016`, 720 planned episodes. It reuses the completed smoke only after checking the evaluator, agent, controller and inference-service source hashes. The local launcher owns the inference services and SSH tunnel, mirrors artifacts, and closes its services on termination. The smoke launcher lost its original shell owner; its already-detached workers completed successfully, then a bounded recovery process archived results and closed only its owned services. The formal launcher was started in an independent session and its PID/status are recorded in `launcher_status.json`.

## Migration to local evaluation after transport failure

The remote `large_20261016` run stopped early. The only infrastructure-clean completed batch was Astra/continue, 2/6 successes and four drops (two while pending). Astra/slow then encountered two active-request `SSLError` failures; the original shared failure handler also interrupted latent/continue. These are incomplete results, not a completed six-arm benchmark. The 31 API attempts reported 58,585 tokens; failed-stream token usage may be unavailable. The bridge exposed only the error class, so the logs do not establish whether the TLS failure originated from the network or the upstream service.

At the user's request, evaluation moved to the **local RTX 3090**, with no further jobs or SSH connections to host 10.3.100.21. The new finite runner is `scripts/agentic/run_local_async_v7.py`, output `outputs/async_v7/local_20261016`. Simulation and fine-tuned VLM inference run in separate local processes on GPU 0; simulator batches run sequentially. Astra remains a live online model call through a local loopback bridge. Moving the simulator does not itself fix upstream HTTPS failures.

The local run starts fresh and does not pool 5090 outcomes into its 720-episode result. A new local smoke is required to validate GPU memory sharing and the local runtime. Backend services and evaluation errors are isolated: failure of one backend does not abort the other backend's remaining work.

Transport retry is bounded to one additional attempt, only for an explicit transport-error reason. It never disables certificate verification or changes authentication. Both attempts consume the original request ledger, and total request/retry wall time contributes to physical waiting. Budget, authentication, and malformed-output errors are not retried by the transport layer. A batch can have at most two recorded attempts; persistent failure blocks that backend while the other continues. Failed attempts are retained separately and excluded from model success rates, with exclusion records reported; discarded responses after a previously scored terminal state remain accounting events.

Prior API requests and reported token use are subtracted from the local run's allowance, so restarting does not silently reset the announced experiment budget. The local progress file distinguishes starting/running/blocked backend states and is refreshed every five seconds while a simulator runs. No scheduled automation is created.

## Completed local evaluation — checked 2026-10-17

The local RTX 3090 run completed all 720 formal episodes (120 per arm, ten
seeds, canonical/paraphrase instructions). The runner exited normally, neither
backend was disabled, and no batch was excluded for an infrastructure failure.
One late failed inference remains in the audit, after its episode had already
terminated; it is not a zero-error claim. All 100 cross-arm initial-state
comparisons have maximum absolute difference zero.

| Waiting behavior | Astra | Fine-tuned latent VLM |
|---|---:|---:|
| Continue frozen student | 60/120 (50.0%) | 59/120 (49.2%) |
| Fixed slowing | 68/120 (56.7%) | 47/120 (39.2%) |
| Learned v1 supervisor | 61/120 (50.8%) | 43/120 (35.8%) |

Astra waiting drops, including cold starts, are 36/120, 20/120 and 35/120 for
continue, slow and learned. Each has 12 cold-start drops under the common fixed
initial hold, leaving 24, 8 and 23 drops during known-goal inference waits.
Thus waiting risk is observed and fixed slowing has the lowest observed wait
drop count, but the learned supervisor has no established gain. Learned versus
continue is +0.83 percentage points for Astra, with a seed-cluster bootstrap
95% interval of [-15.83, +15.83] points; for latent VLM it is -13.33 points,
interval [-21.67, -5.83]. These paired comparisons share initial states, while
live response contents and latency are independently sampled in each arm.

Full results: `outputs/async_v7/local_20261016/summary.json`.
Drop-phase diagnostics and paired intervals:
`outputs/async_v7/local_20261016/final_diagnostics.json`.
The subsequent fixed-delay counterfactual experiment is a separate protocol.
