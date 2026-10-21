# Astra agent coordination: first simulation diagnostic

## Research question

Astra reasons from an earlier image while the hand continues moving. When its tool call arrives, how much has the state changed, and does response-transition control affect completion?

This experiment is a development diagnostic for agent execution coordination. It uses real Astra calls. It does not yet train a new policy for accepting/rejecting instructions or requesting replanning. State motion alone is not evidence that a returned instruction is invalid.

## Design

All arms use the frozen visual student, identical Astra prompt and six-call budget, and fixed slow control while a request with a known skill is pending. Initial cold start holds the commanded joint targets in every arm.

| Response handoff | Additional simulated response delay |
| --- | --- |
| Direct frozen student | 0 seconds |
| Existing learned controller for two seconds, then student | 0 seconds |
| Direct frozen student | 6 seconds |
| Existing learned controller for two seconds, then student | 6 seconds |

The extra delay applies only to requests made after a skill has been delivered. It is added to measured wall inference latency when calculating the response delivery step. Actual API latency and added simulated delay are logged separately. Physics does not pause. A completed Future can be delivered only when the prescribed simulation time has elapsed. Responses after terminal states remain discarded.

Two reset seeds, 75201 and 75202, six canonical language goals per batch and four conditions give 48 formal episodes. Each cell has 12 episodes. An excluded six-case smoke run uses seed 75199 with learned handoff and extra delay. The finite runner executes smoke first and verifies actual response delivery, learned transition use and injected delay before proceeding. No automatic retries or baseline promotion occur.

The seed-75201 and smoke batches record videos. Every requested image and corresponding delivery-time image are also saved as PNG pairs. PNG writing occurs after scored physics finishes. Images are buffered during rollout. Sensor logging still has runtime overhead, common to both arms.

## Measurements and learning data

For every request and delivered response:

- Previous delivered intent and returned Astra tool decision.
- RGB-D compact features, proprioception and normalized commanded joint targets.
- Request-image age and actual simulation steps advanced during inference.
- Saved RGB image pairs with their image step indices. A delivery on an odd physics step uses the preceding camera frame, explicitly timestamped.
- Object translation/rotation and joint drift, recorded as privileged diagnostics only.
- Drops and successes within two seconds after response, including censored follow-up windows.
- Episode completion, drop, waiting drop, false stop and number of calls.

At handoff, sensor histories, returned goal embedding, proposed student action and applied action are also stored as tensors. Privileged physics fields are never passed to Astra, the student or the learned controller. These new cases are designated development data and must not later be presented as untouched test data if used for training.

Response windows may overlap. The summary distinguishes response counts and unique episode counts. Request-level observations are correlated and must not be treated as independent trials.

## Interpretation

Compare handoff methods within each delay condition on matched initial cases. Record the additional-delay effect separately. Live Astra outputs and latency are stochastic, and future observations diverge with trajectories. Therefore matched initialization does not create identical high-level decision sequences.

The key initial evidence is whether substantial state drift occurs after an instruction is requested and whether response handling has a measurable effect. The two-seed diagnostic is exploratory. A stronger learning experiment would subsequently need action/decision alternatives under controlled states and separate new test seeds. An increased image age or rotation difference is not automatically an invalid command.

Current learned handoff reuses `outputs/astra_learning_v1/train_20261019/controller_final.pt`. It is the already trained visual/goal/history correction model. It does not yet consume the full Astra dialogue, learn when to call Astra, or decide whether a returned instruction should be rejected.

## Execution and artifacts

- Runner: `scripts/astra_coordination_v1/run.py`.
- Evaluator: `scripts/astra_coordination_v1/evaluate.py`.
- Delivery wrapper: `scripts/astra_coordination_v1/async_control.py`.
- State audit: `scripts/astra_coordination_v1/audit.py`.
- Output: `outputs/astra_coordination_v1/diagnostic_20261019`.
- Status: `progress.json`.
- Aggregate: `summary.json`, generated as completed batches arrive.
- Per-batch: `report.json`, `response_audit.json`, `handoff_samples.pt`, `state_pairs/`, video files where enabled.

The local GPU runner is queued after the existing 96-episode live Astra pilot. It waits at most four hours, then launches an isolated loopback bridge. It allows at most 324 API attempts. The reported-token stop threshold is 900,000, checked before accepting new requests. Concurrent in-flight calls can exceed that threshold and failed request usage may be unknown. This is a finite process, with no recurring scheduled monitor.

Four CPU unit checks pass: added delay only after a known skill, no premature delivery, discarded late replies, and quaternion/state audit correctness. Python compilation passes. The simulator smoke check will occur after the active GPU job releases its lock.

The frozen historical baseline and existing 96-episode pilot remain separate. No performance gain is claimed before this diagnostic finishes.

## 2026-10-19 15:55 recovery

The original pilot stopped at 30 validated episodes following an active Astra SSL transport error in `eval_74101_paraphrase_v3`. The original coordination attempt then failed during `/v1/models` preflight with SSL unexpected EOF and made no generation requests. These were infrastructure interruptions, not evidence of policy failure.

Authenticated preflight later passed. The coordination diagnostic restarted under `outputs/astra_coordination_v1/diagnostic_20261019_retry1`, with its unchanged 324-attempt allowance. It now owns the GPU and is running its six-case smoke. The 96-episode pilot continuation is queued under `outputs/astra_learning_v1/live_20261019_resume1`. It links the five valid prior batches without rerunning them and explicitly records the interrupted batch for replacement. Original files remain intact. The original pilot had spent 125 attempts and 289066 reported tokens. Its continuation allows only the remaining 487 attempts and 1210934 reported tokens, preserving the original total budget.

## 2026-10-19 17:25 transport revision

The first restart did not complete. Coordination smoke failed on an active ReadTimeout after 28 attempts and 67,667 reported tokens. The learning continuation advanced to 42 validated formal episodes, then stopped on active SSL/proxy errors. It used another 49 attempts and 106,569 reported tokens. Two late transport errors in an otherwise accepted learned batch remain recorded. No performance improvement is established: among 12 matched cases, continue succeeded 6 times, V3 5 times, and learned 4 times. The slow arm has only 6 validated cases and is not yet matched to all others.

Connection probing through the unchanged configured local proxy succeeded, and authenticated preflight passed. The new isolated transport implementation in `scripts/astra_transport_v2` uses a per-episode requests Session and permits one retry for a whitelist of SSL, proxy, connection and timeout errors. It never disables certificate verification or changes proxy/authentication settings. Each retry goes through the existing usage ledger. Entire retry/backoff time remains inside measured inference latency while physics advances. Authentication, quota and model/tool-format errors are not retried.

New outputs:

- Coordination: `outputs/astra_coordination_v1/diagnostic_20261019_transport2`.
- Learning comparison: `outputs/astra_learning_v1/live_20261019_transport2`.

The coordination budget remaining is 296 attempts and 832,333 reported tokens. The learning budget remaining is 438 attempts and 1,104,365 reported tokens. These amounts preserve the original total allowances across all attempts. Seven validated learning batches are inherited unchanged. Both interrupted original batches remain in their original folders and the continuation protocol records their exclusion/replacement history.

For newly attempted formal batches, isolated exhausted transport retries are retained in the total episode denominator and flagged separately. A batch with three or more affected cases stops the run as a network circuit breaker. Smoke remains strict. Non-transport errors stop the run. Summary output stratifies inherited single-attempt data and new two-attempt data. Pooled results must not be interpreted as a controlled comparison with identical transport conditions. A separate error-free subset is descriptive and potentially selected.

Three retry tests passed. A separate analysis check verifies that an exhausted-retry episode remains in the denominator and appears in the infrastructure audit. The current coordination smoke is running, followed by the queued learning continuation. `outputs/astra_coordination_v1/current_runs.json` points to these latest runs. No recurring monitor was created.

## 2026-10-19 evening runtime diagnosis

Transport2's six smoke requests all completed. An offline replay of the shared controller runtime exposed an integer request-age assignment into a floating buffer. Isaac Sim cleanup then hung, masking the exception until the batch watchdog. The isolated fix is in `scripts/astra_runtime_v2`, with regression and simulator replay verification. Current coordination run: `outputs/astra_coordination_v1/diagnostic_20261019_runtime2`, queued behind the resumed learning pilot on the GPU lock. Formal coordination results remain 0/48 at this checkpoint. Its remaining original allowance is 290 attempts and 824,892 reported tokens. No new budget or seed has been added.
