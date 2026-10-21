# Astra baseline freeze and learned temporal controller v1

## Frozen historical live baseline

Date: 2026-10-19. Original experiment: `outputs/async_v7/local_20261016`.

| Backend and waiting policy | Success | Rate | Drops | Drops while pending |
| --- | ---: | ---: | ---: | ---: |
| Astra continue | 60/120 | 50.00% | 56 | 36 |
| Astra fixed slow | 68/120 | 56.67% | 40 | 20 |
| Astra old learned v1 | 61/120 | 50.83% | 51 | 35 |
| Fine-tuned VLM continue | 59/120 | 49.17% | 47 | 9 |
| Fine-tuned VLM fixed slow | 47/120 | 39.17% | 59 | 13 |
| Fine-tuned VLM old learned v1 | 43/120 | 35.83% | 64 | 13 |

The raw 720 formal episodes were recounted. Each arm has 120 unique seed/template/case combinations. The original protocol, source snapshot, report/initial/manifest files and five weights were checked and copied into `outputs/astra_learning_v1/baseline_freeze_20261019/artifacts`. The freeze manifest records 426 SHA-256 checksums. Authentication files are excluded. Historical files are not overwritten.

These are full live asynchronous results. The scorer requires the correct delivered semantic binding, execute/verify, orientation error <=0.16 radians and no drop. It ends a successful episode immediately. It does not require the six-frame hold used in later known-goal controller diagnostics. Autonomous stop is a separate metric. One Astra continue request failed after the episode had already ended and was discarded. Historical initial-state comparisons were exact.

This freeze establishes a versioned reference, not a claim that 56.67% is an immutable population success rate. New methods must be compared against freshly rerun controls on matching new cases. Known-goal recovery success around 70% is a separate quantity.

## New learning module

Astra remains a frozen high-level tool caller. Its color/skill output is mapped to the existing canonical goal embedding. Astra hidden embeddings are not available to this controller.

The local learned representation receives:

- Four compact RGB-D frames at 15 Hz, encoded by a small GRU.
- Current proprioception, student action proposal and command history.
- Delivered goal embedding, centered and projected into the five-dimensional span of the six canonical goals.
- Observed request age, pending flag and elapsed time since the response.

The output is a 128-dimensional state representation. Outcome and return heads supervise this representation using historical training rollouts. Outcome classes are success, drop and non-drop failure under the mixed collection policies. These are empirical behavior outcomes, not oracle recoverability or calibrated safety predictions.

After fitting the representation, its encoder is frozen. An action head is trained by offline advantage-weighted behavior regression with regularization toward frozen V3. It adds a bounded correction to the V3 raw action. The maximum absolute raw correction is 0.1. The existing V3 mapping retains the normalized action bound and the joint-residual bound of 0.02. The visual student and its action scale remain frozen.

The outcome head is logged during this first live pilot. It does not trigger stopping and is not sent to Astra yet. That isolates the effect of learning the local representation and control correction from changing Astra prompts or feedback.

## Training and development evidence

Output: `outputs/astra_learning_v1/train_20261019`.

- Source: actual tensor packs in the prior factorial **training** directory. No evaluation rollout is loaded.
- 1,524 episodes with valid controller decisions, sampled at 24 frames per episode.
- Fit: seeds 930100, 930101, 930102, 1,128 episodes.
- Development selection: whole seed 930103, 396 episodes.
- Development Brier error: 0.48747, versus 0.55004 for training class-frequency prediction. Lower is better.
- Selected representation epoch: 1. Later epochs overfit the development split.
- Selected correction epoch: 0. Weighted regression objective 1.36335 versus 1.36433 with no correction. This is a small offline change.
- Model checksum: `8c941d5b578393e619aabd2fe540a387a346d736aba6ce71b75bbb85a282c9f2`.

The development seed was excluded from fitting this model, but historical collecting policies evolved across these seeds. It is not an untouched test split. The terminal labels also come from known-correct-goal runs and a different completion protocol. These limitations prohibit interpreting prediction metrics as live Astra success. No increase in end-to-end success has been established yet.

## Live pilot

Output: `outputs/astra_learning_v1/live_20261019`.

Four arms share the frozen student, Astra prompt, semantic interface and live scorer:

1. Astra + continue.
2. Astra + fixed slow.
3. Astra + frozen V3 through the new asynchronous runtime.
4. Astra + V3 with the learned representation/correction.

Two new formal seeds 74101 and 74102, canonical and paraphrase instructions, six cases per batch. This gives 24 episodes per arm, 96 formal episodes in total. A separate six-case smoke run uses seed 74099 and is excluded. Arm order is seeded and shuffled. Initial states are compared across arms. All seed-74101 batches and smoke cases record video.

Physics continues during actual inference. Delivery waits for both actual Future completion and the corresponding elapsed physics steps. This does not claim wall-clock real-time simulator throughput. A learned controller runs only with an already delivered skill, while a call is pending and for two seconds after each response. Cold start holds the original targets. Response arrival or goal changes refresh the control immediately even on odd physics steps. At the end of recovery the frozen student resumes exactly. Request duration is retained after response, matching the training observation convention.

The finite local runner waits for the current phase-crossover experiment to release the GPU lock. It has a two-hour queue timeout. It then runs smoke and formal batches sequentially. It creates no scheduled monitor. The bridge allows at most 612 API attempts, at most six calls per episode, and stops accepting new requests after 1.5 million reported tokens. In-flight calls can exceed that token threshold and failed-call usage may be unknown. There are no silent batch retries or automatic baseline replacement.

## Verification and current status

- Historical counts and 426 artifact hashes verified.
- Safe tensor loading succeeds for student, adapter and controller checkpoints.
- Four CPU tests pass: exact zero-correction V3 behavior, response refresh on an odd step, exclusion of unknown goals from learned control, and equality with training observation features during recovery.
- Astra authentication/model availability preflight passed without a generation request.
- CPU training completed.
- Live runner launched and queued behind the existing GPU experiment. End-to-end result remains pending. The live progress file and summary, once generated, are authoritative.

## 2026-10-19 evening status correction

The former queued/running description above is historical. Training is complete. The live pilot reached 66/96 complete reports before a request-age dtype error was masked by hanging Isaac Sim cleanup. An offline replay reproduced the error. The isolated runtime fix passes two regression tests and a six-environment 350-step replay. Current continuation: `outputs/astra_learning_v1/live_20261019_runtime2`. The 66-report descriptive totals are continue 10/18, slow 7/18, V3 11/18 and new learned 4/12. On the 12 common cases they are 6, 4, 5 and 4 successes. No improvement is established. See `docs/training_evaluation_status_20261019.md` for protocol boundaries, budgets and detailed analysis.
