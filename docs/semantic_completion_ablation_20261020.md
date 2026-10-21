# Semantic evidence and autonomous completion ablation — 2026-10-20

## Question

Can learned visual completion verification and learned observation selection convert stable in-hand execution into successful autonomous completion?

The four groups share Astra, the frozen RGB-D student, action scale, the V3 waiting/handoff controller, initial cold hold, six calls and 3,600 control steps. They differ only in the enabled learned assistance:

| Arm | Learned verifier | Learned observation selection |
|---|---|---|
| baseline | off | off |
| verify | on | off |
| observe | off | on |
| joint | on | on |

All groups receive remaining call/time budgets and actual executed skill feedback. All execute the same local evidence frontend to avoid different local computational load. The baseline is therefore a newly specified baseline with budget feedback. It is not interchangeable with the historical reach-terminated baseline.

## Completion protocol

Reaching the goal does not terminate the episode. Autonomous completion requires a stop declaration, correct delivered semantic binding, full quaternion orientation error at the time of stopping at most 0.16 rad, and no drop. The evaluator records reach separately and never passes goal color, symbol layout, orientation, error or scored completion to the agent.

The verifier can reject an Astra stop proposal using current sensor predictions. It continues that skill, allows remaining Astra calls, and can accept the same pending proposal later. A subsequent non-stop Astra reply cancels the proposal. Rejecting a stop never refunds calls. A pending proposal that never becomes acceptable ends in timeout or another actual terminal outcome.

The first observation module predicts readability and survival over 90-step candidate color skills. It selects an alternative only for an Astra `inspect_face` reply, records the actual executed skill, and supplies that information on the next call. It does not replace execute/verify with a ground-truth target. Its score combines predicted evidence gain, predicted drop cost and a disclosed small repeated-inspection penalty. Duration and risk weights are fixed in this initial pilot, not learned.

## Learning implementation and limits

A 552,238-parameter model combines per-color RGB patches, a temporal RGB evidence encoder and RGB-D feature/proprioception history. Heads predict symbol/unknown probabilities, current visibility, full-pose completion, and action-conditioned future readability/survival. The 128-dimensional temporal state is a locally trained latent, not an Astra hidden state.

Color patches are extracted from RGB only. Per-face symbol memory retains the latest confident observation and does not multiply confidence on repeated views. The local helper parses the closed six-symbol vocabulary from instruction text with disclosed aliases. It never derives the goal from case indices or layouts. Astra still receives the full original instruction. This is a first implementation of learned visual assistance, not an unrestricted learned language parser.

Training uses frozen-student random skill rollouts on eight training seeds 96111–96118 and validation seeds 96211–96212. Symbol layouts come from disjoint train/validation permutation banks. Four-frame evidence histories and student histories provide temporal information. Full-pose and readability labels are training-only. Readability uses the existing corrected symbol-visible versus symbol-hidden render comparison, not just the earlier depth proxy.

The first pipeline smoke at `outputs/semantic_completion_v1/pilot_20261020/data/train_96101` used the earlier depth proxy and is excluded from all training and formal results.

The initial future-action head uses observed random skill transitions, not same-state counterfactual branching and not a final task-success Q function. Joint completion-value training remains a later extension. Best checkpoint selection uses validation loss only. The pose gate threshold is chosen on validation data for at least 10 accepted examples and at least 90% empirical precision when possible. Failure to obtain that calibration is explicitly reported. Correlated frames and two validation seeds do not establish deployment calibration. Before admitting the cloud experiment, require that validation finds a calibrated pose gate under the stated rule and readable-symbol accuracy is at least 70%. These checks can reject an unusable first model, but do not establish generalization or an improvement over the baseline. A separate per-seed audit reports coverage, false accepts and confidence-filtered symbol precision.

## Planned finite evaluation

- Test seeds: 86101 and 86102, unused for model fitting or checkpoint selection.
- Six semantic targets, canonical/paraphrase instructions, two repetitions.
- Four groups, 48 episodes per group, 192 episodes total.
- Randomized group order within matched seed/template/repetition blocks.
- Fresh simulator processes and saved initial states. Abort on mismatched initial states.
- Actual asynchronous Astra requests, moving simulation. Delivery aligns measured inference latency with 30 Hz control time. This is not a wall-clock real-time throughput claim.
- Six logical calls per episode. One transport attempt per call. Whole experiment at most 1,152 attempted requests and a 3,000,000 reported-token admission threshold. Already in-flight calls may exceed the token threshold. Failed requests with missing usage remain unknown cost.
- Record all six cases for both templates of the first test seed in the first repetition, for all four groups: 48 videos, including failures.
- No silent outcome-based retries, no scheduled task, no additional seeds.

## Service anomalies

For this new run, define a service-abnormal episode before evaluation as any active API error or any request with measured latency above 120 seconds. Keep the original episode and raw outcome. In the matched normal-service analysis, remove the same seed/template/repetition/case key from every group. Report excluded keys and counts. This rule applies to every arm, regardless of success.

Normal inference delay remains part of the moving-physics task. The raw and service-normal results are distinct estimands. This new preregistered rule does not retroactively make the prior exploratory exclusion analysis preregistered.

## Account migration and approval

The user completed a new app-specific Sign in with ChatGPT authorization. The new identity differs from the old one. A text smoke using `gpt-6-astra` completed in 2.95 seconds with 26 reported tokens. New credentials are isolated outside the repository, never copied to reports or remote workers. The new bridge checks a nonsecret identity digest before refresh/inference and cannot fall back to the old profile.

Automatic approval review subsequently rejected the proposed transmission of a local simulation frame, despite the user's authorization of Astra evaluation. A separate explicit request now names the image, endpoint, subsequent experiment payloads and budget. Until that request is approved, only local collection, training and offline validation may run. The evaluator runner requires the recorded approval before starting its cloud bridge.

## Status at implementation

- Real local collection is running on the RTX 3090.
- Training and validation splits are fixed.
- Seven evaluator/agent regression tests pass.
- Four-group evaluator, new-account bridge and paired summary are implemented.
- Training metrics and simulator integration results are pending.
- No formal four-arm success numbers exist yet.

Entry points: `scripts/semantic_completion_v1/prepare.py`, `train.py`, `evaluate.py`, `run.py`, `analyze.py`, `test_completion.py`.

Artifacts: `outputs/semantic_completion_v1/learning_20261020` and `outputs/semantic_completion_v1/account_20261020`.

Local integration runner: `outputs/semantic_completion_v1/offline_check_20261021`, explicitly launched with `--offline-only`. It waits for training and then runs 140 steps for each group using deterministic stub replies. It cannot start the cloud bridge. These 24 smoke episodes are excluded from the 192 planned formal episodes.

## Development update — 2026-10-21

The new account text test passed. The four offline simulator arms completed 24 smoke episodes, exercising inspect overrides, rejected stops, drop termination, horizon termination and budget preservation. Those stub-reply episodes are plumbing checks, not method results.

The first collected dataset contained 2,482 training frames and 705 validation frames. Its best-total-loss model had 1.59% readable-symbol accuracy with the seven-class decision (six symbols plus unknown). Approximately 86% of training labels were unknown. The star class had only 34 temporal training labels and no readable validation labels. An initial class-balanced repair reached 29.96% readable-symbol accuracy, still below the formal admission gate. No formal cloud episodes were launched from either model.

Further inspection found a labeling omission in this implementation: the collector copied the older render-difference rule but omitted the actual black-ink radius correction already present in `scripts/agentic/prepare_latent_dataset_v4.py`. It also treated ambiguous partial views as unknown-class negatives. The current collector now applies the existing `(0.0075/0.009)^2` area correction and masks ambiguous symbol supervision. This corrects training labels, not the autonomous-stop success threshold. Earlier pilot artifacts remain available and are superseded for semantic-head training.

The active finite local pipeline is `outputs/semantic_completion_v1/corrected_repair_20261021` (entry point `repair_all.py`). It re-collects the original train/validation temporal splits with corrected labels, renders additional layouts at replayed training/validation physical states, then trains and repeats offline integration checks. Symbol-layout variants stay inside their original split permutation banks. They are single-view auxiliary examples, not temporal continuations or action counterfactuals. The trained pose branch from the class-balanced development checkpoint is frozen during this semantic repair. It used only the same development validation seeds; formal evaluation seeds remain unused.

The cloud admission check additionally requires six-class macro accuracy of at least 70% on the auxiliary held-out symbol views, at least 30 confidence-filtered predictions, and at least 90% empirical precision for those predictions. These are pilot quality checks, not deployment guarantees. Selection uses development validation only. A failed gate is recorded rather than bypassed.

Local collection/training uses no cloud calls. Real-image Astra testing and the planned 192 formal episodes remain blocked on the explicit image-transfer confirmation requested after automatic approval review rejected that transfer. Account OAuth itself is already verified.


## Completed corrected run — 2026-10-21 00:51 CST

The finite local pipeline completed normally with final status `needs_improvement`. It is no longer running. It collected 2,949 temporal training frames and 706 validation frames, plus 1,884 auxiliary training views and 864 auxiliary validation views. All six symbols are represented in the auxiliary validation set. The repaired semantic model trained for 80 epochs and selected epoch 73. A further 24 offline simulator smoke episodes completed across all four arms, with zero cloud calls. These are integration checks and are excluded from formal evaluation.

Held-out results of the selected model:

- Original temporal readable-symbol accuracy: 36.12%.
- Auxiliary six-symbol macro accuracy: 28.21%.
- Confidence-filtered auxiliary symbol precision: 61.60% on 125 accepted predictions.
- Learned completion gate: no tested threshold met the predeclared minimum count and 90% precision criterion.
- Cloud admission: false. Formal four-arm episodes: zero.

A read-only diagnostic at `outputs/semantic_completion_v1/corrected_repair_20261021/diagnosis_20261021.json` found auxiliary symbol precision of 90.43% on the training split versus 61.60% on validation. Readable-symbol training accuracy was itself only 49.39%, so both representational/classification difficulty and a generalization gap remain. At pose probability threshold 0.95, 29 of 53 accepted validation samples were truly within the completion tolerance (54.72% precision). At 0.99, 8 of 11 were correct (72.73%). These are correlated development frames, not independent task episodes.

The pose branch retained from the earlier development checkpoint did not transfer its previous empirical calibration to the newly collected validation trajectories. It must be retrained or recalibrated on the corrected dataset with a valid held-out check. Merely raising the threshold does not meet the current admission criterion. The baseline student and V3 controller have not been replaced, and these auxiliary module percentages are not task success rates. Comparisons against older labeling versions are exploratory and do not establish an apples-to-apples improvement.

Next development priority: preserve the existing Astra semantic grounding, improve the target-conditioned completion/evidence representation, and train explicitly on near-completion false-stop states. Reassess learned observation selection after the evidence model passes validation. The image-transfer confirmation required by automatic approval review also remains outstanding, but the current model validation failure independently prevents a formal run.


## Completion representation revision — 2026-10-21

V2 keeps Astra responsible for semantic binding and learns completion and short-horizon progress conditional on its proposed color goal. It reuses the frozen RGB-D student memory, commands and measured proprioception. All six candidate language goals are queried at the same recorded state. Simulator orientation and hidden symbol layouts are never feature inputs. This avoids requiring the unsuccessful V1 symbol classifier to become a second semantic grounder. It also narrows the claim: a pose completion prediction cannot independently detect an incorrect Astra symbol-to-color binding.

The old-data V2 MLP pilot did not obtain a completion threshold with 90% empirical precision. The calibration trajectory contained only 25 positive frame labels, with no blue completion example. A linear pilot also failed the same gate. Neither model was admitted to formal evaluation. Coarse goal ranking and fine completion precision are reported separately.

The finite replacement collection is `outputs/semantic_completion_v2/fixed_goal_20261021`. Eight training seeds 96121–96128 each assign all six fixed goals to six environments. Each episode continues for up to 900 steps without success termination, collecting approach, reach, subsequent drift and failures. Calibration seed 96311 and audit seed 96312 use disjoint validation symbol layouts. The original eight training shards may be reused, but their validation shards are excluded. Formal seeds 86101/86102 remain untouched.

The queued optimizer is `outputs/semantic_completion_v2/optimization_20261021`. It fits seven ordinal pose-tolerance heads, an error estimate and observed same-skill 30-step progress. Missing future suffixes after drop or horizon are censored. The progress head is not a counterfactual action-value model. Training emphasizes near misses and pairs positive and incorrect goal queries. Threshold selection uses the calibration seed only, followed by the audit seed. Subsequent shadow rollouts on these same two development seeds record the first proposed stop while the frozen policy continues. Those rollouts are diagnostic and do not constitute new held-out semantic task results.

A useful candidate must have acceptable completion precision, nontrivial coverage and reliable first-stop proposals. Passing a frame-level threshold check alone does not establish autonomous success. All current V2 work is local with zero cloud calls. Existing student, teacher, action scale, V3 controller and the 0.16-radian full-quaternion success criterion remain frozen. V2 completion/progress results must not be presented as the original V1 four-arm experiment, whose learned observation-selection head has not passed validation.


## V2 completion and first-stop development results — 2026-10-21

The fixed-goal collection completed all ten shards. The merged dataset has 6,379 training frames from sixteen training seeds and 933 validation frames from two development seeds. The trainable completion/progress head has 160,329 parameters. Frozen student and adapter hashes match across caches.

The frame-trained model selected epoch 10 and probability threshold 0.85 using calibration seed 96311. On audit seed 96312, 70 of 72 accepted frame hypotheses were correct (97.22% precision, 53.03% recall). Goal ranking on reached frames was 100%. These are correlated frame metrics. The progress head failed its baseline comparison: Brier error 0.2871 versus 0.2472 for a constant based on the audit prevalence. Progress feedback is disabled by default and cannot be sent to the agent without an explicit validated flag.

A first-stop analysis exposed the remaining failure. On the recorded calibration/audit trajectories, the frame threshold gave 3 and 1 incorrect first proposals respectively. A subsequent fresh-process shadow diagnostic across the same two seeds produced 8 first proposals, only 5 correct, over 12 given-goal episodes. Policy execution continued after all proposals. The runner initially classified the candidate using only the frame gate. Its readiness logic and recorded status were corrected to require a first-stop check as well. The result is `needs_improvement`; no formal evaluation was admitted.

A matched continuation experiment adds a differentiable first-stop objective. It models a soft stop hazard from the completion logit and minimizes the negative log likelihood that the first stop occurs at a true completed state. For a training trajectory that never reaches completion, it favors no stop. Only the 48 fixed-goal training trajectories supply this sequence loss. All six goal hypotheses retain the original per-frame ordinal, pose and ranking supervision. The continuation control receives the same initial checkpoint, optimizer, frame batches, sequence forwards and training budget, with sequence-loss coefficient zero; the new arm uses coefficient 0.2. Both select checkpoints and thresholds on the calibration seed. Runtime receives only measured sensors and the queried goal; labels remain training/evaluator-only.

Recorded-trajectory development results on audit seed 96312:

| Variant | Correct first stops | Incorrect first stops | Trajectories that reached a sampled target state |
|---|---:|---:|---:|
| Original frame gate | 4 | 1 | 5 |
| Same model, trajectory-based threshold only | 3 | 0 | 5 |
| Matched continuation control | 4 | 0 | 5 |
| Added first-stop training objective | 5 | 0 | 5 |

There are six trajectories in this split. These seeds have already been inspected during development, so the table is exploratory and does not demonstrate independent generalization. The sequence arm selected epoch 3 and threshold 0.975; the continuation control selected epoch 1 and the same threshold. This table is not an Astra or language-task success result.

A prospective comparison is running at `outputs/semantic_completion_v2/paired_shadow_20261021`. All four heads observe exactly the same live sensor histories and physics trajectory in each simulator process. They do not change the frozen student's actions. Both development seeds are reused, six cases each. This design isolates stop advice from differences between separately simulated trajectories. Initial states, model hashes, all first proposals, failures and per-frame traces are saved. No cloud calls are made. The original 192-episode four-arm semantic protocol remains pending, and is not replaced by this local comparison.


### Prospective paired pilot completed

The twelve live paired shadow episodes completed. The original frame gate made 9 first-stop proposals, 8 correct. Threshold recalibration alone made 8 proposals, 7 correct. The matched continuation control made 9 proposals, all 9 correct. The new first-stop-loss model made 9 proposals, 8 correct. Ten of the twelve trajectories reached the actual pose tolerance at some point, including short intervals between sampled observations. A correct first proposal is evaluated at that proposal's physical state. These are known-goal stop-advice counts, not full semantic-agent task success rates.

The recorded-trajectory advantage of the new sequence loss did not persist in this fresh-process comparison. It is not promoted as a method improvement. The ordinary continuation model is retained as a provisional completion candidate. Its lack of false first proposals in nine cases is insufficient evidence of reliable stopping.

A frozen expansion at `outputs/semantic_completion_v2/paired_shadow48_20261021` now evaluates 24 simultaneous cases per seed on the same two development seeds, balanced across six goals. All four models and thresholds are unchanged. All heads again share exactly the same physical trajectory and observations for each case. Increasing case count changes the sampled initial states and symbol layouts; this is an additional development diagnostic, not a reuse of the twelve previous episode outcomes. The source code is snapshotted with hashes. No new training, checkpoint selection or cloud calls occur during this expansion. The two formal test seeds remain untouched.


### Expanded paired validation completed, candidate withdrawn

All 48 expanded shadow episodes completed normally. Four frozen heads shared each physical trajectory and every sensor observation. Results are first-stop advice counts:

| Variant | Correct first proposals | Incorrect first proposals | No proposal |
|---|---:|---:|---:|
| Original frame-trained completion head | 26 | 12 | 10 |
| Threshold recalibration only | 17 | 2 | 29 |
| Ordinary continuation control | 21 | 8 | 19 |
| Added first-stop training loss | 21 | 10 | 17 |

Forty-four trajectories reached the pose tolerance at least once. The prior 12-case advantage did not hold. The ordinary continuation candidate is withdrawn, and the new sequence objective is not supported as an improvement. Threshold recalibration reduces false stops but misses many reachable completion opportunities. The scientific conclusion is a development generalization failure of the completion estimates, not a successful increase in full-agent task success. All original artifacts and failed variants are retained. No new Astra or language-task evaluation was run.

The next bounded local stage is `outputs/semantic_completion_v3/dense_20261021`. It collects 24 balanced-goal cases for eight new training seeds 96421–96428 and the same two development validation seeds 96311/96312. The frozen student continues for up to 900 steps, retaining reaches, drift and failures. In addition to the existing compact history and proprioception, it saves 96-by-96 RGB-D object regions, normalized crop boxes and a sensor-derived visibility flag. Regions are selected from RGB color evidence; no simulator pose, ground-truth bounding box or hidden symbol layout determines cropping. A full-frame fallback is explicitly marked when no colored region is visible. Object quaternion and six-goal errors are stored as separate supervision fields only.

The initial RGB crop was inspected and contains the colored cube and symbol detail. Two crop/depth contract checks passed. V2 has seven sensor/gate/loss tests passing. New collection code, reused sensor dependencies and frozen asset hashes are saved or checked. The student, teacher, action scale and success tolerance remain unchanged. Richer visual input is a hypothesis to test, not yet a demonstrated fix. Only collection is currently running; a new V3 model-training job has not been launched. There is no scheduled monitor and no external API traffic.


### Dense collection completed and matched training launched — 2026-10-21

The ten-shard collection finished at 15:54:37 CST. It produced 14,062 training frames from 192 trajectories and 3,717 validation frames from 48 trajectories. All ten shards passed finite-tensor, RGB-D shape and quaternion normalization checks. Recomputed quaternion errors matched saved labels within 2.4e-7 radians. An RGB region was found in 17,778 of 17,779 frames. The remaining frame retains the full-frame fallback and a false visibility flag. These sensor checks establish data consistency, not task success or a guarantee of ideal cropping for every frame. Recorded drop endings were 53 training cases and 10 validation cases, with no success termination in this collection.

Completion labels at 0.16 radians are present for all six goals. Training positive counts are [279, 407, 772, 324, 1416, 602]; validation counts are [85, 126, 177, 104, 544, 73]. The uneven frequency remains a limitation, particularly for the sixth validation goal. Full counts and audits are under `outputs/semantic_completion_v3/dense_20261021/collection_summary.json` and each shard's `sensor_audit.json`.

The newly launched finite local pipeline is `outputs/semantic_completion_v3/training_20261021`. It caches the frozen student features, then fits compact and dense variants for 60 epochs each. Both variants receive all six candidate language queries and use the same training examples, sample order, optimizer, loss weights and budget. Both have a goal-independent pose auxiliary, trained from simulator quaternion labels. Only the dense variant additionally reads 96-by-96 RGB-D crops, measured crop boxes and visibility. No pose labels or hidden symbol layouts enter inference. Adding the dense branch also adds parameters, so this pilot alone does not separate visual information from model capacity.

Checkpoint selection uses balanced completion BCE over positive, near-negative and far-negative groups on calibration seed 96311. The stopping threshold is then fitted to first-stop outcomes on that seed, requiring at least eight proposals and at least 90% precision. Audit seed 96312 is evaluated after checkpoint and threshold selection. A passing offline gate still requires shared-trajectory live validation before any promotion. Both seeds are reused development seeds, not untouched test data. The original student, controller, action scale and 0.16-radian full-quaternion success criterion remain frozen.

Four sensor/model contract tests passed, including orientation convention, quaternion sign invariance, goal-query equivariance and independence of the compact arm from dense pixels. Training results are not yet available at launch. This pipeline does not automatically launch cloud calls or formal Astra episodes. The explicit image-transfer confirmation remains pending. No scheduled monitor was created.


### Dense training completed, offline improvement below admission gate

Both matched 60-epoch fits completed normally. Both selected epoch 5 using calibration BCE. All 17,779 collected frames were cached successfully. The compact arm selected threshold 0.975 and the dense arm selected threshold 0.7, each using only calibration seed 96311. On calibration they made 19/19 and 21/21 correct first proposals respectively.

On the separate recorded development audit seed 96312, both observe exactly the same 24 trajectories:

| Variant | Correct first proposals | Incorrect first proposals | No proposal | Proposal precision |
|---|---:|---:|---:|---:|
| Compact features with pose auxiliary | 12 | 4 | 8 | 75.0% |
| Added dense RGB-D with the same losses | 18 | 3 | 3 | 85.7% |

Twenty-one audit trajectories reached a sampled state inside the 0.16-radian tolerance. The dense model improves correct-first-stop coverage on this development split. Median auxiliary orientation error also decreased from 0.1654 to 0.1281 radians. Neither model passes the predeclared 90% first-proposal precision gate. Neither is promoted, and neither has been evaluated as an autonomous Astra stopping controller. These counts must not be interpreted as new whole-task success rates. Thresholds were not retuned after seeing audit errors.

Results are under `outputs/semantic_completion_v3/training_20261021/{compact,dense}/result.json`; the concise decision record is `review.json`. The finite local training pipeline has ended. No model training or new Astra evaluation remains running from this pipeline. The next development priority is near-threshold uncertainty and calibration transfer, using training/calibration data rather than tuning to the three observed audit mistakes. A subsequent frozen candidate still needs live shared-trajectory validation on at most two seeds.


### Completion evidence, physical retention, and historical scoring

A descriptive timing audit of the frozen dense model is saved in `outputs/semantic_completion_v3/training_20261021/stop_timing_diagnostic.json`. All three incorrect first proposals occurred before the first sampled true reach. Their proposal/first-reach steps were 70/80, 360/400 and 50/380. The last failure was not merely a marginal threshold crossing. These audit examples are explanatory evidence, not additional training or calibration data.

The 18 correct hypothetical first stops were evaluated on recorded trajectories where the frozen student continued without acting on the proposal. At +10, +30 and +60 control steps, respectively 10, 9 and 5 of those 18 trajectories were still within 0.16 radians. All 18 had a sample at each offset. At 30 Hz these offsets are about 0.33, 1 and 2 seconds. This measures pose retention under the original continuing policy, not drops and not a counterfactual hold controller. It establishes that an accurate instantaneous completion prediction does not demonstrate stable physical retention. The dense head has not yet controlled stopping or holding in a live rollout.

The historical full-agent score terminated on a correctly grounded execute/verify reach using evaluator ground truth (`scripts/astra_runtime_v2/evaluate_learning.py`). It also allowed early agent stop failures. A better completion model could potentially prevent premature stop failures when coupled to continued execution, but it cannot automatically improve successful reach episodes that the old evaluator already ended. The new autonomous-stop score requires the agent to declare completion. Rates from these two scoring protocols must remain separate. Any proposed learned stopping/holding integration must show both reach and autonomous completion with matched controls, plus physical retention as a separate measurement.

Next useful mechanism comparison: keep goals, student and thresholds frozen, and compare continued execution with a sensor-triggered transition to stabilization and renewed verification. Measure whether a correct prediction leads to a correct declaration and whether stabilization retains the pose. Do not claim that continued-policy drift proves a hold intervention would succeed. A production candidate must first resolve premature stops and pass live validation.


### Code publication verification — 2026-10-21

The source publication includes the accumulated asynchronous-agent, waiting/recovery and completion-learning experiments, together with their protocols and tests. All 130 pending Python source files parsed successfully. All 57 tests across 17 changed test files passed in the local Isaac Lab Python environment. Some feature-contract tests require local frozen checkpoints and collected data, so a source-only clone does not include all test fixtures. These checks do not replace simulation evaluation or establish a task-success improvement. Local authentication, datasets, checkpoints, raw reports and media are not part of this source commit.
