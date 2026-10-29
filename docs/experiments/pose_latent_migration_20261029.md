# Pose-latent student migration: first training run

> 最新状态：新增实验已于 2026-10-29 按用户要求暂停。下文运行状态是历史记录。额外 student 训练已完成，实际 CNN 候选在新开发集达到 40/40 底层到达，但未正式部署。请以[当前项目说明](../project_status_and_usage_20261029.md)为准。

## Latest completed bare baselines and active student training

Updated 29 October 2026, 08:34 UTC. The two bare VLM conditions completed all eighty new raw trials at 08:10 UTC, with valid physical logs, no service exclusions and all forty initial states/goals matched to the parent Astra condition. No waiting actor, execution actor or learned completion checkpoint was loaded. The common frozen student, CNN pose predictor and measured stop protocol remain unchanged.

| Main system comparison | Raw success / 40 | Raw drops | Common eligible success / 39 |
| --- | ---: | ---: | ---: |
| Astra full method | 29 | 10 | 29 (74.4%) |
| Bare fine-tuned VLM | 27 | 9 | 26 (66.7%) |
| Bare rule VLM | 14 | 15 | 13 (33.3%) |

The parent Astra service-error key 183102011 is excluded symmetrically from all three arms, preserving raw forty-slot denominators. Astra versus bare fine-tuned VLM has nine exclusive successes and six exclusive failures, with unadjusted exact paired McNemar p=0.60724. Astra versus bare rule VLM has eighteen exclusive successes and two exclusive failures, p=0.00040245. The main comparison is between complete systems, not intrinsic model capabilities. The observed Astra advantage over fine-tuned VLM is not yet statistically established.

The bare fine-tuned model has nine drops, two pose false stops and two wrong semantic bindings. Its seed counts are 14/20 and 13/20. The bare rule model has fifteen drops, nine wrong bindings and two horizons, with seed counts 5/20 and 9/20. The retained auxiliary fine-tuned-with-modules result is 30/40 with five drops, versus bare 27/40 with nine drops; its paired p=0.58105. The tuned model therefore retains substantial capability without our control modules. The contribution of those modules cannot be inferred by attributing the entire old 30/40 to control adaptation.

The two equal-budget successor student jobs actually started at 08:11 UTC on GPUs 3 and 4, after both predecessor exports passed. At the latest log check they reached 149/400 simulated-noise updates and 111/400 deployed-CNN-pose updates. Their progress files are current and no fatal or nonfinite errors were found in the inspected logs. These are ongoing training observations, not new closed-loop success rates. Four fresh forty-trial development comparisons follow training automatically. Existing formal checkpoints and results remain frozen and no successor is deployed automatically.

The complete supplemental summary, two new arms plus retained parent records, and manifests are archived in `outputs/pose_latent_student_v1/pure_vlm_addendum_20261029/formal_export/` and the paper's `data/pure_vlm_addendum_20261029/formal_export/`.

---

## Current interpretation and baseline correction, 29 October 2026

This section supersedes shorthand labels and pending-status sentences in the chronological record below. The original eight-condition release remains immutable. Its `finetuned_vlm_full` result is **fine-tuned VLM with learned waiting v1 and learned execution recovery v6**, not a bare fine-tuned baseline. The same applies to `rule_vlm_full`. Both use the common measured-pose stop gate and no learned completion model.

The intended main comparison is now **Astra with the proposed control modules versus bare fine-tuned VLM and bare rule VLM**. The two bare conditions retain the same frozen latent student, RGB pose predictor, VLM weights, model-specific latent/readability heads, and common measured stop protocol. They use `maintenance=continue` and no waiting, execution, learned-completion or calibration checkpoint. The existing VLM observation schedule remains part of the baseline planner. The initial eight-condition common-controller comparison is retained as auxiliary evidence for isolating provider differences. The new main comparison measures complete systems and does not by itself establish intrinsic Astra model superiority.

Two additional forty-trial conditions, `finetuned_vlm_bare` and `rule_vlm_bare`, passed real-runtime preparation with zero calls and zero trials and were frozen under a separate addendum at 07:10 UTC. Its release SHA is `c74dfbbceaa00e27db4b2add0624dd1efe94d3100be6e585c65cbeb976fa1a7e`. They use the same two reserved seed streams and immutable parent release `07922f1a…`. The addendum permits only removal of the learned controllers. It preserves source identities apart from its release guard, all weights, physical settings, success and service-exclusion rules. This correction was declared after original results existed and is not represented as the initial preregistration. No new weights are selected from those outcomes. Every raw trial is retained, including errors, and paired initial states and goals must match exactly before comparison.

### Verified completed raw results

| Current frozen condition | Success / 40 | Drops | Own service exclusions |
| --- | ---: | ---: | ---: |
| Astra, fixed slowdown without execution recovery | 23 | 16 | 0 |
| Astra, learned waiting only | 26 | 11 | 1 |
| Astra, learned execution with fixed slowdown | 28 | 8 | 1 |
| Astra, learned waiting and execution | 29 | 10 | 1 |
| Fine-tuned VLM, learned waiting and execution | 30 | 5 | 0 |
| Rule VLM, learned waiting and execution | 16 | 12 | 0 |

The original suite completed all 320 raw trials at 07:12:30 UTC. All completed conditions pass the physical-log checks. The original common-controller provider panel has 39 matched eligible slots: Astra 29/39, fine-tuned VLM 29/39, rule VLM 15/39. Astra versus fine-tuned VLM has an unadjusted exact paired McNemar p-value of 1.0. Thus raw 29/40 versus 30/40 must not be interpreted as a one-trial provider advantage after the frozen common exclusion.

The Astra factorial excludes three service-error keys symmetrically and has 37 matched eligible slots: full 27/37, fixed slowdown 20/37, waiting only 25/37, execution only 26/37. Full versus fixed has unadjusted exact paired p=0.21004, full versus waiting p=0.79053, and full versus execution p=1.0. These descriptive gains do not establish significant component improvement or a benefit from combining both modules. The matched provider panel and factorial have different, explicitly stated exclusion sets.

The two new bare conditions do not yet have a completed result. The 33/40 fine-tuned VLM result cited earlier belongs to repeated development evaluation with learned control, not a bare formal result. Final original summaries, all 320 trial records and manifests are archived under `formal_serialization_repair_20261029/completed_export/` in the local migration outputs and `data/pose_latent_migration_formal/completed_export/` in the paper repository.

### Why the tuned VLM result is relatively high

The model is a task-adapted Qwen3-VL-4B with LoRA and a learned continuous goal/readability head. It first uses 4,096 teacher-view training frames with 1,024 validation frames, then 4,800 student-view training frames with 2,400 validation frames and mixed replay for a further 240 updates. The deployment grounding input is RGB plus the instruction. Pose/depth are training-label sources, not additional grounding inference inputs. The latest goal head is `69beb7…` and LoRA adapter `d5a060…`.

The evaluated scope is one cube with six fixed letter/color associations. Memorization of this mapping is not ruled out. Training, validation, development and formal seeds differ, but this does not constitute unseen-layout, unseen-object or open-vocabulary validation. The old high score includes shared learned control. It cannot be decomposed into a model-only score by subtracting an assumed controller benefit. On the repeated development set, the latest VLM has 33/40 with waiting only and still 33/40 after adding execution v6, with different failure types.

Observed formal client latency is much shorter for the local fine-tuned head: median 0.113 seconds versus 5.142 seconds for Astra. Physics continues during requests, so the waiting exposure differs. This observation is a system-level timing difference, not a controlled inference-cost or reasoning-capability comparison. Neither provider has established superiority in the new pose-latent series.

### Completed training and the queued successor

The current latent student completed 200 interface updates and 400 teacher DAgger updates. Its formal RGB-CNN diagnostic reaches 35/40 targets with five drops, compared with 39/40 and one drop for the colleague's original pose-goal student. Those are low-level reach diagnostics, not autonomous language success. The ten-point observed transfer regression remains unresolved.

The task-specific VLM adaptation and waiting/execution actors are trained. Learned completion candidates were evaluated during development and are not included in the selected full method. The CNN pose estimator is reused from the colleague's release and has not been retrained here.

A separate successor study queues equal-budget extra DAgger with simulated pose noise and with the deployed frozen CNN measurements, followed by four forty-trial development conditions on fresh seeds. Its training plan and weights remain unchanged. It now waits for both the original 320-slot export and the new 80-slot bare-VLM addendum so the baseline remains frozen and avoids concurrent successor-training load. The revised finite supervisor is PID 684127. It has not started new training. A narrowly registered small resident GPU context may coexist only under the documented low-memory/low-utilization checks; no other user's process is stopped. This scheduling amendment is retained separately from the original plan.

Addendum code, launch and subsequent release evidence: `outputs/pose_latent_student_v1/pure_vlm_addendum_20261029/` in the project. Original raw records remain under `outputs/pose_latent_student_v1/formal_completed_conditions_20261029/` and the paper's `data/pose_latent_migration_formal/`.

### Bare-baseline launch confirmation

The two new bare conditions began at 07:12:46 UTC on `.20`: fine-tuned VLM on physical GPU 5 (worker PID 688084) and rule VLM on GPU 6 (PID 688106). The finite supervisor is PID 684926. Live manifests confirm `continue`, student/student control, no actor, execution or learned-completion checkpoint, and the unchanged measured stopping rule. At 07:20 UTC, thirteen fine-tuned and three rule trials had completed. Every recorded initial-state digest and requested goal matched the original Astra arm. These partial counts are progress only, not interim performance claims. The successor training supervisor is alive at `waiting_for_bare_vlm_addendum` and has launched no training job. Its additional eighty-trial gate is intentional.

---

> **2026-10-29 validity correction:** A full-log audit found MJWarp broadphase contact-capacity overflow in several earlier development evaluations, including learned control, fixed slowdown, low-level RGB and actual Astra/VLM runs. The affected full-condition success rates and their comparative claims below are withdrawn pending reruns. Raw logs are retained. Training/collection logs audited so far contain no such warning. All new arms use nconmax=4096 and njmax=1024 through the official preset override, with immediate abort on capacity warnings. The frozen success definition, 40 raw trials and formal seeds are unchanged.


## Scope

Use the colleague's released RSLCube and Revo3 student from commit
`93317c16e3f6ab6b35c7d36174c5b6c7fe090e80`. The new student receives
109 physical/history features, 7 current pose features, and a continuous
768-dimensional goal latent. Target and relative-goal quaternions are excluded
from its input. They remain available to the simulation and training reference.

The first two stages use the released simulated perception noise. They do not
run the RGB pose estimator. Training diagnostics must not be presented as
rendered-vision performance or autonomous semantic task success.

## Runtime and provenance

- Host: `brainco@10.3.100.20`, independent root
  `/home/brainco/agentic_rotation_newton`.
- Isaac Lab v3.0.0, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.
- Isaac Sim 6.2.0.0, Newton 1.6.1, Warp 1.17.0, Torch 2.11.0+cu130.
- User approved the NVIDIA license and project/checkpoint upload.
- Python development headers were extracted into the independent environment.
  No system package installation was needed.
- `prepare_pretrained.py` verified the 43 published file checksums, relocated
  asset references, and verified that student model/optimizer tensors were unchanged.
- Original student SHA256:
  `da23adb565d8f26c4c73e3b12d5bc1f53747dcb6d4e7cccf1cbaf169a2604016`.
- Prepared student SHA256 (path metadata changed):
  `022f963f3fa177cc84d3ca14888ba2872e594577ccf1614713834eb10f9336f1`.
- Teacher SHA256:
  `fc3a2e23b13eef53d8c192672436bc949dece8c5afa423c3bf7138e85e9e52c0`.

## Completed checks

| Check | Configuration | Observed result |
| --- | --- | --- |
| Official runtime smoke | Cartpole, Newton, 2 environments, 32 steps | Exit 0 |
| Original student smoke | 8 environments, 360 steps, seed 182901, released noise | Exit 0, 81 target-counter increments, 4 out-of-reach terminations |
| Latent interface pilot | 32 environments, 3 updates, 2 epochs | 4,320 environment steps, action MSE 3.0684 → 2.6281, checkpoints saved |
| Teacher DAgger pilot | 32 environments, 2 updates, 2 epochs | 2,880 environment steps, action MSE 2.9899 → 2.1892, checkpoints saved |

The original smoke has no autonomous stop decision. Its target-counter increments
are repeated reorientation events, not a percentage of successful language trials.
No contact/constraint capacity overflow was reported in these runs.

## First training experiment (completed)

Remote output root:
`/home/brainco/agentic_rotation_newton/outputs/pose_latent_20261029_v1`.

Physical GPU 1 was selected after checking free memory. It is shared with existing
workloads. The run uses 128 environments and optimization batches of 32 sequences.
Other processes were left running.

1. `interface`: 200 updates × 45 steps × 128 environments = 1,152,000 steps.
   Only the goal projection and new latent input columns train. Original student
   actions drive the rollouts and provide labels. Inherited physical weights and
   normalization statistics stay frozen.
2. `adapt`: 400 updates = 2,304,000 additional steps. Initialize from interface
   update 200 and unfreeze the student. The original PPO teacher labels the
   rollout states. Teacher action probability follows `0.9 * 0.95 ** update`.

Both stages use seed 182801, 4 epochs, zero-weight-decay Adam, and the audited six
cardinal cube goals. New checkpoints are saved after the first update and every
20 updates. The launcher checks successful stage completion before advancing.
An unsuccessful process exits with its stage recorded in `process_status.txt`.

Launcher: `scripts/pose_latent_student_v1/run_stages.sh`.
Supervisor PID at launch: `105062`.
Supervisor log: `/home/brainco/agentic_rotation_newton/logs/pose_latent_20261029_v1_supervisor.log`.
Stage logs and `progress.json` are under the output root.

The first three interface updates completed with action MSE
2.7546 → 2.2001 → 1.9676. This is a startup snapshot, not a convergence claim.

## Planned admission sequence

Evaluate the migrated student without teacher actions on development seeds
182901 and 182902. Check goal binding and compare physical reach/drop behavior
with the released student. Then validate calibrated RGB pose-estimator feedback
and adapt the agent's waiting/recovery/completion modules.

Formal evaluation remains governed by
`configs/evaluation/pose_latent_migration_v1.json`: 40 raw trials per condition,
20 each on two held-out seeds, with the existing success definition. No formal
evaluation has started. Results from the previous backend remain separate.

## Completed student and RGB development results

Both stages completed with exit 0: interface 200 updates and DAgger 400 updates,
3,456,000 environment steps in total. The final new checkpoint is
`adapt/latent_student_00400.pt` in the remote training output root.

Each entry below contains 40 raw trials, 20 on each development seed 182901 and
182902. No trial began already within the orientation threshold.

| Feedback | Released student reach / drop | New latent student reach / drop |
| --- | --- | --- |
| Released simulated perception noise | 39 / 1 | 39 / 1 |
| Actual calibrated RGB-CNN estimate | 40 / 0 | 39 / 1 |

Reach means full-quaternion error at most 0.16 rad before an original
`object_out_of_reach` termination. These diagnostics preserve the released
20-second environment timeout and end at first reach. They do **not** measure
the frozen 120-second autonomous semantic success criterion.

The image pipeline renders the original Gaussian-splat object with live,
depth-ordered black hand occlusion at 120×120. The CNN reads those pixels.
It does not receive simulator object pose, and synthetic pose noise is not
added after CNN inference. This is not full-scene RGB.

Local evidence:
`outputs/pose_latent_student_v1/development_eval_20261029/`.
The paired audit found identical recorded roots, joints and raw action buffers.
The first runner also hashed the previous episode's stale processed-action
output cache, producing hash differences after the first episode. In the pinned
Isaac Lab implementation that cache is replaced by `process_actions` before
`apply_action` or a physics step. EMA memory is reset to the identical joint
positions. The audit records this limitation; future runs explicitly record and
check EMA memory and exclude the overwritten output cache from the causal hash.
The initial raw artifacts are retained.

## Frozen CNN diagnostic

A separate teacher-driven collection contained 1,024 calibrated RGB frames.
The released CNN measured mean translation error 2.856 mm, mean rotation error
1.964 degrees and rotation p95 4.032 degrees. 99.22% met the diagnostic
10 mm / 10 degree pose criterion. This is pose accuracy, not task success.
Only one sample had at least 50% hand occlusion, so this collection does not
establish robustness under severe occlusion.

Evidence: `outputs/pose_latent_student_v1/rgb_diagnostic_20261029/`.

## Letter grounding migration

Training capture: 4,096 frames, seed 182811. Validation capture: 1,024 frames,
seed 182812. Conservative front-facing, projected-region and hand-depth rules
produce visibility labels for T/E/R/L/S/H. Ambiguous views are omitted.
Forty-eight stratified tiles from each split were inspected by the assistant.
These remain geometric supervision proxies, not human annotated readability.

Qwen3-VL-4B QLoRA warmup completed 240 updates on the local RTX 3090. The previous
LoRA and visual/readability trunk were reused, while the continuous goal head's
basis and projection were rebuilt for the new student's exact 768D bank.

On the fixed balanced validation subset:

- 60 of 60 visible targets had the correct nearest goal.
- The fixed 0.5 readability threshold accepted 51 of those 60.
- It accepted none of the 60 invisible-target examples.

The checkpoint is
`outputs/pose_latent_student_v1/grounding_qlora_20261029/step_00240/`.
The learned continuous vector is passed directly to the student. Nearest-goal
classification is an offline diagnostic, not an inference lookup.
The fixed letter/color layout permits memorization. These numbers do not
demonstrate unseen-layout grounding or autonomous success. Student action
consistency and closed-loop agent validation are still pending.

## Maintenance and completion adaptation

New collection uses the final student, actual RGB pose estimation, and scripted
1–6 second service delays. Physics continues throughout. It mixes continued
student execution, fixed 0.15 interpolation, latched measured joints, and
teacher anchor holding to expose stable and failing transitions. Original
physics and domain randomization remain bound to the released configuration.

At each saved physical state, all six goal errors and teacher action queries
are labels only. Model features contain measured pose, proprioception, student
belief/proposal, a measured request anchor and public request state.
Ground-truth object orientation and teacher outputs never become model inputs.

Remote collection:
`/home/brainco/agentic_rotation_newton/outputs/control_collection_20261029_v1`.
Training split has completed: 7,680 saved states, 23 out-of-reach events across
the mixed behavior collection. These are deliberately varied training
trajectories, not 23 failed benchmark trials. Independent validation collection
is running. Both splits have distinct seeds.

A dependent training job is queued at
`/home/brainco/agentic_rotation_newton/outputs/control_modules_20261029_v1`.
It will fit a lightweight maintenance actor against teacher actions and a
goal-conditioned completion head against the paired goal labels. Calibration
and reporting use separate whole trajectory groups. Offline action error or
completion precision is insufficient for deployment admission; a subsequent
closed-loop comparison is required.

## Astra interface and remaining work

The new Astra wrapper reuses the existing identity-pinned, separately consented
transport and requests the T/E/R/L/S/H protocol. Both high-level models receive
the same 120×120 RGB with the same image-only crop. The legacy bridge defaults
remain 512×512. This adapter has been compiled but has not made a live request
for the new backend.

Astra returns structured tool calls. Their declared color is mapped locally to
the frozen student latent. No hidden Astra embedding is extracted. The
fine-tuned VLM instead predicts a continuous latent through its learned head.

Remaining admission work:

1. Complete maintenance/evidence fitting and inspect failure cases.
2. Run closed-loop module comparisons, including waiting and post-delivery
   recovery. Compare learned completion with a measured-pose baseline.
3. Validate the actual asynchronous Astra, fine-tuned VLM and rule VLM loops.
4. Port the original 0.24 m drop predicate to the documented grasp reference
   frame and preserve the frozen 120-second success protocol. The colleague's
   generic out-of-reach termination is not a substitute.
5. Freeze the adapted implementation and then run 40 raw trials per formal
   condition and module ablation. Keep these results separate from historical
   backend results.

## First maintenance/evidence fitting completed

Collection finished with exit 0 on both splits: 7,680 training states and 3,840
validation states. The validation collector recorded 17 out-of-reach events
across 111 automatic resets. These mixed trajectories are not benchmark trials.
The follow-on fitting completed 1,200 updates with exit 0.

Independent offline action imitation errors:

| Phase | Fixed slow / original execution baseline | Learned maintenance actor |
| --- | --- | --- |
| All saved states | 0.5104 | 0.3220 |
| Pending request | 0.7536 | 0.4539 |
| Execution | 0.1200 | 0.1102 |

Errors are MSE against the frozen teacher's clipped actions. This approximately
37% overall reduction is not evidence of improved closed-loop success or reduced
drops. The actor remains a candidate until online comparison.

Completion reporting uses 2,122 states from entire trajectory groups excluded
from threshold calibration:

| Completion method | Accepted queries | Correct accepted | Precision | Recall |
| --- | --- | --- | --- | --- |
| Learned head, calibrated threshold 0.98897 | 576 | 560 | 97.22% | 75.78% |
| Calibrated RGB pose + fixed 0.16 rad comparison | 731 | 721 | 98.63% | 97.56% |

The pose-only baseline uses the predicted object orientation, camera/base
calibration and known target orientation. Simulator object orientation is used
only for scoring. Both methods are scored on exactly the same report groups.

**The first learned completion head is not promoted.** The measured-pose
baseline has both higher precision and higher recall here. Passing the learned
head's standalone 95% precision screen does not establish value over that
baseline. Old-backend module advantages must be re-established after migration.

Artifacts:
`outputs/pose_latent_student_v1/control_modules_20261029_v1/fit/`.
The student and VLM hashes are also fixed in
`outputs/pose_latent_student_v1/candidate_manifest_20261029.json`.
No Git commit or push was performed in this work phase.

Next priority is the 120-second uninterrupted agent runtime and closed-loop
maintenance comparison, followed by actual Astra / fine-tuned VLM / rule VLM
runs. The frozen success definition and 40 raw trials per condition remain
unchanged.

## Continuous agent migration and runs started on 2026-10-29

The deployment runtime now removes the released training task's success-count reset and changes the episode cap to 120 seconds. The only active termination terms are the frozen 0.24 m nominal-grasp displacement predicate and the 3600-step time limit. Physical assets, solver settings, contact geometry and physical randomization remain bound to the published environment. The student keeps the released 20-second remaining-time feature scale, clamped to zero after 600 steps. This extrapolation in episode duration still requires closed-loop evidence.

The continuous sensor path uses the actual frozen RGB CNN at every step. The student receives proprio109, measured current pose7 and the continuous latent768. Six measured orientation errors are available to the public verifier. True object state is used only by the independent scorer and diagnostic logs.

### Development batches

All conditions below use two seeds and 20 raw trials per seed. Formal seeds 183101 and 183102 remain untouched.

| Batch | Conditions | Seeds | Planned trials |
|---|---|---|---:|
| Injected waiting delays with correct scripted bindings | continue, fixed_slow, learned maintenance | 182911, 182912 | 40 each |
| Real semantic inference and continuous physics | Astra, fine-tuned VLM, frozen rule VLM | 182921, 182922 | 40 each |

The first batch is control isolation. It is not an Astra or VLM success result. Delays of 1–6 seconds are injected in simulation time, including initial startup and later requests. The first 17 matched trials have exactly equal initial physical-state hashes across continue and fixed_slow.

The real-inference batch is a new-backend development baseline with fixed slowing during pending requests and a common measured-pose stopping gate. It is not yet the proposed full method. A model must declare the requested symbol binding before the local gate can stop. The independent scorer still requires the correct semantic target, full quaternion error at most 0.16 rad and no preceding drop. A wrong model binding can produce a false stop and remains a failure.

Real requests run in a worker while physics continues. Responses are delivered only after both the request completes and at least ceil(actual_request_latency * 30) physical steps elapse. Simulation can run below real time on shared hardware. This pacing provides the required physical-duration exposure but does not establish real-time controller throughput.

The fine-tuned VLM's returned vector enters the student unchanged. Its nearest skill is used only for public orientation verification and logs. The rule VLM generates JSON from the same frame and crop and maps the declared color to the frozen bank. Astra produces bounded tool calls mapped locally to that same bank. These are not Astra hidden-state embeddings.

Astra identity remains pinned to the already consented account. OAuth credentials remain local. Only a run-scoped bridge key was copied to the authorized simulator, and tunnels listen on loopback. The new batch is capped at 240 actual Astra calls and 40 sessions, with transport retries disabled.

### Early observations and limitations

The initial Astra launch failed before any model request because the new remote project copy lacked four existing client modules. These were copied and the corrected run uses a separate v2 output directory. The failed directory is preserved.

Live Astra calls have now completed. The first episode included a wrong cyan-to-T declaration. The observed image and raw answer are saved in outputs/pose_latent_student_v1/live_development_20261029/first_astra_episode. The declared goal is not repaired using the known cube layout. This provides a concrete semantic grounding failure to analyze; it does not yet support an aggregate comparison.

The learned maintenance actor is pending closed-loop admission. The learned completion head is not promoted because the pose-only baseline is stronger on the same held-out split. We do not count offline loss reduction as success-rate improvement.

Current run roots, supervisor PIDs, queue dependencies and service locations are recorded in outputs/pose_latent_student_v1/setup_status.json. Final comparisons must include raw trial counts, matched service exclusions, per-seed counts, startup/wait/execution drops, false stops and failures to complete.

## Completed first maintenance comparison and second training round

The first three control-isolation conditions each completed 40 raw trials on seeds 182911 and 182912. The semantic bindings in this experiment were scripted and correct.

| Controller | Success | Startup drops | Execution drops | False stops | Success by seed |
|---|---:|---:|---:|---:|---|
| Continue student | 33/40 | 4 | 2 | 1 | 18/20, 15/20 |
| Fixed slowing while pending | 34/40 | 3 | 1 | 2 | 16/20, 18/20 |
| Learned maintenance during waiting and execution | 33/40 | 1 | 4 | 2 | 19/20, 14/20 |

No waiting-after-binding drops occurred in these conditions. The learned controller reduced the observed startup drop count but did not improve total success. The first seed's 19/20 result did not carry over to the second seed. The full maintenance candidate is therefore not admitted as an improved method.

A separate 40-trial ablation now applies the same learned actor only while a request is pending. After delivery it uses the unchanged student. This tests whether continuing the residual during execution contributes to the extra execution drops.

### Pose correction candidate

A bounded pose-error correction and uncertainty head completed 1200 updates using the existing development captures. On the previous reporting trajectory groups, the raw and calibrated geometric baselines made 10 false accepts, while the learned correction made 8, with the same 721 correct accepts. Near-target absolute error increased slightly from 0.01277 to 0.01311 rad. This is a small offline change on a reused development split and is not evidence of autonomous improvement.

The selected correction is undergoing its own 40-trial control ablation. During audit, corresponding runs showed identical initial physical state and first measured pose but slightly different first post-action orientations before the stop gate could act. The differences were roughly 1e-6 rad and could grow into different contact trajectories. The old logs did not save the first action and full noisy proprioception, so the numerical source has not been localized. The official installed MJWarp manager does include solver-state reset handling. A loaded-model reference with the original measured-pose decision and additional input/action traces is queued. Small success-count changes must not be attributed solely to the stop module.

### Continuous DAgger v2

The next collection uses 10 environments, dense sampling at every control step, the continuous 120-second protocol and the frozen 0.24 m drop predicate. It removes training success-counter resets. Behaviors include continued student execution, fixed slowing, measured-joint latching, the frozen teacher holding the request anchor and the learned maintenance actor.

The frozen timing profile contains 41 completed development Astra requests, from 4.45 to 11.03 seconds. This extends the original 1–6 second training coverage. These timings are replayed without making new upstream model requests during collection. True orientation and frozen-teacher actions remain labels only.

The pipeline plans 48,000 training states and 24,000 validation states on separate seeds 182861 and 182862, then 2400 actor adaptation updates starting from the first maintenance checkpoint. Both full-time and waiting-only use of the adapted actor have separate 40-trial evaluations queued. No running evaluation weights are overwritten.

### Real-model development findings so far

Astra and the fine-tuned VLM are now producing real online episodes while physics continues. The frozen rule VLM is queued afterward. Current partial results do not establish Astra superiority. Observed Astra failures include wrong letter/color grounding and startup drops. One fine-tuned VLM failure recognized the correct nearest latent but rejected the view as unreadable, continued inspecting and later dropped.

A small wording diagnostic used the same six readable validation images with three instruction templates. All 18 predictions were accepted and correct. This excludes those wording variants as the cause of that small diagnostic's failures; it does not prove broad language generalization.

Raw logs are retained. A reporting-only correction distinguishes a drop before any color skill was delivered from a later inspection drop. Earlier online logs used semantic binding availability for the startup category, which was too broad. The corrected category is reconstructed from recorded public controller modes. Success and failure outcomes are unchanged.

The corrected-stop control subsequently completed 40 trials: 30 successes,
10 drops and no false stops, split 17/20 and 13/20. This is below the original
33/40 and is not admitted. Removing false stops alone did not improve episode
success. The numerical comparison concern above remains unresolved.

Continuous DAgger training capture completed 48,000 dense states with 23 drop
events and 27 resets. Validation capture has started on its separate seed.
The waiting-only maintenance ablation is now running.

## Continuous DAgger v2 and capacity-corrected reruns

- Dense continuous data: 48,000 training and 24,000 validation states, five behavior sources, measured Astra latency profile, 2,400 fitting updates. Training completed with exit 0.
- Actor validation teacher-action MSE: 0.22770 versus 0.33916 for the baseline. Waiting MSE: 0.33511 versus 0.51633. These are offline imitation metrics, not autonomous success.
- New evidence head: 3,854 correct of 3,861 accepted queries on the report split, precision 99.82%, recall 50.72%. Not admitted without closed-loop evaluation and a pose-only reference.
- Capacity-corrected queues cover continue, fixed slowdown, v1/v2 maintenance both full-time and waiting-only, loaded-reference numerical diagnostic, original versus migrated RGB student, and the three real semantic providers. Each condition uses 40 trials, two seeds.
- Jobs and invalidated log paths are archived in outputs/pose_latent_student_v1/live_development_capacity_20261029/. Second-round weights are archived in outputs/pose_latent_student_v1/control_continuous_dagger_20261029_v2/fit/.
- Initial manifests confirm identical physical contract SHA, calibrated camera, object/hand assets, 30 Hz and increased capacities. No claim of Astra superiority or a controller gain is supported yet.

## Same-state goal-paired recovery training v3

The v2 collector saved six teacher actions per physical state, but actor fitting used only the currently active goal. The new ablation samples alternative correct goals only during execution. Pending requests retain the original hold-anchor labels. Inputs are rebuilt from the same measured sensors and the same recurrent history before the hypothetical goal switch. True state and teacher actions remain labels only.

Replaying 48,000 training and 24,000 validation states reproduces the original student proposals and beliefs exactly (maximum absolute difference 0). V3 uses the same v1 warm-start actor, 2,400 updates, seed 182871 and data as v2, with a separate generator for alternative goals. The completion-evidence weights remain tensor-identical.

On 10,776 validation execution states, evaluating all six hypothetical correct goals gives teacher-action MSE 0.12097 for v2 and 0.09964 for paired v3, versus 0.12117 for the student proposals. Shuffling only the explicit goal latent gives 0.12247 and 0.10366 respectively. These are offline teacher-action diagnostics, not autonomous performance. On the original actually executed goals, v3 MSE 0.09917 is slightly worse than v2 0.09588.

Closed-loop isolation uses the same frozen v1 waiting actor and separately substitutes v2 or v3 during execution. Two 40-trial conditions are launched/queued under the same capacity-corrected settings, seeds and autonomous stop rule. No v3 improvement is admitted until these finish.

## First-stop event audit and parallel semantic runs

The dense validation split contains 24,000 frames but only 63 same-goal segments with eligible stopping decisions (26 calibration, 37 report). We replayed the first stop at 3 Hz when the target was known and no request was pending, using whole-episode calibration/report groups. This is fixed-trajectory replay, not a closed-loop counterfactual.

On report groups, the original pose gate accepted 31 segments: 30 correct, one false stop at 0.16625 rad true error. The v2 learned evidence gate accepted 25 correct segments with no false stop, but missed six reachable segments. Calibration on first-stop events selected an internal pose threshold 0.156 and evidence threshold 0.997388. These accepted 31/31 and 30/30 correct report segments respectively. The small reused development split does not establish population precision. No calibrated threshold has been deployed, and the scored 0.16 rad definition is unchanged. The evidence module remains unadmitted.

The three queued real-model runs were deliberately reprioritized before any episode or request began. Their replacement runs are agent_astra_capacity_now_20261029, agent_vlm_capacity_now_20261029 and agent_rule_capacity_now_20261029 on GPUs 0, 1 and 5. All use fixed slowdown, 4096/1024 capacities, identical frozen 40-trial protocol and real measured provider latency. Workloads overlap other development jobs, so this is not a real-time throughput benchmark. Original empty queue records remain marked superseded.

## Capacity-corrected waiting-only candidate admission

All 40 scripted correct-binding episodes completed without contact-capacity warnings for the continue baseline and v1 waiting-only candidate. Continue: 30/40 (17/20, 13/20), eight drops and two false stops. V1 waiting-only: 38/40 (20/20, 18/20), one execution drop and one false stop. Initial state and target pairing are verified across all 40 trials. This supports a development candidate under injected 1–6 second delays; it is not an Astra end-to-end rate.

Three actual-provider waiting-only ablations, each 40 trials, are queued after each provider's fixed-slow baseline. They use the frozen v1 waiting checkpoint and the unchanged student during execution. The Astra arm has a separate 40-session, 240-call ledger on loopback port 18784 using the existing authorized identity. No formal seeds are consumed.

### Observation and driver scope of the semantic comparison

All providers receive the same calibrated RGB observation and an image-only crop. Their high-level interfaces differ and results must be described as agent-system comparisons, not isolated foundation-model rankings.

| Provider | Additional context available to the upper model | Observation-action driver |
|---|---|---|
| Astra | Within-episode image/tool history, current public skill/binding, remaining calls, RGB-CNN estimated orientation errors by skill and pose fit error | Model selects inspect/reorient/verify/stop/abstain tools |
| Fine-tuned VLM | Current instruction and image; the current service does not consume the Astra structured feedback/history | Learned continuous goal vector and readability, shared deterministic inspection schedule when unreadable |
| Rule VLM | Current instruction and image; no episode history at the model service | Generated readable-color JSON, deterministic lookup and inspection schedule |

All downstream local control and stop gates receive the same measured sensors. The upper-level feedback contains no simulator ground truth. Claims about the benefit of Astra feedback or memory require separate ablations; the three-arm result alone cannot isolate model capacity from access to history and feedback.

## Execution inventory at 2026-10-29 03:49 CST

The first-round latent-student interface/adaptation, QLoRA grounding, v1 maintenance, continuous DAgger v2 and goal-paired v3 training are complete. Completion correction also has a trained checkpoint, but remains unadmitted. The released CNN pose estimator is reused frozen. No additional unconditional training run is needed before the current comparisons resolve which failure to target.

A fresh remote audit found seven active simulation processes and six live queued supervisors on GPUs 0, 1 and 5. All four local-model/bridge endpoints responded successfully through their remote loopback connections. Three real-provider fixed-slow conditions are running; their waiting-only ablations start automatically after each corresponding baseline succeeds. The v3 execution comparison, RGB student comparison and loaded-reference diagnostic are also queued. All comparison conditions retain 40 raw trials and two seeds.

The authoritative current inventory is outputs/pose_latent_student_v1/live_development_capacity_20261029/current_queue.json. Older status entries are historical, and the contact-capacity invalidations take precedence over their numerical results. The full method has not been frozen or evaluated on the untouched formal seeds.

## Completion availability during asynchronous inference

The v1 waiting actor with the v2 execution actor completed 35/40 development trials (18/20 and 17/20), with two execution drops and three false stops. The physics logs passed. This does not improve on v1 waiting-only with unchanged student execution (38/40), so v2 execution is not promoted. The v3 paired-goal execution comparison is still running.

A real Astra failure at seed 182921, episode 3, correctly bound yellow to L and later dropped. During the observed trajectory, 165 control frames had the correct binding, a pending request and true orientation error at or below 0.16 rad. The minimum was 0.03011 rad. The original local gate explicitly excluded pending requests. These truth-based annotations diagnose possible missed opportunities; the old agent step log did not save measured per-frame orientation error, so they do not prove that the sensor gate would have accepted.

A separate 40-trial control condition, control_wait1_async_stop_20261029, enables the same 3 Hz measured-pose gate during pending requests after a semantic binding has been delivered. It uses the v1 waiting actor and unchanged student execution. Threshold, independent scorer, horizon and request budget are unchanged. This is a decision-availability baseline, not a new learned contribution. Default evaluator behavior is preserved and running source snapshots are unchanged. New logs record pre-action observation steps, measured binding errors and stop eligibility. Python compilation and shell syntax checks passed; the queued runtime result is pending.

Fixed-trajectory replay of completed control captures found no new false acceptance under this gate in those observed sequences, and one v2-execution drop had an earlier correct waiting-time acceptance opportunity. This is not a closed-loop improvement claim. The next result must come from the separately launched simulation condition.

Actual failed request images are archived in outputs/pose_latent_student_v1/live_development_capacity_20261029/failed_requests/. Inspection found both genuinely occluded letters and rotated visible letters rejected by the fine-tuned readability head. The next grounding adaptation should use independently seeded student-generated observations with audited labels, retaining teacher-data validation. Lowering the acceptance threshold globally would also accept unobserved or wrongly grounded goals and is not justified by these examples.

The full-time v2 control condition subsequently completed 36/40 without capacity warnings, with two execution drops and two false stops. It remains below v1 waiting-only (38/40). The asynchronous-completion control queue has now started Newton simulation; its manifest confirms the new availability flag and unchanged scored 0.16 rad predicate. Aggregate results are pending.

## Student-view grounding adaptation preparation

The next independently seeded capture uses 10 environments, 2,400 training steps and 1,200 validation steps with stride 5, producing 4,800 and 2,400 planned RGB frames. Seeds are 182931 and 182932, distinct from prior training, development evaluation and reserved formal seeds. The existing control collector now optionally exports aligned object pose and hand depth alongside RGB to a separate HDF5 capture. These values are label sources only and are not supplied to the grounding model. The learned capture behavior uses v1 maintenance during waiting and the unchanged student during execution; other mixed behaviors remain for coverage.

The collection supervisor is live and queued after control_wait1_async_stop_20261029 on GPU 5. Source snapshots, common 4096/1024 capacities and fatal capacity-warning handling are enabled. Capture and label audit must finish before fitting. Runtime validation of the new capture branch is pending.

Grounding training now supports continuing the full same-bank latent head and LoRA, instead of reinitializing the latent projection, plus alternating retained teacher-view data with new student-view data. It evaluates exactly the same selected held-out images before and after fitting and reports both data sources separately. The planned continuation is 240 updates, accumulation 2, learning rate 5e-5 and unchanged acceptance threshold 0.5. No new weights have been fitted or deployed yet. Python compilation, CLI import/help and launcher shell syntax checks passed.

## Verified development results and remaining training at 2026-10-29 04:23 CST

Capacity-corrected measured-RGB low-level diagnostics completed: the released student reached 38/40 targets before dropping, while the migrated latent student reached 37/40. All 40 initial state/goal pairs match and all four physics guards passed. These are short low-level diagnostics without semantic grounding or autonomous stopping.

Scripted correct-binding comparisons completed with valid physics logs. Continue achieved 30/40, fixed slowing 34/40, v1 waiting-only 38/40, and v1 waiting plus paired v3 execution 38/40. The latter improves over v2 execution's 35/40 but does not exceed waiting-only. The pending-request completion option achieved 34/40 and is not admitted. A loaded-model reference achieved 32/40 against 30/40 with unchanged policy, including eight discordant paired outcomes. Exact reset matching therefore must not be presented as bitwise trajectory reproducibility.

The actual Astra v3 execution ablation has started using the exact evaluator snapshot of the waiting-only reference. Its independent loopback bridge uses port 18785, a 40-session/240-call ledger and the existing authorized identity. The fixed-slow three-provider comparisons continue, with three waiting-only comparisons queued. Each condition has 40 raw trials on two development seeds. No formal seeds or scoring changes are introduced.

Student-view RGB capture is running. A separate QLoRA continuation supervisor is queued on remote GPU 1, with 240 updates, learning rate 5e-5 and the original acceptance threshold. It waits for both independent capture splits, successful physics guards, aligned data and reviewed visibility labels. It preserves the full latent head and LoRA, mixes retained teacher views with new student views, and evaluates identical held-out samples before and after fitting. Its isolated environment uses Transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.50.2 and the existing PyTorch 2.11.0+cu130 without changing the active simulation environment. The new model is not yet trained or deployed.

## Student-view label audit and evidence-gate comparison

A visual audit of the new training capture found geometric positives with no usable image signal. The R-positive example at frame 3723 had face-region brightness p95 of 20/255. H-positive frame 3223 had p95 16.5/255. H-positive frame 2893 had 90% of the central face region clipped to white. These examples must not teach the model that the letter is readable.

The new optional label filter omits geometric positives when brightness p95 is below 48, luminance p95-p5 is below 12, or at least 80% of face pixels have all RGB channels at least 245. These cases are omitted, not converted to negative labels. Back-facing and depth-occluded negatives retain the original geometric rule. The thresholds were chosen from the training capture audit, before reviewing the validation previews. Raw captures and original labels are retained. These remain conservative weak labels, not human-annotated readability truth.

The original pending training supervisor was intentionally stopped before any fitting. Its replacement, grounding_student_signal_20261029_v2, retains the same seeds, initial weights, updates, learning rate and replay mixture, with signal filtering applied to new and retained training/validation labels. It waits for separate visual review of the filtered previews.

The trained v2 ordinal completion head is now supported by the shared runtime completion interface. Its acceptance probability and a calibrated pose-only threshold come from the existing first-event calibration groups. Both new 40-trial control conditions load and compute the same evidence model; one uses the pose decision and the other the learned decision. The internal thresholds are 0.1560000032 rad and probability 0.9973879457 respectively. Independent scoring remains full true quaternion error at most 0.16 rad, no prior drop and autonomous stop. Both conditions use v1 waiting-only control, two development seeds, unchanged request/stop timing and no pending-request completion. Results are not yet available.

## Completed semantic baselines and student-view QLoRA continuation

All three fixed-slow development conditions completed 40 raw trials with valid physics logs and no service exclusions. Astra: 26/40 (65%). Fine-tuned VLM: 25/40 (62.5%). Rule VLM: 19/40 (47.5%). All initial-state/goal pairs match. The one-trial Astra versus fine-tuned VLM difference does not establish clear superiority. These are new-backend development results, separate from prior backend rates and the still-unrun formal full-method comparison.

Failure reconstruction found that 11 of the fine-tuned VLM's 13 drops occurred during inspection before a semantic binding had been established. One drop followed a correct binding and one followed an incorrect binding. This supports targeting unreadability rejection and observation trajectories rather than assuming all execution-phase drops are low-level target-control failures.

The independently seeded student-view capture completed 4,800 training and 2,400 validation frames with aligned RGB, depth and pose labels and clean physics guards. Signal filtering omitted 189 training and 43 validation geometric positives. Both filtered contact sheets were reviewed, 48 examples per split. Faint examples remain and the labels continue to be weak supervision proxies. No human readability ground-truth claim is made.

The continuation completed all 240 updates in 208.7 seconds on remote GPU 1. On the same selected 120 student-view validation queries (60 positive), accepted-correct predictions improved from 54 to 60, with zero invisible false accepts both before and after. On 120 retained-data queries (60 positive), accepted-correct predictions improved from 49 to 59, with one invisible false accept both before and after. Acceptance remains 0.5. These are offline weak-label metrics, not episode success.

The final head SHA is 69beb7fbc91b7b2cc5e7bb95aee2187182103a6e103670c6460d809616f5deda; LoRA SHA is d5a060b87eb720b4f08697e6861854a50c8e6780b5c36cb23d33d5e9f9f105c7. Twelve backbone/tokenizer files match between local inference and remote training, aggregate SHA 548f920ce9d0e8f5f04c70e36505a5b9e4d62e17fe8e798879bdd25a6e541ca4. A local service on port 18786 loads the untouched checkpoint using an explicit local backbone path. The original services keep their frozen weights.

A separate 40-trial continuation evaluation, agent_vlm_student_signal_20261029, is running with the same v1 waiting-only controller, seeds and byte-identical evaluator source snapshot as agent_vlm_wait_only_20261029. Its first actual episode completed and service/checkpoint hashes were verified. No full-method admission follows from that single episode. The formal seeds remain untouched.

## Full candidate development queued at 2026-10-29 04:52 CST

The first cube training cycle is complete: the migrated latent student, v1 maintenance actor, dense continuous v2 actor and evidence head, goal-paired v3 recovery actor, and student-view QLoRA continuation all have saved checkpoints. The colleague's RGB pose estimator remains frozen. There is no additional unstarted fitting stage required merely to assemble the current cube candidate.

Four more development conditions are queued with immutable shared evaluator sources, 40 raw episodes each and seeds 182921/182922:

| Condition | Upstream | Waiting | Execution | Stop decision |
| --- | --- | --- | --- | --- |
| Full Astra candidate | Astra | v1 actor | v3 actor | calibrated ordinal evidence |
| Full fine-tuned VLM candidate | student-view QLoRA | v1 actor | v3 actor | calibrated ordinal evidence |
| Full rule VLM candidate | frozen rule VLM | v1 actor | v3 actor | calibrated ordinal evidence |
| Astra completion ablation | Astra | v1 actor | v3 actor | calibrated measured pose |

The stop ablation loads and computes the same evidence head, then decides using the previously calibrated measured-pose threshold. The independent task scorer remains full quaternion error at most 0.16 rad, correct semantic binding, autonomous stop and no preceding drop. The gate calibration predates these runs. No completion during a pending request is enabled. The common maximum remains six upstream calls and 120 seconds of simulation per trial.

These are experimental combinations, not admitted final methods. Jobs wait for successful predecessor exits and the completion-control comparison. GPU 0 runs the two Astra conditions in sequence, GPU 1 runs the adapted VLM, and GPU 5 runs the rule VLM. The Astra conditions have separate 40-session, 240-call ledgers with the already authorized identity, one transport attempt and no fallback. No other user's jobs were stopped and no formal seeds were consumed.

Launch PIDs are 280378, 280379, 280381 and 280383. The plan, source hashes, weight hashes, predecessor dependencies and scoring contract are archived in `outputs/pose_latent_student_v1/live_development_capacity_20261029/full_candidate_plan.json`. Current individual evaluations continue on their original immutable snapshots. The updated runner adds explicit service URLs, calibration and stop-mode inputs. The summary separates failure before the first skill, unbound inspection, correct-bound execution and wrong-bound execution without changing raw scores.

At dispatch, the completed fixed-slow semantic baselines remained Astra 26/40, fine-tuned VLM 25/40 and rule VLM 19/40. Astra's one-trial lead over fine-tuned VLM does not establish a clear superiority result. Formal comparisons, continuous videos and additional object adaptation remain unfinished.

## Continuous qualitative recording and uncertainty reporting

An optional native calibrated Hikrobot camera now records the full hand beside
the actual 120 x 120 GS/occlusion policy image. The native camera is a spectator
only. The student still receives 109 proprioceptive values, seven measured-pose
values and a 768-dimensional goal latent. The active quantitative runs and
queued full candidate conditions retain their pre-recording source snapshots.

The first scripted-goal qualitative episode completed with clean physical
capacity logs. It recorded 51 decoded frames at 15 fps, steps 0 through 100 at
stride two. Frame timestamps are continuous and first, middle and final decoded
frames differ. The target was T upward and the independent final quaternion
error was 0.109298874 rad. This single episode is a video integration result,
not an Astra demonstration or a success-rate estimate. Its native camera audit
matches the delivered calibration and the live geometry remains 116 hand convex
collision shapes and one object box.

Local video: outputs/pose_latent_student_v1/videos_20261029/control/continuous.mp4.
The same directory contains the frame log, camera/physics verification, outcome
audit and sampled decoded frames. The video is played at simulation speed and
does not imply real-time throughput on the shared GPU.

Three six-case qualitative runs are queued after their corresponding full
candidate comparisons. They use the predetermined consecutive cases at seed
182951, covering T, E, R, L, S and H. All outcomes will be retained. These videos
are excluded from the 40-trial numeric summaries. The Astra video service has
its own six-session, 36-call, 300,000-token ceiling, no retry and the existing
authorized identity. The plan is saved in
outputs/pose_latent_student_v1/videos_20261029/semantic_video_plan.json.

The summary now reports Wilson 95% intervals for completed conditions and
unadjusted exact McNemar statistics for completed matched pairs. The baseline
Astra-versus-fine-tuned pair has 9 Astra-only and 8 VLM-only successes, p = 1.0.
Astra-versus-rule has 10 Astra-only and 3 rule-only successes, p = 0.09228515625.
These are descriptive development analyses on reused seeds, not confirmatory
evidence after multiple explored variants. The frozen scorer is unchanged.

## Completion failure diagnosis and conjunction ablation

The completed event-calibrated measured-pose condition scored 37/40, with
three false stops and no drops. Its three false-stop true errors were
0.174263, 0.164823 and 0.165460 rad. The corresponding evidence probabilities
were 0.964737, 0.946276 and 0.960050, all below the previously calibrated
learned threshold.

The running evidence-only condition exposed a different error at trial
182911010: probability 0.997622 accepted a state whose measured error was
0.209669 rad and true error was 0.216661 rad. This is a counterexample to
treating a high classifier probability as sufficient completion evidence.

A separate 40-trial control condition, control_event_conjunction_20261029,
is queued on GPU 5 after the evidence-only comparison. It uses the same
checkpoints, seeds, calibrated thresholds, waiting actor and original
pre-video evaluator snapshot. Only the stop decision is changed to require
both learned acceptance and calibrated measured-pose acceptance. The original
0.16-rad task scorer remains unchanged. No running or queued full-method
condition was changed. It is a diagnostic ablation, not an admitted method.

An offline audit also checked six-goal feature consistency on the existing
24,000 validation states. At the event-calibrated threshold, using recomputed
goal-switch features changed 16 accepted/rejected reporting queries; false
accepts remained 22. The same-active-goal feature discrepancy was at most
0.002414 and probability discrepancy at most 0.003013. The exact single-branch
replay in the original audit remains zero; six-way batching has small numerical
differences. This audit does not identify feature mismatch as the main false-stop
source and does not justify another fitting round by itself.

## Verified service interruption and continuation without replacement trials

The Astra recovery comparison stopped after 28 recorded cases because the
sixth request in trial 182922007 returned error_server_is_overloaded (HTTP 503).
The original process ended with service_blocked and the wrapper exited 1.
The physical-log guard passed. Later requests from the same authorized account
succeeded, so no account change was needed.

The excluded trial and every completed outcome are retained. A new output
fragment continues only the twelve unvisited indices 182922008 through
182922019, with the same weights, prompts, physics and scoring. The new
evaluator adds only a starting episode index. Original outputs and source
snapshots are preserved. The same service ledger retains its 40-session and
240-call ceilings.

Three dependent Astra tasks had ended before starting any trial. Their
dependency-failure status files were archived, and their original immutable
sources and parameters were requeued. The continuation record is
outputs/pose_latent_student_v1/live_development_capacity_20261029/astra_execution_resumption_01.json.
The summary now verifies exactly two seeds, all indices 0 through 19 once,
valid per-fragment results and clean physical logs before reporting completion.
A service-blocked fragment can contribute its retained excluded final case,
but missing cases or duplicate replacements cannot complete a cohort.

## Completed fine-tuned VLM waiting ablation

The unchanged fine-tuned VLM with the v1 waiting actor completed 30/40,
compared with 25/40 for fixed slowdown. The two seeds changed from 12 to 13
and from 13 to 17 successes. Drops decreased from 13 to 8; all eight remaining
drops occurred during inspection without a bound goal. One horizon failure and
one wrong-binding stop remained.

All 40 initial states matched, with 21 shared successes, nine waiting-actor-only
successes, four fixed-slow-only successes and six shared failures. There were no
service exclusions and all physical guards passed. Exact paired p = 0.266846,
so this is a positive development trend, not confirmatory superiority evidence.
The adapted student-view VLM and Astra comparisons remain in progress.

A standalone PNG/SVG of the complete three-provider fixed-slow baselines and
failure contexts is stored under outputs/pose_latent_student_v1/figures_20261029,
with the exact source summary, rendering script and data provenance.

## Completed student-view fine-tuning evaluation

The student-view adapted VLM with the unchanged v1 waiting actor completed
33/40 (82.5%), compared with 30/40 for the original fine-tuned VLM with the
same actor. Seed counts were 17/20 and 16/20, versus 13/20 and 17/20.
All physical guards passed and no trial was excluded. All forty initial
states matched. There were eight adaptation-only successes and five
reference-only successes, with exact paired p = 0.581055. Inspection drops
decreased from eight to five; wrong-binding stops increased from one to two.
These are development results, not a statistically established improvement.

The frozen adapted grounder was also applied offline to eleven first/last
request images from seven completed non-service failures in the Astra v3
recovery cohort. It accepted two images, both correctly. One contradicted a
wrong Astra blue/cyan binding; the other confirmed an already-correct Astra
binding that later dropped. Nine images remained rejected. This diagnosis
does not establish the success rate of a grounder-assisted Astra system.

## Binding-state coverage and targeted controller training

The original v1 collection and the longer v2 collection both contained zero
nonpending states with an unbound semantic target. Their v2 training counts
were 5,822 unbound/pending, 21,640 bound/executing, and 20,538 bound/pending.
Actual inspection can execute an active color skill without binding that
skill to the requested semantic target, so this input combination is missing.

On 24,000 v2 validation states, changing only the binding flag changed v3
execution actions by mean absolute 0.042712. Teacher MSE was 0.107802 under
the unbound flag and 0.099174 under the bound flag. This is an input-coverage
diagnosis, not proof that the flag caused any particular rollout failure.

The v4 experiment repeats the v3 training recipe with the same data, paired
goals, v1 initialization, 2,400 updates, and seed 182871. An independent RNG
balances only the actor's semantic binding flag. Sensors, active latent,
recurrent-history-derived features, and physical teacher targets are kept
unchanged. The original datasets and all running evaluation snapshots remain
unchanged. No formal evaluation states enter training.

Training completed successfully. Validation execution MSE was 0.099933
unbound and 0.099774 bound. Mean absolute action disagreement fell to
0.002049. Pending-state errors were 0.334199 and 0.334183, with mean action
disagreement 0.004229. The evidence model weights were verified bit-identical
to v3, so this experiment does not change the completion head.

A separate 40-trial Astra v4 execution ablation is queued on GPU 1 after
the v3 recovery cohort and the VLM full candidate. It retains the original
v1 waiting actor, measured-pose stop gate, seeds, prompts, physics, and
success definition. Every evaluator source file matches the v3 reference.
Only the execution-actor checkpoint changes. The bridge retains the same
authorized identity and enforces 40 sessions, 240 calls, and 1.5M tokens.
The v4 controller is not admitted to the final method until closed-loop
evidence is available.

## Second service interruption retained

Trial 182922011 ended with bridge HTTP 503 / ChunkedEncodingError after
108.835 seconds. It is retained as a service-excluded trial, together with
182922007. Subsequent same-account requests succeeded. Continuation 02 starts
only the eight unvisited indices 182922012 through 182922019. No completed
trial is retried or replaced. The source snapshot, weights, prompts, scoring,
physics and original service ledger are unchanged. Dependent Astra jobs that
had exited before their first trial were requeued with their original
parameters. Current process IDs are recorded in setup_status.json and the
resumption record, rather than rewriting the historical launch plans.

## Completed Astra v3 execution comparison

The v3 execution-actor condition finished all forty scheduled slots across
its preserved output fragments. It produced 28 successes, eight physical
drops, two wrong-binding stops, and two service-excluded trials. All physical
guards passed. The seeds contributed 15/20 and 13/20 raw successes.

Excluding the same two service-error trial IDs from both this condition and
the fixed-slow Astra reference leaves 38 paired trials: reference 25/38,
v3 28/38. There are 21 shared successes, seven v3-only successes, four
reference-only successes and six shared failures. All 38 initial states
match. Exact paired p = 0.548828. The raw count remains 28/40, so the eligible
rate 73.7% must never be reported without its 38-trial denominator and the
two retained exclusions.

Six drops occurred during unbound inspection and two after the correct
target was bound. There were no drops before the first skill was delivered
in this condition. This does not establish that startup failure is eliminated
in general. The waiting-only Astra condition is still incomplete, so the
isolated effect of adding execution v3 cannot yet be concluded.

The reporter now includes eligible_pairing for semantic comparisons. It uses
the identical group-wide exclusions used by eligible_success, and emits
inferential diagnostics only after all forty raw slots have been recorded
in each arm. Raw pairing is retained. Recomputing existing completed
conditions left all episode counts and no-exclusion paired statistics unchanged.

## Missed completion opportunities and continuous-data correction

The ordinal-evidence control ablation has exposed a decision-recall failure.
In completed horizon trials 182912000 and 182912008, pre-action alignment of
the logged probability and physical state identified 274 and 278 eligible
correct stopping checks. Their highest correct-state probabilities were
0.988715 and 0.970169, below the fixed 0.997388 threshold. Both trajectories
reached the target and remained there for many checks. Two other completed
drop trials also contained missed correct stopping checks. The head is not
admitted to the final method. Its existing comparison and full-candidate
experiments continue unchanged.

An existing measured-angle correction model was refit using the continuous
v2 training captures. The architecture, 1,200 updates and seed 182851 match
the earlier sparse-data correction experiment. No agent evaluation states
were added to training. The scorer remains 0.16 rad.

On the same validation reporting groups, near-target angle MAE improved from
0.011919 to 0.011052 rad. At independently selected calibration settings,
the learned correction accepted 7,155 reporting queries, with 7,103 correct
and 52 false accepts. The calibrated geometric baseline accepted 7,119,
with 7,058 correct and 61 false accepts. These dense correlated counts are
offline diagnostics, not episode-success estimates.

The fixed beta of 0.7 was then replayed at the first eligible 3 Hz stop
event of each captured same-goal segment. It accepted all 24 reachable
calibration segments and all 31 reachable reporting segments without a
false event. The event-calibrated geometric baseline also achieved 31/31
on the reporting segments. This is parity on a small reused development
split, not demonstrated superiority. The original event counts were
reproduced before comparing the new correction.

A separate 40-trial control ablation,
control_event_continuous_correction_20261029, uses the original v1 waiting
actor, student and evaluation seeds, with the new correction only.
Its recorded calibration is unchanged after the event replay. It makes no
Astra calls and must be reported separately from actual semantic agents.

## Resource queue releases and formal-seed status

The VLM full candidate was released from its resource-only wait on the
GPU 5 evidence experiment. Its sources, checkpoints and scoring remain the
original frozen candidate configuration. It is now running on GPU 1.

The Astra v4 execution comparison was released from its resource-only wait
on the VLM full candidate after checking 26,837 MiB free memory on GPU 1.
It still waited for the v3 Astra execution condition to finish before
starting, so that handoff did not add another simultaneous Astra cohort.
Both queue changes occurred before any trial in the affected job. Original
launch plans are retained, with separate queue-release records.

A scan of recorded new-backend episode files found no use of reserved
formal seeds 183101 or 183102. Formal comparisons remain pending method
admission and final checkpoint freezing.


## Completed ordinal gate and actual-model failure mechanisms

The standalone ordinal completion condition completed all forty development
trials with 32 successes, four drops, two false stops and two horizons.
The event-calibrated measured-pose condition achieved 37/40. All initial
physical states matched and both conditions passed capacity guards.
There were seven measured-pose-only and two ordinal-only successes
(unadjusted paired p = 0.1796875). The ordinal head is not admitted.

This completion failure must not be conflated with the original Astra
failure mechanism. A pre-action stop-opportunity audit of completed actual
model cohorts found no non-success with an eligible, correctly bound,
true-success stopping check under the original measured gate. Among the
ten non-service Astra v3 failures, nine never reached the requested target
and one reached it without an eligible correctly bound stopping check.
Inspection stability, grounding and execution remain the main priorities.

In the ongoing VLM full candidate, two completed failures demonstrate a
new gate-induced problem. Trial 182921002 had 311 eligible true-success
checks, 306 also passing the calibrated measured threshold. Trial
182921005 had eleven such checks, nine also passing that threshold.
They ended in a horizon and a drop respectively. Three examined wrong
bindings were wrong from their first accepted binding. These traces do
not support a correct-to-wrong rebinding cascade as their explanation.
They are mechanism diagnostics from an unfinished cohort, not a final
success comparison or proof of an alternative controller outcome.

The failure-conditioned VLM view diagnostic also checked the rejected
predictions. Only two of nine rejected top-one candidates were correct.
The rejected candidate will not be used as a trusted target prior.

The six predetermined Astra video cases were moved to the next GPU 0
slot after the continuous-data correction comparison. Only their resource
wait changed, before any trial began. Sources, weights, seed and service
budget remain unchanged. All outcomes will be kept and excluded from
quantitative forty-trial summaries.


## Continuous-data correction completed forty trials

The continuous-data pose correction completed 36/40, with 18 successes
in each seed. It had three execution drops, one startup drop, and no false
stops. Capacity guards passed. Against the calibrated measured-pose result
of 37/40, there were 33 shared successes, four measured-pose-only and
three correction-only successes. All forty initial states matched and the
unadjusted paired p value was 1.0. The correction is not admitted as an
overall improvement. The observed reduction from three false stops to
zero is worth retaining as a diagnostic, but all four drop differences
cannot be causally assigned to the gate under the measured trajectory
non-determinism. The six predetermined actual Astra videos started after
this job completed.


## Astra waiting-only completed forty trials

Astra with the original v1 waiting controller and unchanged execution
student completed 30/40, or 75 percent, with 15/20 in each seed and no
service exclusions. Capacity guards passed. The forty initial physical
states matched fixed slowdown. There were 24 shared successes, six
waiting-only wins, two fixed-slowdown wins and eight shared failures.
The fixed-slowdown baseline was 26/40. The unadjusted exact paired p value
is 0.2890625, so this is a positive development trend rather than a
confirmed improvement. It is currently the best completed Astra condition.

Its ten failures comprised four unbound inspection drops, two drops after
a correct binding, three wrong semantic bindings and one abstention.
No drops before the first skill were observed in this cohort.

Comparing the completed v3 execution condition requires removing its two
service-error trials from both arms. Waiting-only retains 30/38 successes
and v3 achieves 28/38. There are 25 shared successes, five waiting-only
wins, three v3-only wins and five shared failures. The unadjusted paired
p value is 0.7265625. Execution v3 is not promoted as an incremental gain.
The binding-balanced v4 condition continues unchanged. The original full
candidate and pose-only completion comparison remain exploratory and
will finish their predeclared forty trials before final method selection.


## Conjunction, fixed-trajectory replay and real Astra videos

The learned-and-measured control condition completed 39/40, with one waiting
drop and no false stops. It uses scripted correct semantic deliveries, not
Astra. Against measured pose at 37/40, three trials favored conjunction and
one favored measured pose (unadjusted paired p = 0.625). Against the separate
ordinal-only run at 32/40, seven favored conjunction and none favored ordinal
(p = 0.015625). These are exploratory comparisons across reused seeds.

Recorded pre-action event replay reproduced every original gate decision.
On the conjunction run's own observed trajectories, ordinal evidence alone
and the conjunction each had 39 correct first acceptances and one trace
without acceptance. The conjunction delayed one otherwise correct evidence
acceptance. On the ordinal-only trajectories it rejected one of two false
acceptances, but that trace ended at the original stop, so subsequent recovery
cannot be inferred. The high independent conjunction score is therefore not
proof that conjunction caused all seven additional successes. All forty
physical initial states matched. First-step quaternion outputs differed on
26 paired trials before a stop intervention was possible. The second seed
also had small initial pose-estimation and action differences. Final method
admission remains open.

The first two predetermined actual Astra video cases completed successfully.
Case T had four actual model calls: inspect red, reorient the visually
identified T, verify, and stop. Its video has 416 decoded frames at 15 fps,
covering simulation steps 0 through 830. The request latencies were 5.098,
7.015, 5.650 and 5.531 seconds; corresponding pending advancement was 153,
211, 170 and 166 control steps. The E case had one actual call and 116 decoded
frames through step 230. Both preserve continuing physics, and playback uses
simulation time, not a claim of real-time throughput on the shared GPU.

The third predetermined case was also retained: it dropped after 1,453 steps
and six Astra calls. Although the requested target was physically reached
at some point, no correctly bound eligible true-success stopping check was
recorded. It must not be attributed to a missed completion decision without
that eligibility. All six cases remain part of the qualitative selection,
with no use in forty-trial quantitative rates.

## Frozen visual verifier diagnostic and queue handoff

An offline diagnostic reused the frozen adapted VLM on all 33 last applied
binding images from the completed Astra waiting-only condition. Of thirty
successful cases, it accepted the same correct binding for 29 and rejected
one as unreadable. Of three wrong bindings, it proposed the correct goal
for one, rejected one as unreadable, and repeated one wrong binding.

The check was expanded to all 112 requested images from those forty trials,
reusing the first 33 predictions. Only one drop case and one wrong-binding
case had correct VLM evidence on a strictly earlier request than Astra's
first applied correct binding. Two ultimately successful trials also
contained an earlier wrong accepted VLM proposal. Snapshot availability
is not delivery time and does not establish safe closed-loop improvement.
No automatic VLM override or semantic-verifier module is admitted from
this diagnostic. No extra Astra calls were made.

After the conjunction job freed a simulation slot on GPU 5, the zero-trial
rule-VLM full candidate was released from its resource-only waiting dependency.
Its frozen source hashes and forty cases were preserved. An initial launcher
refused the already existing queue directory before starting any trial; the
launcher was repaired and the replacement process entered rule_182921.
The dependent six-case rule video job was requeued with its original dependency.


## Runtime input alignment and public-agent phase collection

An audit found that the old collector retained the previous pending duration
on every nonpending sample, whereas deployed control supplies zero outside
waiting. This affected all 21,640 training and 10,776 validation nonpending
states. For v4, validation teacher MSE was 0.099774 with the stored timing
feature and 0.102170 with the actual runtime input. This modest offline gap
does not establish the cause of closed-loop failures.

Timing-aligned v5 completed 2,400 updates on the same data, paired goals,
warm start, training seed and binding augmentation as v4. Only the actor's
nonpending elapsed-time input was aligned. Evidence weights and threshold
remain bit-identical to v4. Runtime-aligned execution teacher MSE is 0.099106.
The actor SHA256 is
`e2c9e764bf035fbbb9789066a58fdfac8128636261ad5519ce6977a5c32b4dfe`.
A dedicated actual Astra forty-trial comparison is queued after v4, using
byte-identical evaluation sources and the unchanged v1 waiting controller
and measured-pose stopping gate. Offline regression loss is not task success.

The next collection uses public Astra request/action schedules with fresh
physical seeds 182971 and 182972, 10 environments, and 48,000 training plus
24,000 validation states. It copies public action categories, colors,
execution durations and measured latencies from development request logs.
It does not copy their physical states, images, correct-target annotations
or outcome labels. The frozen teacher supplies labels on newly generated
states. Six-goal paired replay and 2,400 updates are chained after collection.
This data targets unbound multi-view execution, which was absent from the
original nonpending collection. No new Astra requests are required.

The zero-sample v6 initialization attempt exposed a local variable named
`binding` shadowing the environment binding. The failed log and source were
archived before renaming this local variable to `binding_update` and restarting.
No collected sample, physics setting, label or seed was changed. The job uses
the GPU 0 simulation slot released by the completed six-case Astra recording.

The repaired collector wrote its first 2,000 states. Of 462 nonpending
states, 368 had no semantic binding and 94 were bound. All nonpending
elapsed inputs were zero, all checked input tensors were finite, and no
capacity warning had appeared. Live geometry was 10 worlds with 1,160
convex-mesh and 10 box collision shapes. This is partial data verification,
not a completed collection or policy success result.

## Completed VLM full candidate and qualitative set

The adapted fine-tuned VLM full candidate completed 29/40 with no service
exclusions and valid capacity guards. Its waiting-only reference completed
33/40. There were 27 shared successes, six waiting-only wins, two full-candidate
wins and five shared failures, with exact initial-state matches in all forty
cases. The unadjusted paired p value is 0.2890625. Adding the current execution
and evidence modules is not admitted as an improvement.

Failures comprised five wrong semantic bindings, three drops, two horizons,
and one pose false stop. Recorded pre-action diagnostics found three failed
trials with an eligible correctly bound state satisfying both the true 0.16
rad criterion and calibrated measured-pose threshold. These are missed stop
opportunities on the recorded prefixes, not a counterfactual recovery score.

All six predetermined actual Astra video cases completed and were retained:
five successes and one drop, with 22 actual model calls. Every video decoded
to the expected frame count. These qualitative cases are excluded from all
forty-trial quantitative rates. Native spectator footage remains separate
from the 120-by-120 policy RGB, and playback follows simulation time.


## Rule waiting condition completed forty trials

Rule VLM with the v1 waiting controller completed 15/40, with no service
exclusions and valid capacity guards. Fixed slowdown was 19/40. The paired
outcomes were 13 shared successes, six slowdown-only successes, two
waiting-only successes and 19 shared failures, with all forty physical
initial states matched. The unadjusted paired p value is 0.2890625. This
condition does not support a universal benefit of the waiting controller.
Its failures were 19 unbound inspection drops, one correct-bound execution
drop, three horizons and two incorrect semantic bindings.

The original-model waiting-only triplet is now complete: Astra 30/40,
fine-tuned VLM 30/40 and rule VLM 15/40. Astra versus fine-tuned VLM has
seven discordant wins in each direction. Astra versus rule has sixteen
Astra-only and one rule-only success. These are reused development trials,
not untouched confirmatory evidence. The separately adapted VLM remains
33/40 and must not be conflated with the original VLM in this triplet.


## Completed Astra v4 and independent v6 handoff

The binding-balanced v4 execution actor completed 25/40 with no service
exclusions and valid capacity guards, split 13/20 and 12/20. Waiting-only
was 30/40. There were twenty shared successes, ten waiting-only successes,
five v4-only successes and five shared failures, with all forty initial
states matched. The unadjusted paired p value is 0.3017578125. Against v3,
removing its two service-error slots from both arms gives v3 28/38 and
v4 25/38, with six v3-only and three v4-only successes. V4 is not promoted.

V4 failures were three unbound inspection drops, one pre-skill drop, three
correct-bound execution drops, five wrong bindings, two pose false stops
and one abstention. The pre-skill difference occurs before the execution
actor is used and cannot be attributed to it without accounting for the
recorded numerical divergence. V5 has started its unchanged forty-trial
comparison. V6 is queued after its training and v5, with byte-identical
evaluation sources, the same v1 waiting actor and the same measured-pose
stop gate. This comparison replaces only the execution actor, leaving the
new evidence head for a separate evaluation.

V6 finished all 48,000 training states and entered validation collection.
The full training-input audit found 9,551 pending states, 4,106 unbound
execution states and 34,343 bound execution states. Every nonpending elapsed
input was zero. Checked sensor, action and label tensors were finite, and
the physical capacity guard passed. Fifteen drop transitions were retained
as training data. These are state counts, not semantic episode results.
The first 46,000-state coverage audit also found 33,985 states after the last
recorded public delivery. This includes valid convergence and stable holding;
its dominance remains a sampling-efficiency consideration, not proof of failure.

## Completion timing audit and reproducible event calibration

The deployed v2 evidence head was evaluated with both stored validation
inputs and runtime-aligned zero idle elapsed time, using the frozen
probability threshold. Stored probabilities reproduced exactly. On the
37 report segments, both variants accepted thirty correct first stops,
with no false stops and one missed reachable segment. Two first-decision
times changed. Calibration-group counts also stayed unchanged, although
five decision times changed. These captured trajectories do not support
idle timing as the main explanation for the observed missed-stop failures.
No extra head-training condition was launched on that assumption.

`scripts/pose_latent_student_v1/calibrate_completion.py` now reproduces the
first-event procedure. It uses the same whole-episode SHA256 partition,
known nonpending checks at 3 Hz, at least twenty accepted calibration
events, and empirical precision of at least 99 percent. Equal calibration
outcomes select the most permissive threshold. On the existing captures
it reproduces measured threshold 0.15600000321865082 and evidence threshold
0.9973879456520081, plus their report counts. Independent scoring stays at
0.16 rad and retains semantic binding, autonomous stop and no prior drop.

A source-frozen calibration job is queued after v6 fitting. It can report
insufficient calibration coverage instead of silently admitting a threshold.
Its captured-trajectory report is separate from the queued actual Astra v6
recovery comparison. Formal seeds remain unused.


## Completed schedule training and isolated evidence follow-up

V6 completed 48,000 training and 24,000 validation states with valid physics
capacity logs. The actor's execution teacher-action MSE was 0.10144 versus
0.14197 for the unmodified student on the new capture. This does not beat v5:
on the same runtime-aligned new validation states v5 scored 0.09753. On the
older capture, v5 scored 0.09911 and v6 scored 0.10755. These are offline
regression errors, not semantic success rates. Both forty-trial actual Astra
comparisons remain necessary. The v6 comparison was released onto GPU 5 before
its first trial, with evaluation sources unchanged, rather than waiting for v5.

V6 validation contained only twelve known, nonpending same-goal segments
eligible for first-stop calibration. Neither gate could meet the existing
minimum of twenty accepted calibration events. On the old validation capture,
measured pose still calibrated at 0.156 rad, but v6 evidence did not qualify.
Its best precision with at least twenty accepted events was 23/24. Keeping
99 percent empirical precision left only fifteen accepted events and nine
missed reachable segments. No threshold requirement or success definition
was relaxed.

A boundary-negative training variant, v7, replaced half of its negative batch
with queries in (0.16, 0.32] rad. These queries constituted only 1.69 percent
of negatives in the v6 training set. Architecture, updates, data and actor
supervision were held fixed, with an independent evidence sampling RNG.
After training, actor weights and normalization tensors were verified bitwise
identical to v6. V7 still failed the old-capture first-event calibration and
was not deployed. Its model and failed analysis were archived.

V8 isolates replay of earlier transition states: half of each evidence batch
comes from the old training capture, with idle timing aligned to runtime.
It retains v6 evidence architecture and normalization, 2,400 updates, seed
182871 and the original positive/negative proportions. Boundary oversampling
is disabled. Its independent replay RNG preserves actor sampling, and a
bitwise actor comparison is required before proceeding. No validation state
is added to training.

The v8 pipeline also collects 36,000 fresh measured-pose states with seed
182982, ten environments and repeated scripted goal changes. This collection
reuses the frozen empirical request latency distribution and v1 waiting
controller, with continuous physics and unchanged capacity settings. Its
purpose is to replace long settled tails with more eligible decision segments.
It makes no new Astra calls and is not a forty-trial autonomous agent result.
Event calibration retains disjoint episode groups and can fail without
admitting the new head. The untouched formal seeds remain unused.

V8 fitting subsequently completed. The actor isolation audit passed bitwise,
but the evidence head still failed first-event calibration on the old capture.
Its fresh validation collection is running. Neither the v7 nor v8 head has
been deployed into an autonomous comparison.


## Completion coverage, threshold resolution and fresh transfer audit

The first-event threshold-resolution audit searched all observed calibration
scores inside the existing [0.5, 0.9999] interval. V6, v7 and v8 still failed
the twenty-accept / 99-percent-empirical-precision gate. The failure is not
explained by the original 200-point threshold grid. The deployed v2 retained
22 correct calibration accepts and thirty correct report accepts, with no
false stops. The grid and score-based searches gave the same report counts.
No deployed threshold was changed by this audit.

Goal coverage is strongly uneven in v6 training. Goal 2 has 348 positive
states across three episodes; goal 1 has 13,318 across ten episodes. An
isolated v9 variant samples goals uniformly inside each binary class, while
retaining the v6 data, architecture, update count and training seed. Its
sampling RNG is independent, and actor tensors were verified bitwise equal
to v6 after fitting. Boundary oversampling and old-data replay are disabled.
V9 calibrates at 0.989851713180542 on the old calibration groups. On the old
report groups it accepts 25 correct stops with no false stops, missing six
reachable segments. V2 accepts thirty, missing one. V9 therefore passes the
calibration prerequisite but is not an established improvement.

Five heads (v2, v6, v7, v8, v9) are queued for comparison on the same fresh
seed-182982 capture. In addition to the same disjoint-group development
calibration, the analysis reports deployed v2 and old-calibrated v9 thresholds
applied unchanged to all fresh segments. The calibrated geometric reference
is likewise carried over unchanged at 0.15600000321865082 rad. No formal
semantic trial is used in fitting or threshold selection.

An input audit of the two completed adapted-VLM conditions found unit-norm
accepted embeddings, no mismatch between declared skill and nearest frozen
latent, and no goal-feature clipping in the completion encoder. The mean
cosine to the selected frozen latent was 0.99763 for waiting-only and 0.99667
for the full candidate. These checks rule out a gross normalization or
clipping failure; they do not prove the head is insensitive to continuous
latent differences.

Future agent source snapshots now record completion probability, estimated
error and gate rejection at eligible nonterminal decisions. This is logging
only. Existing running source snapshots and their decision rules remain
unchanged, and earlier misses must not be retrospectively assigned invented
network scores.


## Fresh first-stop transfer and completed Astra full candidate

Fresh validation seed 182982 completed 36,000 states, 100 eligible same-goal
segments and a valid capacity guard. There were 43 calibration and 57 report
segments. Sixteen drop transitions and nineteen resets were retained in this
capture. These segment counts are physical diagnostics, not semantic agent
success rates or independent forty-trial cohorts.

With thresholds fixed from the older capture, calibrated geometry accepted
86 correct and one false first stop, with no missed reachable segment. The
v2 evidence head accepted 73 correct and one false stop, missing thirteen
reachable segments. V9 accepted 68 correct and one false stop, missing
eighteen. Recalibrating each head on the new calibration partition did not
establish an advantage on its report partition: v2 gave 47 correct / 4 false,
v6 30 / 0, v7 32 / 1, v8 45 / 1 and v9 43 / 6. The corresponding geometric
calibration gave 52 / 0. These development comparisons do not admit a new
completion head or alter any deployed threshold.

The existing continuous-data pose correction was also carried over at its
unchanged beta of 0.699999988079071. Stored and runtime-aligned inputs both
reproduced 55 correct and zero false first stops on the old capture. On the
fresh capture it matched the fixed geometric gate at 86 correct / 1 false
with no missed reachable segment, including the same false segment. The
audit does not support a further timing-input retrain of this correction.

The actual Astra full candidate completed 27/40 (12/20 and 15/20), with no
service exclusions and valid capacity guards. Waiting-only was 30/40. Its
failures comprise three abstentions, five drops, three pose false stops,
one wrong semantic binding at stop, and one horizon. The full candidate
is not admitted as an improvement. The measured-stop ablation has started
automatically and holds its execution actor fixed. The separate v5/v6
execution comparisons remain in progress and must finish all forty trials.


## Fixed public-plan controller comparison and common-stack provider evaluation

Three controller conditions are dispatched on fresh development seeds 182991
and 182992, with twenty trials per seed. They replay the same forty public
tool sequences and their 112 recorded request durations from the completed
Astra waiting-only cohort. Source images, physical states, outcome labels
and hidden model representations are not transferred. The plan SHA-256 is
`5aafddd74d2ec48ee0614c221f7e685c11c1b0e6239a6ebb00850eaf134ee30e`.

The arms use unchanged v1 waiting control, with either the original student
execution, v5 execution recovery or v6 execution recovery. After the last
recorded public call, the last skill continues under the same local stop,
drop and horizon rules. These 120 raw trials isolate controller behavior
under fixed public instructions and latencies. They are model-free mechanism
diagnostics and are not added to the live Astra/VLM performance table.
Initial records confirm zero actual model calls, exact tool/argument/duration
replay and no response delivery before its recorded latency.

The actual adapted-VLM and rule-VLM services are also queued for forty raw
trials each with the same v1 waiting actor, v6 execution actor and measured
pose stop as the current Astra v6 condition. Their evaluator snapshots are
byte-identical to Astra v6. This comparison preserves the strongest existing
adapted-VLM waiting result of 33/40; the full candidate's 29/40 is not used as
a substitute for that baseline. No formal seed has been used.

All six predetermined Astra clips and all six adapted-VLM clips are now
archived with frame-count checks and a combined `qualitative_index.json`.
The corresponding initial-state hashes and low-level checkpoint hashes
match across the two providers. Each set includes its unsuccessful trial.
These twelve clips are qualitative examples, excluded from numerical rates.


## V6 provider launcher compatibility repair

The adapted-VLM v6 condition was released into a second simulation slot on
GPU 1 before Astra v5 completed. Its first launch exited before producing
any trial because the generic runner passed `--stop_mode` to the frozen
Astra v6 evaluator, which predates that option. Startup logs were retained
under the run's `startup_failures/unused_stop_mode` directory.

The launcher now omits that unsupported argument. The evaluator sources,
checkpoints, seeds, measured 0.16 gate and scoring remain unchanged. The
queued rule-VLM v6 launcher received the same correction before starting
any trial. This is an entrypoint repair, not a replacement of an evaluated
trial or a change in the success criterion.


## Completed timing-aligned execution recovery v5

Actual Astra v5 completed 31/40 autonomous successes (16/20 and 15/20),
with all capacity guards valid and no service exclusions. It made 110
model calls. The nine failures comprise one abstention, five drops and
three wrong semantic bindings at stop. Among the drops, one precedes the
first skill, two occur during unbound inspection and two after correct
goal binding. No pose-only false stop was recorded.

The unchanged waiting-only reference is 30/40. Their paired outcomes are
25 both-success, five waiting-only successes, six v5-only successes and
four both-failures (unadjusted exact McNemar p=1.0). Against v4's 25/40,
v5 has ten exclusive successes and four exclusive failures (p=0.1795654).
These reused-development comparisons do not establish a reliable gain
over waiting-only. V5 is retained as a timing-alignment ablation while
v6 and the separate fixed-plan mechanism comparisons finish.


## Inspection-state sampling ablation v10

The v6 training capture contains 9,551 pending states, 4,106 unbound execution
states and 34,343 bound execution states. The uniform execution sampler
therefore gives unbound inspection only about 10.7 percent of execution
examples. The remaining v6 drops observed before this ablation were all
unbound inspections.

V10 changes only actor sampling to 50 percent pending, 25 percent unbound
execution and 25 percent bound execution. It retains the same capture,
paired teacher goals, warm start, architecture, binding augmentation, idle
time input alignment, seed 182871 and 2,400 updates. An independent RNG
preserves original global draws. The evidence model is bitwise identical
to v6 after fitting and is not deployed.

On the same 2,338 validation inspection states, teacher action MSE changes
from 0.117845 to 0.113884. On 13,684 bound execution states it changes from
0.098634 to 0.098532. These are offline action errors, not task success. The
waiting controller remains v1. V10 joins the fixed-public-plan experiment
as a fourth forty-trial arm, making 160 planned raw diagnostic trials.
The three original arms, plans and scoring remain unchanged.

The first v10 training snapshot accidentally reuses the timer variable as
a batch offset. Its logged elapsed-seconds values are invalid. Sampling,
losses, model updates and RNG isolation are unaffected. Original logs and
source remain archived. The source fix only renames the offset variable.
Wall time is derived separately from the launch and completion timestamps.


## Prospective formal component design before any formal trial

The reserved formal conditions are prospectively revised to a 2x2 learned
waiting by learned execution-recovery factorial. Its cells are fixed
slowdown with student execution, learned waiting only, learned execution
with fixed slowdown, and both learned components. The full cell is reused
in the three-provider comparison. Two low-level diagnostics plus three
main provider cells plus three additional Astra cells remain 320 raw trials.

The earlier completion-learning factorial is retained in the configuration
as a superseded proposal. Fresh completion evidence did not justify that
head, so it is not forced into the full method. Its completed positive and
negative development studies remain available. Final recovery checkpoints
are still pending the common-stack and fixed-plan results. No formal seed
has been run or used to select a checkpoint. The frozen success-contract
SHA remains `85da9291e02596910cf0bad1e633b8dd3e5fd6500d7f43e083588b5dbbc3e594`.


## Completed schedule-trained Astra v6

V6 completed 33/40 autonomous successes, with 18/20 and 15/20 by seed.
All contact-capacity guards passed, with 113 model calls and no service
exclusions. The failures are three wrong semantic bindings at stop, three
drops during unbound inspection and one abstention after correct binding.
There are no pose-only false stops or correctly bound drops in this cohort.

Against waiting-only at 30/40, paired outcomes are 28 both-success, five
v6-only successes, two waiting-only successes and five both-failures
(unadjusted exact McNemar p=0.453125). Against v5 at 31/40, exclusive wins
and losses are four and two (p=0.6875). Against fixed slowdown at 26/40,
they are eight and one (p=0.0390625). These comparisons reuse development
data and are exploratory across multiple examined variants. They do not
establish stable superiority over the strongest reference.

The strongest completed adapted-VLM waiting reference is also 33/40.
Provider comparisons with the same v6 downstream stack remain incomplete.
The v6 raw records, images, logs, sources and model identities are archived
in `astra_execution_v6_completed_20261029.tar.gz`. The paper repository
now retains thirteen completed actual-model conditions and 520 raw trial
records in a separate development export. Its main LaTeX tables and PDFs
remain the historical series, pending the new formal experiment set.

The rule-model qualitative recording is queued after its forty-trial v6
comparison to avoid concurrent qualitative and quantitative calls to the
same local model service. No recording trial had begun before requeuing.

## Formal entrypoint preparation, before final release selection

The semantic evaluator now exposes learned execution with fixed-slow waiting
as a separate maintenance mode. Pending calls use the same 0.15 interpolation
as the fixed baseline. Only non-pending steps use the selected execution
actor. Existing maintenance modes retain their action branches.

The launcher defaults to the existing development seeds. Supplying both
FORMAL_RELEASE and FORMAL_CONDITION selects reserved seeds 183101/183102.
The evaluator rejects those seeds without a frozen release. Before the
first trial it verifies the unchanged success-contract digest, the complete
Python source snapshot, external client/observation sources, and the
condition's exact runtime, perception, weight and stopping contract.
A formal continuation can cover only the remaining contiguous tail of a
scheduled twenty-trial seed. Summary completeness still requires every
scheduled raw slot, including service exclusions.

Only syntax, shell parsing and diff checks have run for these additions.
The new action mode still needs a simulator rollout. No final release has
been selected or generated, no formal seed has been launched, and active
development snapshots have not changed. Final admission must also pin the
actual local model-server configuration and prepare the low-level diagnostic
launch metadata. These changes do not establish a new policy result.

## Completed rule full candidate and v10 live comparison

Rule VLM with v3 execution recovery and v2 learned stopping completes 20/40
(10/20 per seed, 172 actual calls), without exclusions or capacity warnings.
Failures are six wrong bindings, six horizons, six unbound-inspection drops,
one correctly bound drop and one pose-only false stop. Against waiting-only
15/40, paired exclusive counts are eight wins and three losses (unadjusted
p=0.2265625). The rule v6 common-stack experiment has started automatically.

An audit of the forty fixed public plans finds 32 containing a correct target
binding, three with wrong bindings only and five with inspection/abstention
only. Their frozen replay therefore has at most 32 possible semantic successes.
All raw outcomes and denominators remain intact. Descriptive strata now show
plan coverage and drops/horizons/false stops. Preventing a drop without a
sufficient high-level plan cannot demonstrate autonomous semantic completion.

Astra v10 is queued on GPU 1 after the adapted VLM v6 experiment. It uses
forty actual-model development trials and sources byte-identical to the
completed Astra v6. Only execution actor weights change to the inspection-
balanced candidate. The waiting actor, student, prompts, observation, stopping,
seeds, scoring and horizon remain unchanged. Service port 18793 uses the same
consented identity, loopback tunnel, forty-session cap, 240-call cap and
1.5-million-token budget. PID 423968 was confirmed live in predecessor wait.
No formal seed is used.

The paper development export now contains fourteen completed actual-model
conditions and 560 raw records. The manuscript's historical main tables are
not replaced by reused-seed development evidence.

## Formal low-level entrypoint and model service inventory

The original-policy and migrated-policy diagnostic runner now requires the
same frozen release before reserved-seed use. Its RGB-CNN feedback and
original twenty-second environment termination are preserved, as is its
reach-before-drop diagnostic metric. Formal metadata explicitly distinguishes
these policy diagnostics from autonomous semantic-agent success. The agent
benchmark continues to use its frozen 120-second horizon and autonomous stop.

Both launchers have passed shell parsing and their modified Python files pass
AST parsing. No formal runtime has been launched and the new release branches
have not yet been exercised through simulation.

A local service inventory records hashes for twelve Qwen3-VL-4B files
(8,887,284,080 bytes), the relevant serving/prompt sources and dependency
versions. The adapted service health matches the selected step-240 head and
LoRA hashes. The rule service uses the same backbone with four-bit NF4,
double quantization, bfloat16 compute and deterministic generation. Astra uses
the existing low-reasoning structured-tool interface. This is an input
inventory, not the final release or proof that a running process reloaded a
particular source snapshot. No active service or credential was changed.

## Completed v6 VLM comparison and measured stopping ablation

Adapted VLM v6 completes 33/40 (16/20 and 17/20), with 130 actual calls,
no exclusions and valid capacity checks. Its failures are four wrong
bindings, two unbound inspection drops and one pose-only false stop.
Astra v6 is also 33/40. Their fifteen audited downstream fields match,
and paired outcomes are 29 shared successes, four exclusive successes
each and three shared failures (exploratory p=1.0). This does not
establish Astra superiority or statistical equivalence.

VLM waiting alone is also 33/40. Moving to v6 execution reduces drops
from five to two but adds two wrong bindings and one pose-only false
stop in these runs. The same total completion count masks different
failure compositions. Actual feedback and physical trajectories differ,
so this is a descriptive pattern, not isolated causal attribution.

The Astra v3 execution / measured-stop ablation completes 32/40
(17/20 and 15/20), with 110 calls and no exclusions or capacity warnings.
The local measured threshold is the pre-existing calibration at 0.156,
while scored success stays at 0.16. Against the learned-stop version at
27/40 there are ten measured-only versus five learned-only successes
(exploratory paired p=0.3017578125). The evidence does not admit the
learned completion head.

Astra v10 started automatically after the VLM run. Its actual runtime
comparison confirms only the execution actor differs from v6. V10 fixed
replay was released from its zero-trial wait after the measured-stop job
freed one GPU 0 slot. Old launcher 411504 was explicitly stopped while
waiting, its logs archived, and new launcher 433678 uses the same sources,
weights, plan and seeds. This was a resource rescheduling, not a restarted
trial. The paper development export now preserves sixteen complete
conditions and 640 raw records.


## Completed waiting replay and independent execution branch

The fixed-plan waiting-only arm completes 24/40, twelve per seed, with zero
live model calls, no exclusions and valid contact-capacity logs. It has ten
drops, three wrong-binding stops, one pose-only false stop, one abstention and
one horizon. These are fresh physical seeds and frozen upper-level schedules.
The result is not the actual-Astra waiting-only rate of 30/40.

The forty replay plans contain 32 with a correct binding available, three with
only wrong bindings and five with inspection or abstention alone. The original
forty-trial denominator is retained. No survival-only outcome is rescored as a
success. Completed replay rows are exported separately from actual-model rows.

A fifth forty-trial replay arm uses fixed 0.15 slowdown while a request is
pending and the v6 recovery actor during execution. The waiting v1 and execution
v6 checkpoints are loaded in the same order as the full-v6 arm. Actual runtime
checks verify 402 pending steps at speed 0.15 and 180 execution steps at speed 1,
with matching loaded policies and physical setup. This verifies branch usage,
not task improvement or bitwise reproducibility. The total planned mechanism
cohort is now 200 raw records across five arms, with zero live model calls.

The formal-release guard requires all eight planned conditions before any
reserved-seed rollout. All six agent cells must load the same selected student,
waiting and execution checkpoints while enabling the intended components.
Static Python and shell checks pass. Final release selection and its runtime
validation remain pending. Active development source snapshots are unchanged.


## Paired v6 grounding and request-time audit

An offline audit of the completed Astra and adapted-VLM v6 logs preserves all
forty raw slots per model and their eight discordant trial pairs. Exact initial
states match in all forty pairs. This reuses the existing 80 records and adds no
new trial to the 640-record actual-model development export.

Both models correctly bind the requested target at least once in 34/40 trials.
Astra has three wrong first bindings and three trials without a binding. The
adapted VLM has four wrong first bindings and two without a binding. Neither
model corrects an earlier wrong binding in this cohort. Both finish 33/40.
Astra's remaining correctly bound failure is an abstention. The VLM's is a
pose-only false stop. Conditional binding counts are not a substitute for the
forty-trial autonomous-success denominator.

Astra makes 113 calls with mean caller-observed latency 5.3149 seconds, median
4.9650 and maximum 21.0489. The VLM makes 130 calls with mean 0.1209 seconds,
median 0.1145 and maximum 0.2479. Mean accumulated simulated waiting is 14.9592
seconds per Astra trial and 0.4567 per VLM trial. Simulated waiting and completion
times are not real-time throughput measurements under shared GPU execution.
Astra responses report 204,886 input tokens and 6,305 output tokens. The local
VLM does not report token usage. Missing usage is recorded as null, not zero,
and these counts are not monetary costs or account allowances.

Among Astra's four exclusive successes, three VLM runs bind the wrong target
and one stops with pose error 0.18394 rad. Among the VLM's four exclusive
successes, two Astra runs bind incorrectly, one drops during unbound inspection
and one abstains after correct binding. Raw first-binding request images have
been inspected and indexed. Views differ across the physical trajectories, so
this does not isolate a causal model-perception effect. It does locate the
remaining errors primarily in acquisition and semantic binding rather than
post-binding dropping in this specific development cohort.

The audit is produced by `scripts/pose_latent_student_v1/analyze_agent_pair.py`.
It records source and input hashes, per-call public bindings, response timing,
and original outcomes. It does not inspect or alter hidden model states.
The paper copy is `data/pose_latent_migration_development/mechanism_v6/`.

The formal-release check now also requires identical downstream runtime fields
across the six agent cells and validates their registered provider/control modes.
Astra ablations must share the same upstream contract. This is a configuration
boundary check. The final release has not been selected or run.


## Snapshotted upper services for the pending formal release

Fifteen source files covering the actual Astra prompt, client, bridge and
transitive imports, plus the rule and latent VLM paths, are copied into an
independent service candidate. The adapted head and its three LoRA files are
copied without changing weights. All twelve backbone/tokenizer/configuration
files are rehashed against the previous inventory. The 8.9 GB backbone is
referenced in place and revalidated by the launcher before a VLM server starts.
No credential is copied into the snapshot or paper evidence.

Four Astra processes are running from this source candidate. Their actual
process command lines match the recorded commands. Local and remote loopback
health checks pass on ports 18801--18804. They retain one worker, 120-pixel
frames, the existing identity binding, and per-condition limits of forty
sessions, 240 calls and 1.5 million recorded tokens. They have made zero model
inference calls. This verifies service readiness, not an evaluated method.
The rule and adapted-VLM services are prepared but have not yet started from
the snapshot. Existing development services and jobs continue unchanged.

The prospective formal resource matrix covers all eight conditions and 320
raw trials on the two reserved seeds. It schedules one simulator process per
project GPU (0, 1 and 5), and runs the fine-tuned and rule VLM conditions
sequentially on the shared local inference GPU. Control-actor selection,
complete runtime contracts, final VLM service startup and formal rollout
validation remain pending. No reserved-seed trial has started.


## Explicit controller-goal input diagnostic

A CPU-only audit evaluates frozen v1, v5, v6 and v10 actors on 24,000 saved
validation states from seed 182972. There are 7,978 waiting states, 2,338
unbound execution states and 13,684 bound execution states. It also evaluates
six reconstructed goal-specific execution features per non-waiting state.
No rollout, training update, live model request or formal trial is added.

Changing only the explicit actor latent while preserving the student proposal,
belief and physical inputs changes the selected action. For v6, the observed
execution action MSE against the frozen teacher rises from 0.117845 to
0.120355 during unbound execution, and from 0.098634 to 0.103617 during bound
execution, when averaged over the five wrong canonical goals. The corresponding
v10 values are 0.113884 to 0.116782 and 0.098532 to 0.103065. Paired execution
queries show the same direction. These are local input-dependence measurements,
not online success effects or a retrained no-goal baseline. Student proposals
and beliefs retain goal information, and deliberately mismatched inputs can be
out of distribution. The many queries are correlated, not independent trials.

Correct goal conditioning does not consistently improve waiting-action MSE.
For v6 it is 0.402816 with the matched latent and 0.398894 averaged over wrong
latents. For v10 the values are 0.404253 and 0.402894. Holding-teacher labels do
not depend on the requested target. This diagnostic therefore supplies no
evidence that explicit goal conditioning improves waiting. The deployed
waiting actor remains v1; the newer candidates concern execution recovery.
The actor consumes measured pose and student belief, without a raw RGB input.

The report and source hashes are archived in
`data/pose_latent_migration_development/controller_input_analysis/summary.json`.
Final actor selection remains pending the complete closed-loop development
comparisons. All six ongoing evaluation processes are producing fresh steps.


## Runtime contract preparation before the formal release

The evaluator now supports preparation-only initialization on a non-formal
seed. It loads the actual policy, measured-pose estimator, camera and physics
configuration, then records the complete manifest and source identities before
any scheduled trial or model inference. Preparation status is distinct from a
completed evaluation and contains zero scored episodes. Reserved formal seeds
are rejected in this mode. Active development snapshots are unchanged.

The release assembler requires completed preparations for all eight registered
conditions, matching source hashes, valid physics logs, a documented
development-only controller selection and the public frozen model-service
inventory. It rejects incomplete or changed contracts and never overwrites a
release. Python parsing, shell syntax, CLI import and the reserved-seed
preparation guard have been checked. Runtime initialization remains pending.

The two low-level measured-pose preparations have been queued on GPU 1 behind
the existing Astra v10 and v5 replay evaluations. The launcher is live. Six
agent preparations, final controller selection and formal rollouts remain to
be completed. This change adds no training result or success-rate evidence.


## Completed v6 public-plan controller replay

The v6 execution-recovery arm completes all forty fresh replay trials with
21 successes (11/20 and 10/20), compared with 24/40 (12/20 and 12/20) for
learned waiting alone. Both have valid physics logs, no excluded trial and
zero live model calls. Initial-state hashes match in all forty pairs. There
are seventeen shared successes, seven waiting-only successes, four v6-only
successes and twelve shared failures. The descriptive exact paired p value
is 0.548828125. This does not establish either superiority or degradation.

Both arms have ten drops overall. On the thirty-two schedules that contain a
correct requested binding, v6 has twenty-one successes, nine drops and two
pose false stops; waiting alone has twenty-four successes, seven drops and
one pose false stop. On the five schedules containing only inspection or
abstention, v6 drops once and reaches the horizon four times, while waiting
alone drops three times, abstains once and reaches the horizon once. Extra
survival without a supplied correct target does not count as semantic success.
All forty raw slots remain in the main denominator.

The separate replay export now contains eighty completed records in two
conditions. Three other arms are still running, for two hundred planned raw
records overall. The result supplies no consistent execution-recovery benefit
for v6 on this fixed-plan cohort, despite its 33/40 live-Astra development
result on different seeds. Final actor selection must retain both findings.
Fixed upper actions, matched initial states and paired counts do not establish
bitwise trajectory identity or replace a live-model evaluation.


## Completed v10 live evaluation and v5 replay

Inspection-balanced v10 completes the live Astra comparison at 31/40
(18/20 and 13/20), with one hundred actual calls, no service exclusions and
valid physics logs. Its failures include five unbound inspection drops, one
correctly bound drop, one wrongly bound drop, one pose false stop and one
wrong semantic stop. V6 completes 33/40. Their matched comparison has twenty-eight
shared successes, five v6-only successes and three v10-only successes, with
descriptive exact p=0.7265625. The offline inspection-action improvement does
not translate into an observed live completion advantage. V10 is not promoted
as an improvement. The actual-model export contains seventeen conditions and
680 raw records, all on the development split.

The completed v5 fixed-plan replay is 29/40 (15/20 and 14/20), with four drops,
four horizons, two wrong semantic stops and one abstention. It uses zero live
model calls and has valid physics with no exclusions. Relative to waiting-only
24/40, there are six v5-only successes and one waiting-only success (descriptive
exact p=0.125). Relative to v6 replay 21/40, there are eight v5-only successes
and no v6-only success (descriptive exact p=0.0078125, unadjusted across the
explored comparisons). All forty initial-state hashes match in each pair.
These are development diagnostics with fixed upper actions, not confirmatory
live-agent evidence. The replay export now contains 120 completed rows, kept
separate from actual-model results.

V5 previously completed live Astra at 31/40, compared with v6 at 33/40. Because
the replay favors v5 while live point estimates favor v6, two additional
forty-trial development conditions have been queued for adapted VLM and rule
VLM under the exact v5 evaluator snapshot and controller weights. No training
is repeated. Local model inference is serialized after existing rule-model
evaluation and qualitative work. These complete the provider comparison for
an alternative candidate before final controller selection. They do not
change the registered 320-trial formal matrix or its held-out seeds.

Both low-level runtime preparations have now completed in the actual simulator.
The original and latent students load with the measured-pose estimator,
calibrated camera and matching frozen Python sources. Both produce valid
physics logs, preparation manifests and zero scheduled episodes. Agent-cell
preparations and the final formal release remain incomplete.


## Actual Astra runtime preparations for both controller candidates

Eight preparation-only launches have completed successfully in the actual
simulator: the four registered Astra control combinations for v5 and for v6.
Every launch uses a non-formal seed and records zero scheduled episodes and
zero model inference calls. All four local candidate bridges still have zero
recorded inference requests. Each simulator log passes the capacity check.

For each actor candidate, every shared runtime-contract field agrees across
all four controller combinations. Loaded student, waiting actor, execution
actor, camera, pose estimator and stopping configuration stay the same while
the registered waiting/execution switches change. Between the v5 and v6 full
preparations, the execution-actor hash is the only contract-field difference.
This verifies initialization and configuration identity. It is not an online
component-effect measurement.

Together with the two completed policy-diagnostic preparations, each candidate
has six of the eight required formal-condition manifests ready. The two VLM
service preparations and final controller selection remain pending. The
release is not frozen and no reserved-seed trial has started. The public audit
is `data/pose_latent_migration_development/runtime_preparation_candidates.json`.


## Completed inspection-balanced v10 replay

The fixed-plan v10 arm completes 26/40 trials (13/20 in each seed), with seven
drops, three wrong semantic stops, one pose false stop, two horizons and one
abstention. All forty raw slots are retained. Physics logs are valid, no
service exclusions occur and there are zero live model calls. Against
waiting-only 24/40, it has five exclusive successes and three exclusive
failures (descriptive exact p=0.7265625). Against v6 replay 21/40, it retains
all twenty-one v6 successes and adds five (descriptive exact p=0.0625).

The separate mechanism export now contains 160 completed records in four
arms. V5 remains the highest observed replay result at 29/40 with four drops,
while both v5 and v10 have completed live Astra at 31/40. These development
comparisons do not establish a universally better controller. The independent
fixed-wait/v6-execution arm and v5 provider evaluations remain incomplete.
The actual-model export stays at seventeen conditions and 680 raw records.


## Prepared formal suite launcher

`run_formal_suite.sh` implements the registered GPU queues for all 320 raw
trials, using the already snapshotted evaluation runners. It checks the full
frozen release, source and checkpoint hashes, completed predecessor jobs and
unused output paths before launching. GPU 0 runs both policy diagnostics then
the two VLM conditions in sequence. GPUs 1 and 5 each run two Astra conditions
in sequence. Each worker keeps its log and exit status. The launcher performs
no retry, trial replacement or controller selection. Python evaluators and the
success contract are unchanged. Shell syntax has been checked locally and
remotely, and the help entrypoint works. The suite itself has not been executed
because final actor selection and frozen VLM preparation are still pending.


## Queued frozen VLM initialization

All currently planned cube student, QLoRA and controller training has finished.
A one-shot dependency pipeline now waits for rule-v6, rule videos and the two
v5 provider comparisons. It then releases only the three identified project
VLM services, verifies the frozen inference inputs, starts the fine and rule
services on ports 18794/18795 and prepares both v5/v6 controller candidates.
The four runtime preparations execute no scheduled episode and make no model
inference call. Existing development inference stays serialized on the local
GPU. The pipeline has started and its waiting state has been verified. The
service replacement and four preparations have not executed yet.

The final actor remains unselected. The 320-trial formal launcher is prepared
but has not been run. Astra-v6 and adapted-VLM-v6 both completed 33/40 on the
shared development protocol. Astra-v10 completed 31/40 and is not promoted as
an improvement. No Astra-superiority or formal-test claim follows from these
development observations.


## Formal-suite analysis entrypoint

The new post-processing entrypoint `summarize_formal.py` uses the statistics
helpers from the frozen evaluator snapshot. It requires all eight completed
conditions, forty reserved-seed records per condition, matching runtime and
source hashes, and valid physics logs. It exports the 320 raw records with
separate low-level and semantic metrics. The three-provider comparison uses
one symmetric triplet exclusion set. The four Astra cells use a separately
reported symmetric factorial exclusion set. Raw results always retain their
forty-trial denominators. Exact paired tests are explicitly unadjusted across
the listed contrasts, and do not assert superiority automatically.

Syntax and the CLI help entrypoint have been checked. No formal dataset exists
yet, so the full formal export has not been exercised. The frozen runtime
source snapshot and evaluation protocol were not modified.


## Completed five-arm fixed-plan controller comparison

All five controller arms have now finished their forty scheduled trials. The
new fixed-slow waiting plus v6 execution arm achieves 24/40, with 12/20 in each
seed, six drops, three pose false stops, three wrong semantic stops, three
horizons and one abstention. Both process and physics validity checks pass.
There are no service exclusions and no live model calls.

| Waiting control | Execution control | Success | Drops |
|---|---|---:|---:|
| Learned v1 | Student | 24/40 | 10 |
| Learned v1 | v5 | 29/40 | 4 |
| Learned v1 | v6 | 21/40 | 10 |
| Learned v1 | v10 | 26/40 | 7 |
| Fixed speed 0.15 | v6 | 24/40 | 6 |

The last two v6 waiting alternatives share all forty initial states and the
same recorded public action/delay plans. Fixed slowing has six exclusive
successes against three for learned waiting, with eighteen shared successes
and thirteen shared failures. The unadjusted exact paired p value is 0.5078125.
These observations do not demonstrate a benefit from learned waiting with
the v6 execution actor on this replay distribution. The strongest observed
replay controller remains v5, and final selection still awaits the live v5
provider comparisons.

The separate replay export now contains 200 records across five complete
conditions. Eight of the forty public plans never supply a correct requested
binding. All forty trials remain in every denominator. The results measure
control under recorded upper actions, not live Astra or VLM performance.


## Completed common-v6-stack provider comparison

The rule-VLM condition has finished 21/40 autonomous semantic successes
(10/20 and 11/20), with valid physics logs, a clean process exit, 167 actual
model calls and no service exclusions. Its nineteen failures comprise eight
drops during inspection before a semantic binding, five wrong semantic stops
and six horizons. All forty scheduled raw slots are preserved.

The completed same-v6-controller comparison is Astra 33/40, adapted fine-tuned
VLM 33/40 and rule VLM 21/40. Both comparisons against Astra match all forty
initial-state hashes and requested goals, and all forty triplets are eligible.
Astra shares all twenty-one rule successes and adds twelve exclusive
successes, with seven shared failures. Its unadjusted exact paired p value
against rule VLM is 0.00048828125. Against the fine-tuned VLM, twenty-nine trials
succeed in both, four succeed only with each provider, and three fail in both
(p=1.0). These are reused development seeds and explored model variants, not
a confirmatory formal test. The current data do not establish Astra superiority
over the fine-tuned VLM, nor equivalence between the two.

The actual-model development archive now contains eighteen complete conditions
and 720 raw records. It remains separate from the 200 zero-call fixed-plan
replay records and all historical-backend evidence. Final v5 provider results,
controller selection and the reserved-seed formal suite remain pending.


## Qualitative E-target handover case

All eighteen predetermined videos are now archived and fully decoded, six
per provider with matched initial states for T, E, R, L, S and H. The rule-VLM
set contains two successes, three drops and one horizon. Astra and the adapted
VLM each have five successes and one drop in these qualitative examples.
These recordings use the earlier v1 waiting, v3 execution and v2 completion
stack, not the later v6 controller. They remain excluded from quantitative
forty-trial comparisons, and all failure clips are retained.

For the matched E initialization, every provider correctly binds the requested
face to the green skill. Astra receives its sole response at step 174
(measured latency 5.785 seconds) and succeeds at step 230. The fine-tuned VLM
receives its response at step 4 (0.112 seconds) and drops at step 34. The rule
VLM receives its response at step 25 (0.824 seconds) and drops at step 45.
The three initial-state hashes match. Thus these two local-VLM failures occur
after correct skill delivery rather than while awaiting their response.

This is a qualitative clue about handover state, not evidence that longer
latency improves success or that Astra reasons better on this example. The
trajectories differ after initialization and the fine-tuned latent can differ
from the canonical skill vector. No controlled delay intervention was run.
The public record is `data/pose_latent_migration_development/qualitative_E_handover_case.json`.


## Fixed development selection rule and queued formal execution

The remaining pipeline is queued once after the existing VLM runtime
preparations. It considers only v5 and v6 and requires all six real-provider
development conditions to finish their forty raw trials with valid physics.
One common service-exclusion set is applied across all six cells for selection.
The primary criterion is mean eligible semantic success, equally weighting
Astra, adapted VLM and rule VLM. Ties prefer fewer eligible drops, then more
fixed-plan replay successes, then fewer replay drops. An exact remaining tie
retains v5. All raw results remain archived.

This criterion was specified with the completed v5 Astra and v6 provider
results already visible, before either v5 VLM condition began. It is a
development selection rule, not a blind preregistration of that development
study. No reserved-seed outcome may select or replace the controller. The
public rule is `data/pose_latent_migration_development/controller_selection_rule.json`.

After selection the pipeline assembles the full eight-condition frozen release
from actual runtime preparations and invokes the existing 320-trial launcher.
It exports complete formal results only after the suite finishes. There are
no automatic retries or replacement episodes. Prerequisite errors stop the
pipeline with logs. The supervisor is live and waiting for VLM preparation.
No controller has been selected and no formal trial has started.


## Completed v5 fine-tuned VLM comparison

The v5 execution actor with the same fine-tuned VLM completes 32/40 autonomous
semantic successes, with 16/20 in each seed. Both physics checks pass, the
process exits cleanly, and none of the forty raw trials is excluded. The 121
actual calls lead to four wrong-binding stops, three drops during inspection
before binding, and one horizon without a declared target.

Against v6 at 33/40, all forty initial states and goals match. Twenty-eight
trials succeed in both, four succeed only with v5, five only with v6, and three
fail in both. The unadjusted exact paired p value is 1.0. This development
comparison provides no clear separation between the two execution actors for
the fine-tuned VLM. The rule-VLM v5 condition has started, and the common
controller selection still waits for its full forty trials.

The live-model development export now retains nineteen completed conditions
and 760 raw records. The separately compiled supplement keeps the exact prior
summary used at compilation under `supplement_20261029_inputs/summary.json`.
No formal result or controller selection is reported here.


## Completed development selection and pre-trial startup repair

The rule-VLM v5 condition is complete: 21/40 successes (8/20 and 13/20),
158 actual calls, six drops before binding, six wrong semantic stops and
seven horizons. Both physics checks pass and there are no service exclusions.
The live-model development archive now contains twenty complete conditions
and 800 raw records. All prior 760 records were compared field by field and
are unchanged. Historical snapshots and the compiled supplement's input
snapshot remain archived separately.

The documented development rule selected one common v6 execution actor.
Across Astra, adapted VLM and rule VLM, v5 completes 31, 32 and 21 trials,
while v6 completes 33, 33 and 21. The equally weighted totals are 84/120 and
87/120, with fourteen and thirteen drops. No tiebreak or formal outcome was
used. The selection rule was set after some development results were visible,
so this is development selection, not blind preregistration. The public
selection record is `data/pose_latent_migration_development/controller_selection_record.json`.
Astra and the fine-tuned VLM remain tied on this development panel.

The original formal launch failed during runtime-contract validation before
any scheduled episode, action or model call. Python tuples in the goal bank
become lists when serialized to JSON. The runtime compared these representations
directly and rejected identical goal content. The repaired validator compares
the same JSON representation on both sides. It changes no weights, targets,
thresholds, seeds, timing rules or request budgets. The original release and
failed startup logs are preserved.

The separate `formal_serialization_repair_20261029` pipeline was launched at
04:30 UTC on 29 October. It rebuilds all eight preparations with zero trials
and zero model calls, checks every frozen condition contract against the
original release, and then runs the unchanged 320-trial suite as
`pose_latent_formal_20261029_v1_jsonfix`. It has no automatic retries or
replacement episodes. Preparations are currently running. Formal results
are pending and must be reported separately from development selection.


### Formal suite started, 04:35 UTC

All eight runtime preparations passed. One dedicated Astra service had exited
with status 143 before its preparation and was restored from the identical
service snapshot. The failed preparation contains no scheduled trial or model
call. Its logs are retained separately from the successful preparation.

The repaired release hash is
`07922f1aa5522fd665b6fc0a70da6aa75e6d52801b387372427a18d6e3e0ef28`.
Every condition contract, the selected actor, service inventory, external
source hashes and frozen requirements match the original release. The only
evaluator change normalizes the contract's JSON representation.

Three GPU queues launched at 04:35:38 UTC. The 04:36:22 startup snapshot shows
live physics steps in the released-student diagnostic, full Astra method and
fixed-slowdown Astra condition. Seven raw slots had finished at that snapshot.
Remaining conditions follow in their queues. This confirms rollout startup,
not complete-condition performance or passed end-of-run physics validation.
The full suite contains 320 raw slots, forty per condition on seeds 183101 and
183102. Formal release, launch and amendment records are archived in the
paper data directory. Complete formal results remain pending.


## First complete formal comparison: local student transfer

Both low-level conditions finished all forty reserved trials with valid physics
logs and clean exits. The released pose-goal student reaches 39/40 targets with
one drop. The migrated pose-latent student reaches 35/40 with five drops.
All forty recorded causal initial-state hashes, target identities and target
quaternions match. No episode began inside the reach threshold.

Thirty-five pairs reach with both policies, four reach only with the released
policy, and one fails with both. The observed migration difference is minus ten
percentage points. The unadjusted exact paired p value is 0.125. This sample
does not establish equivalence or rule out degradation. Every failure terminates
through `object_out_of_reach`; the four additional latent-policy failures cover
R, S, H and E. These diagnostics contain no upper-model request, semantic
binding decision or waiting controller, so those mechanisms cannot explain
the added failures. The specific policy failure cause remains unresolved.

This comparison uses the released short diagnostic horizon and reach-before-drop
criterion. It is separate from the 120-second autonomous semantic criterion.
It must not be presented as a language success rate or combined with development
records. The complete raw evidence and a descriptive failure analysis are in
`data/pose_latent_migration_formal/low_level/`. The six formal semantic conditions
continue with the same frozen student, services and controllers. No weight,
threshold, seed or trial was changed in response to this result.


### Training/runtime differences remain hypotheses

The archived student completed 200 interface updates and 400 DAgger updates.
Those rollouts use the released simulated perception-noise model. They do not
run the RGB CNN. The runtime diagnostics use that CNN for both policies, with
no extra synthetic pose noise. The direct-latent model also replaces explicit
target and relative-target quaternion channels with a learned 768-to-64 goal
projection. These input and architecture differences are verified in the saved
checkpoint and code, but they do not identify the cause of the four added losses.

The final DAgger training action MSE is 0.529, and the last-twenty-update mean
is 0.520. Training loss is not task success and does not establish that a longer
run will help. The measured-pose control collections trained recovery actors,
not this student checkpoint. Any later measured-pose student adaptation needs
a separate development comparison. Current formal weights and outcomes remain
unchanged. The audit is `data/pose_latent_migration_formal/low_level/training_runtime_analysis.json`.


## Completed formal Astra fixed-slowdown condition

The fixed-slowdown arm, without learned execution recovery, completes 23/40
semantic successes, with 11/20 and 12/20 across the two reserved seeds. Both
runs exit cleanly with valid physics checks and no service exclusion. Its
seventeen failures comprise sixteen drops and one pose false stop. Four drops
occur before the first skill, ten during inspection without a semantic binding,
and two after a correct binding. All forty raw outcomes are retained.

The 85 actual Astra calls report 138,808 total tokens, including 134,054 input
and 4,754 output tokens. Measured client latency has median 5.118 seconds,
p95 6.887 seconds and maximum 8.855 seconds. These are recorded model usage
and latency, not currency or account-wide quota. Every call has usage data.
The complete condition and per-call resource records are archived under
`data/pose_latent_migration_formal/fixed_slow_no_execution_recovery/`.

Other semantic arms remain incomplete. Group-wide matched exclusions and
paired method/module comparisons are therefore pending. This completed raw
rate must not be compared with a selected partial prefix of another arm.
The GPU queue proceeds to fixed slowdown with learned execution recovery
using the same frozen plan, weights and limits.


## Recorded service timeout and unvisited-tail continuation

The full Astra condition stopped after 32 scheduled raw trials. Its second
seed contains twelve records, ending with trial 183102011. The second request
in that trial returned `bridge_http_503: ReadTimeout` after 120.487 seconds.
The evaluator recorded `service_error`, and its simulation fragment passed
physical-log validation. The service remained reachable. The original request,
failed episode, twelve-row fragment and earlier seed are preserved.

The stopped worker was confirmed absent. At 05:28 UTC, a separate recovery
worker continued only the eight unvisited indices 12--19 of seed 183102,
using the already frozen evaluator's `start_episode` support. It keeps the
same release, source snapshot, checkpoint, service, timing and stopping rules.
The 183102011 record is neither rerun nor replaced. The raw condition still
contains forty planned slots. Final matched analyses apply the frozen symmetric
exclusion rule to this key and any other recorded service-excluded keys.

The worker hashes all existing episode records and logs before continuation
and checks them unchanged before declaring the condition complete. It then
executes the originally queued learned-waiting-only condition. Original queue
failure metadata is retained alongside the separate recovery launch. The other
two GPU queues continue independently. The recovery has no automatic retry.
Evidence is under `data/pose_latent_migration_formal/queue1_recovery/`.


## Completed formal fine-tuned VLM condition

The frozen fine-tuned VLM full stack completes all forty reserved trials: **30/40 raw semantic successes**, with 15/20 in each seed. The remaining outcomes are five drops, three horizons and two incorrect semantic bindings. Three drops happen during inspection before binding and two after a correct binding. Both physical logs pass the capacity checks, with no service exclusions in this condition.

The run records 149 actual local model calls. Client inference latency has median 0.113 s, 95th percentile 0.150 s and maximum 0.352 s. Token usage is not reported by this local service, so it is stored as null. This is not a cross-provider compute-cost comparison.

The complete raw records, manifests, validity reports and call resource summaries are archived under `data/pose_latent_migration_formal/finetuned_vlm_full/` in the paper repository and `outputs/pose_latent_student_v1/formal_completed_conditions_20261029/finetuned_vlm_full/` in the project. The final provider table remains pending: the previously recorded Astra service error will be excluded symmetrically according to the unchanged protocol. These completed outcomes do not select or update any weights.


## Completed formal Astra condition and second transport recovery

The full Astra stack now contains all forty reserved raw slots: **29 successes, ten drops and one service error**. The two seeds contain 13 and 16 successes respectively. The own-condition service-eligible rate is 29/39 (74.4%). This is not yet the finalized provider panel because group-wide exclusions also depend on the rule VLM condition. The fine-tuned VLM raw result remains 30/40. These data do not establish Astra superiority.

All ten Astra drops occur during inspection before any target binding is established. Nine occur outside a pending model call and one while a call is pending. Every dropped trial had already received at least one inspection skill, so none is a startup drop before the first skill. There are no false stops or horizons in this condition. Zero observed drops after binding is conditional on reaching that stage and does not establish perfect low-level control. The public-decision audit is archived in `data/pose_latent_migration_formal/astra_full/phase_decision_analysis.json`.

The full condition records 94 model requests. Ninety-three return token usage, totaling 154,859 reported tokens, with one failed call lacking usage. Client latency has median 5.142 s and 95th percentile 7.145 s. The maximum is the preserved 120.487 s timeout. This is measured client latency in simulation, not a hardware real-time result or exact billed cost.

The separate fixed-waiting plus learned-execution condition subsequently stopped on upstream `SSLError` at trial `183102000`. The failure and preceding twenty trials remain unchanged. A single recovery worker, PID 639323, runs only unvisited indices 1 through 19 of seed 183102. It changes only output path, start index and remaining episode count. Its source digest is `a7bb6447804c9e3c71ba3c5e8441fb475c64941811c06b7d24f77490c30c5a66`. No trial is replaced and no model, success rule, source snapshot or physical setting is changed.

The recovery worker performs the unchanged final export after every logical condition finishes. Original failed queue status files remain as historical evidence. The final exporter validates every fragment, the exact forty reserved keys, source and weight identity, and symmetric exclusions. The provider triplet and Astra factorial retain their separate registered exclusion sets. As of 05:43 UTC, 228 of 320 raw slots are recorded. Rule VLM, learned waiting only and fixed waiting with learned execution are all active. No additional cube training is pending for this frozen release.


## Queued student sensor-distribution adaptation study

A separate development plan was recorded after observing the current formal low-level regression. It compares additional simulated-noise DAgger with DAgger using the same frozen calibrated RGB-CNN pose estimator used at deployment. Both start from the identical released latent checkpoint and original PPO teacher. The architecture, continuous 768-dimensional goal input and action interface stay fixed. The measured path uses no object-state pose fallback.

Each condition has 400 updates, sixteen environments, forty-five rollout steps, four epochs and learning rate 0.0003, giving 288,000 training transitions per condition. The final checkpoint is used directly. Training seed 183201 is separate from both the ongoing formal seeds and fresh development seeds 183221/183222. All four development comparators—original released student, frozen latent student, extra simulated training and extra measured training—receive forty trials each with identical RGB-CNN input and the existing low-level reach metric. These are development diagnostics, not language success.

The new queue (PID 646172) is alive and waiting for the verified 320-slot formal export before allocating GPUs 3 and 4. It also waits for those devices to be free of other compute processes. This is a finite training workflow, not a recurring monitor. No new training has run yet. Syntax and shell parsing pass, and the active formal source and checkpoint hashes are unchanged. Actual training/runtime validation remains pending.

No new candidate is deployed automatically. The measured-pose candidate must improve reach counts over both the frozen latent student and the equally trained simulated-pose control without increasing drops before proceeding to controller adaptation. This is a development admission rule, not a statistical superiority claim. Recovery actors are bound to the old student, so a changed student also requires new control-data collection, fitting and agent validation. Future agent seeds 183301/183302 remain unvisited. Every unsuccessful candidate is retained.

The exact plan, source identities, launch and waiting-process evidence are in `data/student_pose_adaptation_20261029/` in the paper repository and `outputs/pose_latent_student_v1/student_pose_adaptation_20261029/` in the project. This study may diagnose a transfer issue. It does not retroactively explain the earlier formal result or change its frozen metric.


### Canonical goal representation audit

A deterministic audit of the frozen student on its six existing canonical training vectors finds five nonzero centered singular values (the maximum centered rank for six points), a minimum encoded pairwise Euclidean distance of 1.560 and no components above absolute tanh value 0.98. This does not indicate a collapse of those six goal encodings. It does not test arbitrary VLM latents, prove adequate action conditioning or identify the cause of the transfer regression. No simulation trajectory or API request was generated. The exact measurements and checkpoint/cache identities are saved in `data/student_pose_adaptation_20261029/goal_representation_audit.json`. The equal-budget sensor adaptation plan remains unchanged.


## Completed formal execution-only ablation

The fixed-slow-waiting plus learned-execution condition completes all forty raw slots with **28 successes and eight drops**. The remaining outcomes are one pose false stop, one incorrect semantic binding, one abstention and the preserved SSL service error. Both seed fragments and the nineteen-slot continuation pass the physical checks. The raw seed success counts are 16/20 and 12/20. The condition records 106 actual model calls, with token usage reported for 105 calls (190,658 reported tokens). No replacement trial was run.

| Completed Astra condition | Raw success / 40 | Drops | Own service errors |
| --- | --- | --- | --- |
| Fixed slowdown, no execution recovery | 23 | 16 | 0 |
| Fixed slowdown with learned execution | 28 | 8 | 1 |
| Learned waiting and learned execution | 29 | 10 | 1 |

All forty initial-state hashes and task goals match across these three conditions. This is a descriptive raw table. The registered factorial still awaits learned-waiting-only results and symmetric group-wide exclusions. It does not yet establish significance, an interaction or an advantage from combining both modules. The complete new condition is retained under `data/pose_latent_migration_formal/learned_execution_only_with_fixed_slow_waiting/`.

The four startup drops occur on the same initial-state keys in both fixed-waiting conditions: 183101007, 183102004, 183102012 and 183102019. With the full stack, all four reach the first skill delivery. Two subsequently succeed and two drop during inspection. Thus avoiding an initial-wait failure is not equivalent to completing the task. These are paired initial conditions with distinct live model calls and physical histories, not a same-trajectory causal test. Every case is listed in `data/pose_latent_migration_formal/startup_case_progression.json`.

The execution recovery supervisor has finished its condition and is waiting for the remaining logical conditions before running the unchanged final exporter. The original SSL fragment and failed queue status remain preserved. Six of eight formal conditions are now complete. The successor student training still waits for the full verified export and has not started.


## Waiting-only transport recovery and development handoff audit

At 06:29 UTC, the learned-waiting-only formal condition stopped after thirty-one recorded slots. Trial 183102010 returned `bridge_http_503: SSLError` on its first request after 0.542 seconds. The fragment retains eleven second-seed records and passes the physical-log check. Its worker and the previous finalizer were confirmed absent, and the original Astra service passed a health check. The service error is retained for the frozen symmetric exclusion rule.

A separate one-shot worker (PID 666219, launched at 06:31 UTC) continues only indices 11 through 19 of seed 183102 on GPU 1. It preserves and hashes the first thirty-one records and their logs, uses the same frozen sources, models and protocol, and does not retry trial 183102010. Its source SHA-256 is `cb1ef0fe1b77ac569653d6b093b3bbd8f2654c6df06a9afc3871948d9f72bc8f`. After all logical conditions finish, it runs the unchanged summarizer into the same canonical export directory expected by the queued student study. Historical failed supervisors remain archived. Evidence is under `data/pose_latent_migration_formal/queue1_waiting_recovery/`. At 06:34 UTC the live suite contains 296 of 320 raw records.

A separate read-only audit of all forty Astra v6 development trajectories identifies three physical drops, all during inspection before target binding. Two occur 22 and 24 simulated steps after the most recent skill delivery, and one occurs 87 steps after delivery during the next pending request. The corresponding counts are based on public delivery timestamps and recorded physical steps. They motivate checking the transition between holding and active observation, but they do not demonstrate a causal action jump: the old agent step logs omit executed joint actions. No cutoff, weight or policy is selected from these three events. The development cohort, its archive hash, all forty outcome records and phase-exposure counts are retained under `data/development_handoff_audit_20261029/`. No formal state is reused for this audit or for training.


### Existing control-data transition coverage

A read-only audit of the completed v6 control collections finds 48,000 training states and 24,000 validation states, with 15 and 3 recorded drops respectively. After excluding environment-reset boundaries and requiring consecutive physical steps, there are only 57 training and 45 validation transitions from pending to executing. Of these, 33 and 30 respectively are unbound inspection transitions. This limits the diversity of direct handoff supervision despite the large number of individual states.

The normalized command difference is larger at unbound handoffs than in unchanged execution in these mixed-behavior trajectories. These are raw policy commands, not applied joint angles or forces. The audit also reports commands clamped to [-1, 1] separately. Most handoffs are not followed by a drop: only one training and one validation handoff have a recorded drop within the next thirty consecutive steps. This does not establish that a command jump causes failure, and state-level windows are correlated. Future controller collection should cover both transitions and sustained inspection. No balancing weight, cutoff or checkpoint has been chosen using this audit. Exact counts, raw-command summaries, limitations and the analysis source are in `data/development_handoff_audit_20261029/collection_actions.json` and `audit_collection_actions.py`.


## Completed successor student and paper export, 2026-10-29

Both 400-update adaptation arms and all 160 fresh low-level diagnostic trials have completed with valid physics logs and matching initial state hashes. Released student: 39/40 reach, 1 drop. Frozen latent: 38/40, 2 drops. Extra simulated DAgger: 39/40, 1 drop. Extra actual-CNN DAgger: 40/40, 0 drops. The CNN stays frozen. The candidate meets the development gate but has not been deployed. These are not Astra semantic results and do not replace cube-v1.

Original cube-v1's 400 trials and source manifests are archived under `outputs/frozen_results/RSLCube/cube-v1-20261029`. Successor evidence is independently archived under `outputs/frozen_results/RSLCube/student-adaptation-dev-20261029`. The CVPR 2027 paper repository contains sanitized evidence and explicit TODOs for unmeasured experiments.
