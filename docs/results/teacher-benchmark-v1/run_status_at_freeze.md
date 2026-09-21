# Revo3 teacher continuation — 2026-09-21

Objective: stabilize the single-Cube Revo3 privileged-state teacher before
student distillation. No real-hand motor commands are involved.

## Recovered state

- Source conversation: `01a0beda-231d-7711-90c4-4b08b568bb02`.
- Remote root: `/home/jiaju/src/RevoLab` on the existing 5090 host.
- Latest checkpoint: `logs/rsl_rl/brainco_hand/2026-09-20_18-22-13_revo3_viserdex_cube_formal/model_500.pt`.
- Read-only remote check confirmed no running Revo3 teacher/evaluator initially.
- Prior training log ends with a Kit mutex assertion and timeout; it did not
  complete the planned training.
- Original training: 10-second episodes, 0.16-radian tolerance, zero hold time,
  no action clipping, EMA 0.35 per physics substep, 21-D action, 158-D state.

## Changes

- Added repeated-goal evaluation to `evaluate_state_baseline.py`, retaining the
  original fixed-face hold evaluation mode.
- Added pre-reset success-event counts, CS mean/std, per-face counts and explicit
  horizon censoring, rollout progress, bounded simulator steps, CPU thread limit,
  configurable tolerance/dwell, and complete original command-line manifest.
- Fixed training dashboard CS to use completed-episode moving average; report
  active-episode count separately, plus drop/hold/orientation diagnostics.
- Added `run_teacher_stage.sh`: smoke (8 envs, 3-second horizon), baseline
  (128 envs, 30 seconds), and candidate continuation (2048 envs, 1500 additional
  iterations, 0.2-second hold, shaping 2.0, save every 100 iterations).
- Remote pre-change evaluator/environment backed up under
  `outputs/teacher_20260921/code_before/`.

## Validation and launch status

- Five pure-Python protocol tests pass with pytest plugin auto-loading disabled
  (the workstation's unrelated ROS plugin otherwise requires missing `lark`).
- Python compilation, shell syntax and git whitespace checks pass.
- Initial smoke attempts exited before simulation because the installed Kit
  requires `OMNI_KIT_ACCEPT_EULA=YES` and a user-owned `TMPDIR=/home/jiaju/tmp`.
  Both settings are now in the launcher, matching the existing installation.
- Corrected smoke launch was submitted successfully over SSH, bounded by a
  10-minute timeout plus 30-second termination grace. Output:
  `outputs/teacher_20260921/smoke.log`, expected report `smoke.json`.
- The downstream SSH banner at forwarded port 22031 intermittently timed out.
  Connection recovered; the earlier attempts were confirmed to have exited.
  Corrected smoke completed: 8 episodes, 3-second horizon, zero drops, CS 1.125.
- Model500 baseline completed: 128 episodes, seed 101, 30-second horizon,
  tolerance 0.4 rad, zero dwell. CS 7.3125 ± 5.5138; 28/128 drops (21.875%,
  Wilson 95% interval 15.59–29.80%); 100 surviving episodes are horizon-censored.
  Face success counts: [140,153,169,145,142,187]. This is not lifetime CS.
  Mean joint velocity RMS 2.137 rad/s; mean object angular speed 3.833 rad/s.
- Reports downloaded to local `outputs/teacher_20260921/`.
- Formal continuation started after baseline completion and confirmed at
  iteration 500/2000 (11.36 s/iteration initially). The new run is
  `2026-09-21_10-35-09_revo3_teacher_hold02_s21`; its first checkpoint exists.
  Remote log: `outputs/teacher_20260921/train.log`. Initial ETA approximately
  4h44m, subject to shared GPU load. Never launch a duplicate without checking.
- Final logging change (capture hold fraction before goal resampling), evaluator,
  summary and launcher changes have been uploaded.

## Next actions when the connection is restored

1. Check the ongoing training log and new checkpoints. Do not relaunch it.
2. Evaluate candidates with identical held-out seeds under repeated and fixed-face
   hold protocols. Compare drops, CS, held-at-end and control metrics, then refine
   the objective if needed. Do not declare stability from PPO reward alone.
3. Use `TEACHER_CHECKPOINT=<path> TEACHER_REPORT_TAG=<unique-tag> bash
   scripts/rsl_rl/run_teacher_stage.sh baseline` or `hold` for comparison.
4. Extend evaluation to additional seeds and physical perturbations before
   freezing a teacher for student training. Stability has not yet been achieved.

## Latest live check — 2026-09-21 19:16 China time

- Remote training process is still running on `brainco`; no duplicate launch was made.
- Candidate `revo3_teacher_hold02_s21` reached iteration **800/2000** and produced
  `model_800.pt`.
- Latest rollout dashboard: mean reward 967.11, consecutive successes 2.6736,
  active-episode successes 1.2988, drop fraction 0.0010, hold fraction 0.0122,
  orientation error 1.3655 rad.
- The run reported about **2h44m** remaining at that check. These are training
  rollout diagnostics only; no held-out evaluation has yet shown a stable teacher.

## Live continuation — 2026-09-21 19:42 China time

- The same teacher process is still running and reached **983/2000** iterations;
  the latest saved checkpoint observed was `model_900.pt`.
- Recent training diagnostics were consecutive successes about 3.2–3.5,
  drop fraction 0–0.001, hold fraction about 0.01–0.015, and orientation error
  about 1.31–1.36 rad. These remain rollout diagnostics, not an acceptance result.
- A remote post-training evaluator was started in
  `scripts/rsl_rl/teacher_posttrain_eval.sh`. It waits for the existing training
  PID, then evaluates the newest checkpoint with repeated CS (zero dwell),
  repeated CS with 0.2 s dwell, and balanced six-face 0.5 s hold. No second
  training process was launched.
- A local syntax/protocol check after this continuation passed: **5 tests passed**
  and all modified Python files compiled successfully.

## Final candidate evaluation — 2026-09-22 00:00 China time

- The continuation completed at 2000 iterations. The newest evaluated checkpoint
  was `model_1999.pt`, SHA-256
  `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`.
- Repeated reorientation, 128 episodes, 30 s horizon, random yaw:
  - zero dwell: CS **25.66 ± 11.06**, drop **36/128 (28.1%)**;
  - 0.2 s dwell: CS **23.29 ± 7.43**, drop **23/128 (18.0%)**.
- Balanced six-face hold, 64 episodes per face, 0.5 s hold:
  - instantaneous reach **364/384 (94.8%)**;
  - continuous hold **360/384 (93.8%)**;
  - held at terminal end **293/384 (76.3%)**;
  - drops **18/384 (4.7%)**.
- The candidate is a substantial improvement over `model_500.pt` on repeated CS
  (7.31 mean previously), and is a strong state-teacher candidate. It is not yet
  frozen as the final teacher: the zero-dwell repeated protocol still has high
  transition/drop stress, and the held-at-end rate is not yet near-perfect.
- Final JSON reports were copied to local `outputs/teacher_20260921/final_reports/`.

## Camera rollouts — 2026-09-22

- Using the same `model_1999.pt`, three random-target camera rollouts were
  recorded remotely at 256x256, 30 Hz, 900 frames (30 s) each, then copied and
  encoded locally as MP4 under `outputs/teacher_20260922/videos_mp4/`.
- `random_seed101`: 10 reached transitions, 4 drops, 14 terminal transitions.
- `random_seed102`: 13 reached transitions, 1 drop, 14 terminal transitions.
- `random_seed103`: 14 reached transitions, 6 drops, 20 terminal transitions.
- These are qualitative privileged-state teacher replays in the camera-enabled
  scene; they are not visual-policy results. Full frame traces and metadata are
  under `outputs/teacher_20260922/video_frames/`.
