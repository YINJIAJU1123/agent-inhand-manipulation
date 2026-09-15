# Camera-free state-policy baseline protocol

`evaluate_state_baseline.py` is the reference evaluation for the privileged
semantic Revo3 controller. It does not enable cameras and does not require RTX
rendering. Run it from the repository root with the Isaac Lab Python runtime:

```bash
python scripts/rsl_rl/evaluate_state_baseline.py \
  --checkpoint logs/rsl_rl/brainco_hand/<run>/model_1000.pt \
  --episodes-per-face 100 --num-envs 64 --seeds 0,1,2 --headless \
  --report outputs/state-baseline.json
```

The default experiment evaluates all six target faces, 100 episodes per face,
and seeds `0,1,2`. A face is fixed for each simulator block, so the completed
report has exactly the same number of episodes for every face and seed. The
vectorized episode quota is assigned to slots before stepping; one slot cannot
replace another slot's trial merely because it succeeds sooner. A target is
never resampled after success during this evaluation.

## Goal and episode settings

The evaluator applies these settings to every block:

```text
fixed_target_face       = current face block
goal_yaw                = CLI value, or None for the environment's random yaw
max_consecutive_success = 0
freeze_goal_for_episode = true
goal_hold_time_s        = 0.5
record_eval_metrics     = true
```

`success_tolerance` remains the environment value (`0.16` radians in the
semantic Revo3 configuration). An episode ends only on object drop or the
configured episode horizon. Consequently, a policy that reaches the goal in
one step still has to keep the object and target aligned for the remaining
horizon to receive `held_at_end`.

`--goal-yaw <radians>` evaluates a deterministic yaw. Omitting it evaluates the
normal random in-plane yaw distribution. Use separate reports for the two
conditions; do not mix them in one aggregate.

## Per-episode metrics

The environment writes `semantic_metrics` before Isaac Lab resets a terminal
slot. The evaluator clones these tensors before any episode bookkeeping. The
required fields are:

- `orientation_error`, `object_distance`: terminal/pre-reset geometric values;
- `goal_reached`: this-step instantaneous orientation threshold;
- `dropped`: object-distance failure;
- `hold_complete`, `hold_steps`: continuous threshold validity and elapsed hold
  steps (`ceil(0.5 / step_dt)` steps for the default protocol);
- `target_face`, `goal_rotation`: the target belonging to the episode that just
  ended.

The report stores these derived outcomes for every episode:

- `instant_reach`: reached the orientation threshold at least once without a
  drop;
- `continuous_hold`: completed the configured continuous hold without a drop;
- `held_at_end`: was in the completed hold state on the terminal transition;
- `success`: currently equal to `held_at_end && !drop`;
- `drop`, `steps`, simulated `time_s`, final and minimum orientation error;
- mean clipped action L2, mean action-slew L2, and mean **actual** target
  movement in radians. The target movement uses the environment's normalized
  action scaling and `act_moving_average`, matching the `cur_targets` update;
  it is not the norm of the raw policy output.

Each Bernoulli outcome has a 95% Wilson interval. The JSON contains aggregate,
per-face, and per-seed summaries plus the complete episode records.

## Stress controls

The following CLI overrides map to fields that exist in the current environment
configuration and are recorded in the manifest:

```text
--object-scale S
--object-density D
--static-friction F
--dynamic-friction F
--reset-position-noise N
--reset-dof-pos-noise N
```

The current environment samples initial X/Y object rotation in
`InHandManipulationEnv._reset_idx` without exposing a configuration field.
Therefore `--initial-rotation-noise` intentionally fails with a clear error;
the evaluator never pretends that an unsupported stress setting was applied.

## Reproducibility and failure behavior

Every report includes the absolute checkpoint path and SHA-256 digest, current
Git revision, exact command arguments, seed/face schedule, runtime versions,
stress overrides, and a JSON-safe environment configuration snapshot. Output is
written with `allow_nan=False`; incomplete quotas or missing protocol metrics
produce a nonzero exception rather than a partial success claim.

Running against the old semantic environment is rejected when any of the hold
or terminal pre-reset fields is missing. In particular, a missing
`hold_complete` field is never treated as an instantaneous success. This keeps
old one-step success reports separate from the continuous-hold baseline.
