# VisERDex teacher v2 training status

Updated: 2026-09-22

## Configuration

- Task: `BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0`
- Code revision: `b3e6033` (`Add full VisERDex-style Revo3 teacher protocol`)
- Training host: `brainco`, RTX 5090
- Environments: 2048
- Seed: 123
- Maximum iterations: 2500
- Goal tolerance: 0.1 rad
- Per-goal timeout: 10 s
- Maximum consecutive successes: 50
- EMA alpha: uniform `[0.08, 0.20]`
- Action delay: uniform 0–2 policy steps

## Validation and launch

The 16-environment, 3-iteration smoke run completed successfully.  It built the
158-dimensional policy input and 21-DoF actor, and emitted the randomized EMA,
latency, orientation, velocity, and drop metrics.

The full run was started in the background on `brainco` with:

```text
bash scripts/rsl_rl/run_viserdex_teacher_v2.sh train
```

Remote log:

```text
/home/jiaju/src/RevoLab/logs/viserdex_teacher_v2_full.launch.log
```

The corrected protocol uses the VisERDex PPO schedule (24 steps per
environment, 1024-1024-1024-512 MLP, `gamma=0.998`, adaptive `1e-3` learning
rate), a 244-dimensional Revo3 state input with four-step action context and
randomized action properties, and a success-driven regularization curriculum.
The first corrected 512-environment canary completed without numerical
explosion, but did not yet reduce orientation error.  A deterministic
bootstrap also stayed near 2.2 rad, so a prior successful 158-dimensional
state teacher was embedded into the 244-dimensional network as a warm start.
That checkpoint reached 4/4 fixed-face successes at 0.16 rad in the corrected
environment, proving the action and goal interfaces are compatible.

When the warm start was resumed with the original `-50` torque proxy scale,
the moving curriculum reached about 0.4 and the torque term became roughly
`-22` per step; value loss grew to `1e4--1e5`.  This is the remaining training
bug.  The next run reduces torque/work and related proxy regularizers by one
order of magnitude and restarts the adaptation from the validated warm start.

## Final run check

The 2500-iteration process finished normally and wrote `model_2499.pt`. The
rollout did **not** converge to a usable teacher:

- consecutive successes stayed around 0.00;
- curriculum stayed at 0.00;
- orientation error stayed around 2.20--2.23 rad;
- drop fraction was roughly 3--4% near the end.

The original 2500-iteration checkpoint and the first corrected canary are
diagnostic only; neither is a final teacher.  A final result is recorded only
after the restarted adaptation remains numerically stable and passes the fixed
face/seed evaluation protocol.

## Corrected adaptation result

The warm-start adaptation completed on `brainco` with 512 environments, seed
123, and the full randomized EMA/action-delay protocol.  The run directory is

```text
/home/jiaju/src/RevoLab/logs/rsl_rl/brainco_hand/2026-09-22_09-31-56_viserdex_teacher_v5_scaled
```

The best checkpoint in the matched repeated evaluation was `model_750.pt`
(SHA-256 `d263d7c65f405d7eb385c782b61b91ebd14afd5be45e6608c8a97d027d3c69cd`).
With 16 episodes, seed 101, 30 s horizon, random face plus random in-plane
yaw, 0.1 rad tolerance, and no hold dwell, it achieved mean consecutive
successes 17.75 (std 6.09, range 4--25), with 2/16 drops.  The final
`model_1648.pt` checkpoint was lower at CS 14.50 (std 8.43) with 3/16 drops,
so it is retained as a late-run diagnostic rather than the selected candidate
(SHA-256 `583435836a50deba69a6fe71aec0d629caa9c3dcf658b4bad619d60992c41334`).

For the selected model, a four-episode fixed-face, 10 s hold check reached the
goal in all 24 trials with no drops.  The held-at-end rates for faces 0--5 were
0%, 75%, 75%, 25%, 50%, and 50% (mean 45.8%).  This exposes a remaining
stability gap even though repeated reorientation improves; the six JSON reports
and checkpoint hashes are archived under `outputs/viserdex_teacher_v5/`.
