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

At the first status check, the process was alive at learning iteration 2/2500
with an estimated remaining time of roughly seven hours.  No final teacher
metric is recorded until the run and the fixed evaluation protocol complete.
