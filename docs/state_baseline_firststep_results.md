# State-only first-step result

The hold-aware state policy was trained in the `VLMrotation` Isaac Sim 5.1
container on one RTX 5090, with 512 environments, 158-dimensional state
observations, 21-dimensional actions, random reset yaw, and a 0.5 s continuous
hold requirement. No camera or rendering input was used.

The scratch run completed 1000 iterations. It was useful as a negative control,
but its final checkpoint did not reach face 0 in the hold-aware evaluator.

The usable teacher was obtained by warm-starting the actor and actor
observation normalizer from the previous reorientation policy, while using a
fresh critic and optimizer. At warm-start checkpoint 250, one episode per face
was evaluated with a fixed seed and 64 vectorized environments:

| metric | result |
| --- | ---: |
| instant reach | 6/6 |
| continuous 0.5 s hold | 5/6 |
| held at episode end | 5/6 |
| drops | 0/6 |

Faces 0–4 completed the hold. Face 5 reached the tolerance but did not hold it
for the required duration. Continuing the same warm-start run to checkpoint
499 did not fix that single episode (face 5 remained instant-reach true,
continuous-hold false), so it is recorded as an open robustness issue rather
than hidden by changing the evaluator.

The checkpoint used for the 5/6 result is on the remote container at:

`/tmp/revo-teacher-warmstart/logs/rsl_rl/brainco_hand/2026-09-15_21-47-37/model_250.pt`

Its SHA256 is
`0328cf82b23042bcb4c737512173fe632d6805fb9ec61811a9ab33671929ad1f`.

The hold-aware protocol and evaluator are implemented in
`scripts/rsl_rl/evaluate_state_baseline.py` and
`scripts/rsl_rl/state_baseline_protocol.py`; the protocol tests pass with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_state_baseline_protocol.py`.
