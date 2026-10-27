# Learned shared stopping in the paper

This entry releases both stop corrections used in the completed three-arm,
108-episode Astra study. It keeps the fixed mixed actor and original waiting
controller. The ongoing paired-view training experiment is separate and is not
loaded here.

## What changes

The original verifier produces a six-goal feature bank every ten control steps.
Each correction sees three causal 125D frames per goal and adds a residual to the
original logit. Both models have 39,425 parameters. The bounded model uses
`4*tanh(delta/4)` and the unbounded model uses `delta`. The common threshold is 0.7.
Both heads are computed in all three arms. Only the regular shared-stop proposal
uses the selected head. Native and pending-native gates, agent feedback, action
commands and the independent success scorer retain the original paths.

`original` is the matched reference arm of this stop comparison. It still computes
both corrections, so it differs in inference work from the earlier action-only
comparison. Use `scripts/run_paper_eval.py` for that earlier comparison.

## Fixed weights

| Arm | Selected epoch | Checkpoint |
| --- | --- | --- |
| Bounded residual | 63 | `checkpoints/runtime/stop_bounded4.pt` |
| Unbounded residual | 28 | `checkpoints/runtime/stop_unbounded.pt` |

The files are byte-identical to the calibration-selected research checkpoints.
`configs/paper_stop_assets.json` records hashes and training-audit provenance.
Both are retained. Neither is selected as a winner from live or replay results.

## Checks and execution

CPU verification:

```bash
python scripts/check_paper_stop.py
```

With the original research checkout available, add `--research-root /path/to/research`
to compare the full control-loop AST, inference classes, and probabilities and
proposals on an existing training shard. This reads previously collected training
features and does not launch training, a simulator or a model service.

Scripted simulator integration, with Isaac Lab installed:

```bash
python scripts/run_paper_stop_eval.py --stop-arm original --seed 154201 \
  --offline-smoke --offline-retention-probe --max-steps 900 \
  --output outputs/stop_original_smoke
```

Repeat for `bounded4` and `unbounded`, using separate output directories. The
scripted probe uses fixed public colors and six-second simulated requests. Its
symbols are not derived from the hidden layout. It exercises tensor wiring and
stop routing, not semantic grounding or task performance.

For a genuine-agent six-scene job, configure the bridge as described in
`docs/supervisors.md`, omit both offline flags and keep the fixed full horizon:

```bash
python scripts/run_paper_stop_eval.py --stop-arm bounded4 --seed 154201 \
  --record-video --output outputs/stop_bounded4_seed154201_r1
```

The historical comparison uses seeds 154201/154202, three repeats, all three arms
and counterbalanced order. Preserve all raw cases and the paper's symmetric
service-error/over-120-second exclusion panel. A single job is not the complete
comparison. A new invocation uses the currently configured external service and
does not guarantee reproduction of historical responses or latency.

## Release status

CPU checks passed on 2,160 recorded training rows from six cases. All 38,880
probability values and 38,880 proposal decisions agree exactly with the research
implementation on the checked host. Four inference-class ASTs and the complete
control loop match. Invalid cadence, nonfinite probabilities, stale proposals,
unsynchronized evidence and unknown arms are rejected.

The standalone simulator check for this new entry is pending. The existing
12-case simulator verification of the action-only evaluator does not establish
integration of this additional stop path. No new live evaluation has been run for
this release. Verification is recorded in `docs/paper_stop_verification.json`.
