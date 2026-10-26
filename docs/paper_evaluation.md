# Paper evaluation entry

The dedicated `scripts/run_paper_eval.py` entry uses the paper control loop,
shared-stop eligibility, contextual latent interface, collision-clearance reset,
fixed-green startup and the original visual verifier. It is separate from the
historical reference wrappers. Its local simulator integration passed twelve scripted cases, six in each actor
mode. The complete record is `docs/paper_runtime_verification.json`. Source and metric identities are in `configs/paper_runtime_sources.json`.

## Local integration

```bash
python scripts/check_paper_runtime.py
python scripts/run_paper_eval.py --supervisor astra --actor-mode candidate \
  --seed 147201 --offline-smoke --max-steps 120 --output outputs/paper_smoke
```

The smoke uses scripted public replies and no model service. It checks integration,
not language understanding or task performance. Formal jobs fix the horizon to
3,600 steps and the request budget to six. Short jobs require `--offline-smoke`.

## Genuine-agent jobs

Configure the Astra bridge or local latent service using `docs/supervisors.md`.
The evaluator does not create credentials or start a service automatically.
Then a single job runs six scenes, for example:

```bash
python scripts/run_paper_eval.py --supervisor astra --actor-mode candidate \
  --seed 149201 --record-video --output outputs/astra_adapted_seed149201_r1
```

Reference control uses `--actor-mode original`. The contextual VLM uses
`--supervisor latent --service-url http://127.0.0.1:18772`, with the same two
actor modes. The local VLM backend and its model weights must be supplied
separately. Independent invocations give the three repeats. Preserve every
raw result and use the paper's registered symmetric service-exclusion panels.
A single six-scene invocation does not constitute the 144-episode comparison.

Both old-only and mixed fitted actors are computed in every arm, as in the
original experiment. The additional `paper_comparison_actor.pt` is required for
this computation match. It is the earlier old-data continuation checkpoint and
is distinct from the separately fitted later component-study baseline.

Only the selected action proposal changes between original and candidate arms.
Feedback and native/shared stopping retain the original paths. Public inspection
skills may enable adaptation before target identification. The fixed success
rule is stored byte-identically in `configs/evaluation/autonomous_completion_v1.json`.

Full protocol reproduction also requires the registered seed/repeat schedule,
model-service versions, external Isaac Lab environment and source dataset for
training. This entry does not claim to reproduce a remote model's hidden version
or make historical outcomes deterministic. CPU/source parity and local integration
checks are distinct from a new live performance evaluation.

## Verified scope

The clean evaluator control-loop AST matches the research loop. Source, metric and
eight weight hashes pass CPU checks. Two scripted simulator jobs share exact
initial physical states. Reference action-change counters are zero in all six
cases. Adapted counters are118/48/58/118/58/118, and independent saved-cadence
checks confirm activation and exact reference proposals when disabled. These
120-step cases are excluded from performance tables. Real service delivery and
full-horizon outcomes require a separately authorized live run.
