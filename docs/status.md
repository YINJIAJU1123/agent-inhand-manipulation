# Paper and public-runtime status, 2026-10-26

The fixed paper action weights and an independently checked proposal component
are now included. The original full simulation wrappers remain the reference
runtime snapshot documented below. They are not the evaluator used for the
later paper studies. A separate `run_paper_eval.py` now preserves the paper
control loop and has passed twelve scripted simulator integration cases.
The exact paper action fitting recipe and a verified local data export are now
available in [paper action training](paper_action_training.md). Dataset distribution
and external model-service setup remain separate.

| Complete cohort | Astra reference | Astra adapted | Primary matched contrast |
| --- | ---: | ---: | --- |
| Development, 144 raw episodes | 21/36 | 28/36 | 20/34 vs 28/34, +23.5pp |
| Prospective resets, 144 raw episodes | 27/36 | 28/36 | 27/35 vs 27/35, zero difference |

Both cohorts include four arms, with 36 raw episodes each. Only the registered
service-error/latency rule removes keys symmetrically for matched panels. All
raw outcomes remain. A separate live component comparison supports mixed-data
fitting over old-only fitting, +20.0 pp [5.7, 36.1]. It uses separately fitted weights.
Representation and anchoring contrasts remain uncertain. Do not pool these
cohorts or equate their system-level backend gap with superior reasoning.

Machine-readable source identities and complete contrasts are in
[paper_results.json](paper_results.json), derived from paper commit `cd7c7b6`.
The [paper repository](https://github.com/YINJIAJU1123/CVPR-AgenticRotation) contains
full provenance, shared-stop studies and remaining limitations. The paper also reports a completed
rendering-shift diagnostic, which does not supply a new live success rate.

The following dated results describe the retained original wrappers:

---

# Reference runtime status, 2026-10-22

Training and evaluation were stopped at the user's request before repository
cleanup. DLC `dlc175eknfjz0rkj` was confirmed **Stopped**. Its unfinished candidate
is not the published local policy.

## Historical real Astra development evaluation

| Configuration | Complete tasks | Episodes |
| --- | ---: | ---: |
| Semantic memory, original startup latch | 4 | 12 |
| Semantic memory, fixed-green startup | 4 | 12 |

Complete success requires correct semantic binding, full quaternion error at
most 0.16 radians, no previous drop and a correct autonomous stop. Simulation
continues through real inference. These experiments reuse development states
and observe independent service latency. They are not a final held-out estimate.
The 80% target has not been achieved.

Startup drops decreased from three to zero in the fixed-green continuation,
but overall drops and complete successes did not improve. Remaining failures
include post-reply drops and reaching a correct pose without correctly stopping.

The latest local two-second recovery fine-tuning completed 17/24 known-color
cases versus 19/24 for the original controller. It was not deployed. Those
local results do not measure language grounding or autonomous stopping.

Historical scores around 50–56.7% and the language-only 600/768 result used
other task or termination protocols. They must not be merged with this table.
Detailed historical experiments remain in Git history; this working snapshot
contains the current runtime and its weights.
