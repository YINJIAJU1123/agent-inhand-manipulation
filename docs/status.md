# Frozen status, 2026-10-22

Training and evaluation were stopped at the user's request before repository
cleanup. DLC `dlc175eknfjz0rkj` was confirmed **Stopped**. Its unfinished candidate
is not the published local policy.

## Latest real Astra development evaluation

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
