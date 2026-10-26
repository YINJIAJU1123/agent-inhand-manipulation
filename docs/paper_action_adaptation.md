# Paper action component

## Placement

Astra delivers a validated public skill. The student takes eight frames of RGB-D
features and proprioception plus the delivered 768-dimensional goal. The adapted student
produces 21 normalized joint targets. After public-skill delivery, an active
episode uses that proposal in the same waiting/recovery wrapper as the reference.
An inspection skill can enable adaptation before the requested target is bound.

The public component is `scripts/agentic/adapted_actor.py`. Its `action` method
returns an action tensor and leaves the reference output dictionary intact:

```python
adapted = AdaptedActionProposal(reference, reference_pack, checkpoint_path)
original_output = reference(batch)
proposal = adapted.action(original_output, batch, public_skill_delivered, active)
# Supply proposal to the fixed waiting/recovery controller.
# Use original_output for the existing feedback and stopping path.
```

The caller's delivered-skill flag must come from public agent state. It must not
come from the evaluator's hidden target. Before delivery or after termination,
the returned proposal equals the original action exactly. `enabled=False` selects
the reference proposal while still computing the candidate. Candidate evidence
and stop outputs are discarded. Freezing those head parameters alone would not
preserve their predictions when the representation changes.

## What was learned

The adapted actor fine-tunes the existing student representation and action head.
It is not a small new LoRA module attached to Astra. 943,685 of 1,009,993 student
parameters are trainable. The separate small-VLM baseline uses LoRA.

Teacher actions label student-visited states. New training data contain 16,485
samples from six reset seeds, with 6,339 validation samples from two others. Earlier
data contain 11,634 training and 1,528 validation samples. Frames are correlated.
A mixed batch uses 32 old and 32 new examples. Smooth-L1 teacher loss uses beta 0.1 with
a 0.05 mean-squared action anchor to the frozen reference. AdamW rates are 1e-4 for
the action head and 1e-5 for representation, with 1e-4 weight decay and gradient
norm clipping at 1. Both mixed and old-only controls receive 182 updates per epoch
for 50 epochs from the same prior fitted checkpoint. There is no failure weighting.

Validation weights seeds equally within each data domain and domains equally.
Epoch 47 minimizes held-out teacher loss. The fixed resulting checkpoint is
`b02d80830dcd4aebeec14c7d30e8645a47dc168a6ce059165d4744f683359862`.
Development and prospective outcomes are both retained in [status](status.md).

## CPU checks

```bash
python scripts/check_runtime.py
python scripts/check_adapted_actor.py
```

These verify seven asset hashes, load models, exercise all public-skill/active
mask combinations in the selected test set, and reject malformed masks. They
make no service calls and do not report task performance. The optional
`--observations /path/to/data.pt` accepts a local shard containing `rgb_history`,
`proprio_history` and `goal_id`, using only its first six observations.

`docs/paper_action_verification.json` records exact CPU parity with the frozen
research implementation on 24 recorded observation histories from two validation
seeds and 32 mask/batch checks. The maximum action difference is zero. These are
observation histories, not 24 independent episodes. The reproduction script is
`scripts/paper/check_source_parity.py --research-root /path/to/research-checkout`.
It requires the recorded source data, which are not included in this release.

## Reproduction boundary

This release supplies the fixed weights and proposal selection component. It
does not automatically install the component into the older `run_astra.py` or
`run_vlm.py` wrappers. Those wrappers retain their dated reference behavior.
The paper's evaluator has additional initialization, protocol and trace checks,
and computes both fitted comparison actors in every arm. Reusing this smaller
component does not reproduce that evaluator's computation or timing by itself.

Use the full paper protocol and original feedback/stop path for an end-to-end
comparison. Full evaluator and training-data packaging remain outstanding. No
online success rate is certified by the component checks. Credentials, training
samples, private service traces and unrelated checkpoints are not published here.
