# Visual student protocol v1

This document freezes the first end-to-end visual-student run after the
privileged teacher stage. The frozen manifest is
`configs/visual_student_freeze.json`.

## Frozen control contract

- Teacher: `model_1999.pt` from the hold-aware Revo3 candidate run.
- Teacher SHA-256:
  `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`.
- Student action head: 21-dimensional `tanh` output.
- Evaluator action scale: `6.0`.
- The environment remains responsible for saturating joint targets after the
  scaled command is applied. This is the empirical visual-student contract
  recorded in `docs/visual_student_status.md`; it must not be silently changed
  during an ablation.

## Information boundary

The student may use RGB-D, the natural-language instruction, hand joint state,
fingertip kinematics, previous normalized action, and observation history.
Object pose, target-face IDs, goal quaternion, orientation error, hidden layout,
and reward are forbidden. The collector stores teacher observations only for
provenance and must never pass them to the student.

## Run order

1. Collect fresh shards with the frozen teacher and record checkpoint SHA-256.
2. Merge shards with disjoint environment IDs.
3. Cache one of the versioned visual frontends.
4. Train the recurrent offline baseline with an episode-level train/validation
   split and one of training seeds `0,1,2`.
5. Run live camera evaluation on the frozen visual reorientation task. A
   target-conditioned mean feature is not a closed-loop visual evaluation and
   is rejected.
6. Repeat the same budget with the evidence-memory variant.

## Acceptance criteria

The first baseline is considered wired through only when all of the following
are present in its report:

- exact freeze manifest and code revision;
- dataset sample/episode counts and disjoint split manifest;
- action scale and checkpoint SHA-256;
- terminal success, ever-reach, minimum orientation error, drop, timeout and
  time-to-first-reach;
- per-face and per-reset-condition counts;
- no privileged fields in the student batch contract.

An offline MSE improvement is useful for debugging but is not a visual-control
success claim. This v1 stage validates visual reorientation. The active hidden
surface-search benchmark is a separate protocol and starts only after this
student control path is stable.
