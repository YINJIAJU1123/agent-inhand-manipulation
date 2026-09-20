# VisERDex-to-Revo3 Cube adaptation

This document is the working contract for the single-object Revo3 adaptation.
The first milestone is a faithful VisERDex-style visual reorientation pipeline
for the Cube. Language and the visual-target method are added only after the
visual control path has a reproducible simulation and deployment interface.

## Scope

- One object: Cube.
- One policy and one pose estimator checkpoint.
- Recurrent student receives proprioception, estimated object pose/keypoints,
  and observation history.
- Camera geometry is fixed between simulation and the eventual Revo3 mount.
- The existing Direct policies remain infrastructure and control baselines.

## Baseline ladder

1. Direct privileged/state policy (existing codebase and regression oracle).
2. VisERDex-style visual student without language.
3. The language-conditioned visual-target policy, with the same action budget,
   camera model, reset bank, and success protocol as (2).

The original VisERDex orientation task is a separate reproduction check. The
paper comparison uses the same Cube target-surface task for all three rows;
the visual student receives an oracle target representation in row (2), while
row (3) obtains that representation from the language command.

## Deployment contract

```text
camera frame
  -> Cube mask/keypoints/pose + confidence
  -> recurrent student (pose + proprio + action history)
  -> 21-D Revo3 delta action
  -> existing joint-order/limit/SDK layer
```

The camera front-end and policy must be independently exportable. The current
proprio-only ONNX runner is retained for Direct/HORA regression; a visual
runner must declare its exteroceptive and language inputs explicitly instead of
silently reusing the old `[B, 126]`/`[B, 30, 42]` contract.

## Success protocol

Report first exposure, readable exposure, stable hold, drop, time-to-exposure,
and terminal success separately. Terminal success requires the requested Cube
surface to satisfy the visibility/readability thresholds continuously for the
hold window while the object remains grasped.

## Checklist

- [x] Isaac Sim 5.1 + Isaac Lab 2.3.2 environment on the 5090 host.
- [x] VisERDex Cube assets and one-object pose-estimator data.
- [x] Revo3 action/observation adapter and camera extrinsics.
- [x] Privileged Cube teacher smoke and checkpoint.
- [x] RGB-D camera smoke on the Revo3 hand scene (256x256; valid depth 64.5%).
- [x] Versioned visual-language IO builder and offline dry-run.
- [ ] Recurrent visual student training and checkpoint.
- [ ] Cube pose-estimator train/eval smoke.
- [ ] ONNX export of the visual student with the versioned IO contract.
- [ ] Revo3 dry-run with camera/pose input and no motor command.
- [ ] Real-hardware deployment only after the dry-run and low-speed checks pass.
