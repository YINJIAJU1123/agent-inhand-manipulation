# Visual reorientation camera

The visual semantic task uses one fixed eye-to-hand RGB-D camera.  Its pose is
defined in
`source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/brainco_hand_visual_semantic_reorient_env_cfg.py`:

| quantity | value |
| --- | --- |
| position (world frame) | `(0.00, -0.72, 0.95) m` |
| orientation | `+32°` about world X, quaternion `(w,x,y,z)=(0.9613, 0.2756, 0, 0)` |
| image size | `128 × 128` RGB + metric depth |
| focal length / aperture | `28 mm` / `20.955 mm` |
| approximate horizontal field of view | `41°` |

The hand/object is around `(0, -0.11, 0.56) m` at reset.  Therefore this
camera observes the manipulation from the front and slightly above, with the
optical axis aimed downward by about 32°.  It is deliberately not a camera
directly above the hand: a strict overhead view hides the contact geometry
between the fingers and makes front-facing markers harder to read.  The
elevated 3/4 view exposes several object faces while retaining finger and
palm context.  It is also easy to reproduce in a real setup as a fixed
camera-on-tripod (eye-to-hand) configuration.

This choice is supported by prior in-hand work.  HORA/Shadow Hand used a
camera cage with cameras above and at angled views, while *Visual Dexterity*
showed that a single commodity depth camera can support object reorientation.
More recent monocular work such as *ViserDex* uses a wrist-mounted camera;
that is a useful future ablation, but it changes the viewpoint as the hand
moves and is not the default here.

The pose is kept fixed during data collection so that viewpoint is not a hidden
source of supervision.  Camera randomization should be introduced only as a
separate robustness experiment (small azimuth/elevation/roll perturbations),
and the exact pose should be recorded with every dataset version.

References:

- [Learning dexterous in-hand manipulation](https://doi.org/10.1177/0278364919887447)
- [Visual Dexterity: In-Hand Reorientation](https://arxiv.org/abs/2211.11744)
- [ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation](https://arxiv.org/abs/2604.11138)
