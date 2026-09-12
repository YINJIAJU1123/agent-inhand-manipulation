# Visual stage: validated foundation and remaining work

The standalone Docker environment can render the actual Revo3 hand and a cube
through a fixed RGB-D camera. This is the starting point for the visual stage,
not a completed visual-language dataset or policy.

Before a training-data collection run:

1. Re-evaluate the state teacher after the 2026-09-12 face-reset corrections.
   Earlier reports used accumulated face tokens and reversed side-face labels.
   Compare teacher and visual-task object geometry, mass and contact parameters;
   accepting the same observation/action dimensions does not establish matching
   dynamics. Record the exact checkpoint, source revision and object configuration.
2. Add actual visible symbols to the object. The existing JSON assigns names
   such as `triangle`, but the rendered primitive cube has no symbols. Verify
   the mapping between rendered surfaces, local normals and instruction labels.
3. Specify the visual task reward. Current teacher goals point a face at world
   +Z and include a sampled yaw. Showing a symbol to the camera additionally
   requires camera-relative visibility and finger-occlusion checks. Random yaw
   should not create an unobservable requirement for a language-only goal.
4. Define a deployable student observation. The collector currently stores
   full teacher observations under `policy_observations`; they include true
   object pose, goal quaternion and orientation error. They are privileged
   supervision, not ordinary proprioception. The visual actor should receive
   only its images, instruction, accessible joint state and action history.
5. Validate frame/action/goal alignment through environment and goal resets.
   Record episode IDs, reset flags, terminal outcomes, camera calibration,
   object/marker mapping and checkpoint provenance. Stream bounded shards rather
   than keeping an entire large RGB-D collection in RAM.
6. Validate the image on a 5090 host before scaling collection. The local
   RTX 5060 test proves sensor operation on that machine; it does not resolve
   the separate driver/library mismatch observed in the old remote LXD setup.

Only after these checks should teacher rollouts be treated as training examples
for a language-conditioned visual student. The first student baseline should
be evaluated in closed loop before adding a learned consequence predictor or
agentic decision mechanism.
