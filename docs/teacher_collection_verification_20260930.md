# Teacher collection verification (2026-09-30)

The frozen teacher checkpoint was replayed in the camera-enabled task and its actual sensor stream was saved locally.

- Checkpoint: `model_1999.pt`
- Checkpoint SHA256: `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`
- Task: `BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0`
- Target face: face 0, fixed for this qualitative clip
- Camera: 256x256 RGB, 30 FPS
- Frames: 360 (12 seconds)
- Recorded trials: 8
- Reached target: 6
- Dropped object: 2
- Terminal frames: 4, 63, 118, 158, 237, 277, 311, 351

Artifacts:

- `outputs/teacher_verification_face0/teacher_face0.webm`
- `outputs/teacher_verification_face0/metadata.json`
- `outputs/teacher_verification_face0/frame_*.png`

The clip is qualitative evidence. The larger formal collection remains the stronger quantitative check: 388 complete teacher quality records, 301 successful, with zero rollout alignment, terminal, stride, batch, or camera audit failures.
