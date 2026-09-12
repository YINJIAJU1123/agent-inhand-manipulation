"""Camera-enabled semantic Revo3 environment for visual data collection.

The actor observation intentionally remains the semantic state-teacher
observation by default.  ``capture_camera`` gives the RGB/depth tensors to a
custom visual-language learner without changing the 21-D action interface or
breaking existing PPO checkpoints.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply, quat_mul

from .semantic_reorient import SemanticReorientEnv
from .brainco.brainco_hand_visual_semantic_reorient_env_cfg import (
    BrainCoHandVisualSemanticReorientEnvCfg,
)


class VisualSemanticReorientEnv(SemanticReorientEnv):
    """Semantic Revo3 teacher with a fixed RGB-D camera sensor."""

    cfg: BrainCoHandVisualSemanticReorientEnvCfg

    def __init__(self, cfg: BrainCoHandVisualSemanticReorientEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Goal orientation and debug geometry are privileged teacher state.
        # Do not let their rendered appearance leak into visual observations.
        self.goal_markers.set_visibility(False)
        self.face_markers = VisualizationMarkers(self.cfg.face_marker_cfg)
        self._face_normals = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            device=self.device,
        )
        # Local rotations mapping the patch's +Z normal to each cube face.
        self._face_marker_rots = torch.tensor(
            [[0.70710678, 0.0, 0.70710678, 0.0],
             [0.70710678, 0.0, -0.70710678, 0.0],
             [0.70710678, -0.70710678, 0.0, 0.0],
             [0.70710678, 0.70710678, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            device=self.device,
        )
        self._update_face_markers()
        self._camera_frame_index = 0

    def _update_face_markers(self) -> None:
        """Place six non-colliding colored patches on the moving cube."""
        # Object pose is environment-local; marker visualization uses world
        # coordinates, hence adding the per-environment origins below.
        rot = self.object_rot
        normals = self._face_normals.unsqueeze(0).expand(self.num_envs, -1, -1)
        offsets = quat_apply(rot.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4),
                             (0.0355 * normals).reshape(-1, 3)).reshape(self.num_envs, 6, 3)
        positions = self.object_pos.unsqueeze(1) + offsets + self.scene.env_origins.unsqueeze(1)
        marker_positions = positions.reshape(-1, 3)
        marker_rotations = quat_mul(
            rot.unsqueeze(1).expand(-1, 6, -1),
            self._face_marker_rots.unsqueeze(0).expand(self.num_envs, -1, -1),
        ).reshape(-1, 4)
        marker_indices = torch.arange(6, device=self.device).repeat(self.num_envs)
        self.face_markers.visualize(marker_positions, marker_rotations, marker_indices=marker_indices)

    def _setup_scene(self):
        # TiledCamera must be instantiated before clone_environments() in the
        # parent setup, otherwise only the source environment receives a
        # sensor.  The parent then creates hand/object/lights and performs the
        # clone, after which we register the sensor in the scene collection.
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        super()._setup_scene()
        self.scene.sensors["semantic_camera"] = self._tiled_camera

    @property
    def camera(self) -> TiledCamera:
        """Return the tiled camera instance for custom collectors."""
        return self._tiled_camera

    def capture_camera(self, clone: bool = True) -> dict[str, torch.Tensor]:
        """Return current RGB-D frames and camera metadata.

        RGB is uint8 in Isaac Lab's native camera format.  Depth is in metres
        with ``+inf`` for pixels without a hit.  The returned tensors have
        shape ``(num_envs, H, W, C)``.  Cloning avoids exposing the sensor's
        internal buffers to an asynchronous learner.
        """
        output: dict[str, torch.Tensor] = {}
        for key in ("rgb", "depth"):
            value = self._tiled_camera.data.output.get(key)
            if value is None:
                continue
            output[key] = value.clone() if clone else value
        output["frame_index"] = torch.full(
            (self.num_envs, 1), self._camera_frame_index, dtype=torch.long, device=self.device
        )
        self._camera_frame_index += 1
        return output

    def _get_observations(self) -> dict[str, Any]:
        """Keep teacher observations unchanged; camera is read explicitly.

        A future multimodal runner can override this method and concatenate an
        encoded camera feature with the state observation.  Keeping raw images
        out of the default policy observation avoids accidentally feeding
        privileged-state checkpoints an image tensor.
        """
        self._update_face_markers()
        return super()._get_observations()
