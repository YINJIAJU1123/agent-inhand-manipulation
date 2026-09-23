"""Camera-enabled VisERDex teacher for matched visual-language rollouts."""

from __future__ import annotations

import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply, quat_mul

from .viserdex_teacher import VisERDexTeacherEnv
from .brainco.brainco_hand_viserdex_teacher_visual_env_cfg import (
    BrainCoHandVisERDexTeacherVisualEnvCfg,
)


class VisERDexTeacherVisualEnv(VisERDexTeacherEnv):
    """The frozen 244-D teacher with RGB-D capture and visible face patches."""

    cfg: BrainCoHandVisERDexTeacherVisualEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_markers.set_visibility(False)
        self.face_markers = VisualizationMarkers(self.cfg.face_marker_cfg)
        self._face_normals = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            device=self.device,
        )
        self._face_marker_rots = torch.tensor(
            [[0.70710678, 0.0, 0.70710678, 0.0],
             [0.70710678, 0.0, -0.70710678, 0.0],
             [0.70710678, -0.70710678, 0.0, 0.0],
             [0.70710678, 0.70710678, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=self.device,
        )
        self._camera_frame_index = 0

    def _setup_scene(self):
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        super()._setup_scene()
        self.scene.sensors["viserdex_semantic_camera"] = self._tiled_camera

    @property
    def camera(self) -> TiledCamera:
        return self._tiled_camera

    def _update_face_markers(self) -> None:
        rot = self.object_rot
        normals = self._face_normals.unsqueeze(0).expand(self.num_envs, -1, -1)
        offsets = quat_apply(
            rot.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4),
            (0.0355 * normals).reshape(-1, 3),
        ).reshape(self.num_envs, 6, 3)
        positions = self.object_pos.unsqueeze(1) + offsets + self.scene.env_origins.unsqueeze(1)
        marker_rotations = quat_mul(
            rot.unsqueeze(1).expand(-1, 6, -1),
            self._face_marker_rots.unsqueeze(0).expand(self.num_envs, -1, -1),
        ).reshape(-1, 4)
        self.face_markers.visualize(
            positions.reshape(-1, 3), marker_rotations,
            marker_indices=torch.arange(6, device=self.device).repeat(self.num_envs),
        )

    def _get_observations(self):
        self._update_face_markers()
        return super()._get_observations()

    def capture_camera(self, clone: bool = True, refresh: bool = True) -> dict[str, torch.Tensor]:
        if refresh:
            self.sim.render()
            self._tiled_camera.update(self.step_dt, force_recompute=True)
        output = {}
        for key in ("rgb", "depth"):
            value = self._tiled_camera.data.output.get(key)
            if value is not None:
                output[key] = value.clone() if clone else value
        output["frame_index"] = torch.full(
            (self.num_envs, 1), self._camera_frame_index, dtype=torch.long, device=self.device
        )
        self._camera_frame_index += 1
        return output
