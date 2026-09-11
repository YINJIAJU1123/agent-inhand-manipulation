"""Camera-enabled semantic Revo3 environment for visual data collection.

The actor observation intentionally remains the semantic state-teacher
observation by default.  ``capture_camera`` gives the RGB/depth tensors to a
custom visual-language learner without changing the 21-D action interface or
breaking existing PPO checkpoints.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.sensors import TiledCamera

from .semantic_reorient import SemanticReorientEnv
from .brainco.brainco_hand_visual_semantic_reorient_env_cfg import (
    BrainCoHandVisualSemanticReorientEnvCfg,
)


class VisualSemanticReorientEnv(SemanticReorientEnv):
    """Semantic Revo3 teacher with a fixed RGB-D camera sensor."""

    cfg: BrainCoHandVisualSemanticReorientEnvCfg

    def __init__(self, cfg: BrainCoHandVisualSemanticReorientEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._camera_frame_index = 0

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
        privileged-state checkpoints a 128x128 image tensor.
        """
        return super()._get_observations()
