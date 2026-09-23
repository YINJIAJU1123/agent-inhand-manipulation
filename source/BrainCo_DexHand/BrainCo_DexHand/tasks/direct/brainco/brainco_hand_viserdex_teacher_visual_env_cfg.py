"""Camera-enabled configuration that preserves the 244-D VisERDex teacher input."""

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .brainco_hand_viserdex_teacher_env_cfg import BrainCoHandVisERDexTeacherEnvCfg


@configclass
class BrainCoHandVisERDexTeacherVisualEnvCfg(BrainCoHandVisERDexTeacherEnvCfg):
    """VisERDex-style teacher dynamics plus an explicit RGB-D camera.

    The camera is read by the rollout collector and is deliberately excluded
    from the policy observation, so model_750.pt remains shape-compatible.
    """

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/VisERDexSemanticCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.72, 0.95),
            rot=(0.877115072743617, 0.48028028188336, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.0,
            focus_distance=0.72,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        width=256,
        height=256,
    )
    face_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/viserdex_semantic_face_markers",
        markers={
            "red": sim_utils.CuboidCfg(
                size=(0.028, 0.028, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.05, 0.05)),
            ),
            "green": sim_utils.CuboidCfg(
                size=(0.028, 0.028, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.75, 0.15)),
            ),
            "blue": sim_utils.CuboidCfg(
                size=(0.028, 0.028, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 0.9)),
            ),
            "yellow": sim_utils.CuboidCfg(
                size=(0.028, 0.028, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.05)),
            ),
            "magenta": sim_utils.CuboidCfg(
                size=(0.028, 0.028, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.05, 0.75)),
            ),
            "cyan": sim_utils.CuboidCfg(
                size=(0.028, 0.028, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.8, 0.85)),
            ),
        },
    )
    include_camera_in_policy: bool = False
    camera_frame_stack: int = 1
    # The language contract names a face, not an in-plane rotation.  Remove
    # the benchmark-only random yaw while collecting language demonstrations;
    # the frozen benchmark teacher itself keeps random yaw in its own config.
    goal_yaw = 0.0
