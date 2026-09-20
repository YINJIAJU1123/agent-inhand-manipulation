"""RGB-D random-layout semantic search v0; all thresholds are pilot settings."""
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from .brainco_hand_visual_semantic_reorient_env_cfg import BrainCoHandVisualSemanticReorientEnvCfg


@configclass
class BrainCoHandSearchEnvCfg(BrainCoHandVisualSemanticReorientEnvCfg):
    observation_space = 1111  # 16x16 RGB-D + 6x3 image stats + 63 proprio + 6 language
    state_space = 1127  # actor + privileged object pose/vel + target world normal
    asymmetric_obs = True
    is_finite_horizon = True
    episode_length_s = 20.
    freeze_goal_for_episode = True
    goal_hold_time_s = 1.
    max_consecutive_success = 0
    layout_split: str = "train"
    initial_visibility: str = "mixed"
    minimum_visible_fraction: float = .65
    minimum_projected_area: float = 20.
    maximum_facing_angle_deg: float = 65.
    maximum_linear_speed: float = .04
    maximum_angular_speed: float = .5
    depth_tolerance: float = .004
    # A fixed-layout colored background would confuse the simple detector.
    object_cfg = BrainCoHandVisualSemanticReorientEnvCfg().object_cfg.replace(
        spawn=BrainCoHandVisualSemanticReorientEnvCfg().object_cfg.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.3, .3, .3))
        )
    )
    tiled_camera = BrainCoHandVisualSemanticReorientEnvCfg().tiled_camera.replace(
        width=128, height=128, data_types=["rgb", "distance_to_image_plane"]
    )
    # Explicit optical-axis depth is required by the visibility evaluator.
    scene = BrainCoHandVisualSemanticReorientEnvCfg().scene.replace(num_envs=64)
