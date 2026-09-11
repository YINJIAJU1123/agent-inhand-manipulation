# Copyright (c) 2022-2026, The Isaac Lab Project Developers.

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from .brainco_hand_env_cfg import BrainCoHandEnvCfg as _CubeCfg


@configclass
class BrainCoHandSemanticReorientEnvCfg(_CubeCfg):
    """State-teacher configuration for semantic surface reorientation."""

    # The original implementation concatenates 152 state values.  We append
    # a six-dimensional target-face token for this teacher.
    observation_space = 158
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )

    # Encourage stable control during the target-facing maneuver.
    action_penalty_scale = -0.0004
    reach_goal_bonus = 300.0
    success_tolerance = 0.16
    fall_penalty = -25.0
