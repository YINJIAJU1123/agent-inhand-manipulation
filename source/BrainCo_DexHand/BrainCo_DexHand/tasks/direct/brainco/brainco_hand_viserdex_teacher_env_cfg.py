"""VisERDex-style Revo3 state-teacher configuration.

The task keeps the Revo3 action/observation interface while making the
VisERDex teacher protocol explicit: 0.1-rad training success, per-goal
timeout, repeated target sampling, 50-goal episode cap, randomized EMA and
action latency, and stability regularization.
"""

from typing import Optional

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .brainco_hand_semantic_reorient_env_cfg import BrainCoHandSemanticReorientEnvCfg


@configclass
class BrainCoHandVisERDexTeacherEnvCfg(BrainCoHandSemanticReorientEnvCfg):
    """Revo3 adaptation of the published VisERDex privileged teacher."""

    # Base Revo3 state observation (152) + four-step action history (84),
    # semantic face token (6), and randomized action properties (2).
    observation_space = 244
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    # The published teacher allows up to 50 goals, each with its own 10 s
    # no-success window.  The environment horizon must therefore exceed one
    # goal window; per-goal termination is implemented in the task.
    episode_length_s = 500.0
    goal_timeout_s = 10.0
    success_tolerance = 0.1
    max_consecutive_success = 50
    goal_hold_time_s = 0.0
    hold_progress_reward_scale = 0.0
    freeze_goal_for_episode = False

    # Paper-style EMA alpha is randomized per environment.  The adaptation
    # retains the same target-position action interface for 21 DoF.
    ema_alpha_min = 0.08
    ema_alpha_max = 0.20
    action_delay_min_steps = 0
    action_delay_max_steps = 2

    # Revo3 engineering regularizers.  The task computes the torque/work
    # terms from a documented PD proxy because this position-control asset does
    # not expose the applied PhysX torque in the direct environment.
    dist_reward_scale = -20.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    reach_goal_bonus = 250.0
    fall_penalty = -10.0
    # The PD proxy is normalized by sqrt(DoF); these engineering terms are
    # kept at the same order as the dense task reward so the value target does
    # not become dominated by stabilization before the pose is learned.
    action_penalty_scale = -0.08
    action_slew_penalty_scale = -0.012
    joint_velocity_penalty_scale = -0.008
    object_linear_velocity_penalty_scale = -1.0e-3
    object_angular_velocity_penalty_scale = -1.0e-3
    joint_torque_penalty_scale = -2.0
    joint_work_penalty_scale = -0.02
    torque_proxy_stiffness = 3.0
    torque_proxy_damping = 0.1

    # Performance curriculum: regularization and latency increase as the
    # moving consecutive-success statistic improves.
    curriculum_start_successes = 2.0
    curriculum_full_successes = 25.0

    # None samples a random face and random in-plane yaw.  A fixed face is
    # reserved for balanced evaluation, never for the training distribution.
    goal_yaw: Optional[float] = None
    fixed_target_face: Optional[int] = None
