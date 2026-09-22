"""Revo3 engineering teacher configuration used for the Ours stage.

The observation and action contract stays identical to the VisERDex-style
teacher.  Ours changes the optimization target: a reached face must remain
inside the task tolerance for a short dwell before the next goal is sampled.
This makes the comparison an engineering improvement on the same task rather
than a change of robot, object, or interface.
"""

from isaaclab.utils import configclass

from .brainco_hand_viserdex_teacher_env_cfg import BrainCoHandVisERDexTeacherEnvCfg


@configclass
class BrainCoHandOursTeacherEnvCfg(BrainCoHandVisERDexTeacherEnvCfg):
    """Ours: hold-aware and stability-oriented Revo3 state teacher."""

    # Keep the VisERDex-style 244-D policy input and 21-D action output.
    observation_space = 244
    # Use the task-facing acceptance tolerance during this stage.  The
    # VisERDex-style teacher remains separately frozen at its 0.1-rad target.
    success_tolerance = 0.16
    goal_hold_time_s = 0.5
    hold_progress_reward_scale = 2.0
    freeze_goal_for_episode = False

    # Delay the strongest regularizers until the hold-aware policy has learned
    # to reach.  These are still the same Revo3 proxy terms as T1, but the
    # curriculum gives Ours a stable route to the useful behavior first.
    curriculum_start_successes = 3.0
    curriculum_full_successes = 35.0
