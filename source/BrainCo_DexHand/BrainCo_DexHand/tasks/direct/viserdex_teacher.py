"""VisERDex-style privileged state teacher adapted to the 21-DoF Revo3 hand."""

from __future__ import annotations

import torch

from isaaclab.utils.math import sample_uniform, saturate

from .semantic_reorient import SemanticReorientEnv, required_goal_hold_steps, advance_goal_hold
from .reorient import scale


class VisERDexTeacherEnv(SemanticReorientEnv):
    """Full-protocol state teacher with randomized action dynamics."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ema_alpha = torch.full(
            (self.num_envs,), float(cfg.ema_alpha_min), dtype=torch.float, device=self.device
        )
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_queue = torch.zeros(
            (self.num_envs, int(cfg.action_delay_max_steps) + 1, cfg.action_space),
            dtype=torch.float,
            device=self.device,
        )
        self._action_history = torch.zeros(
            (self.num_envs, 4, cfg.action_space), dtype=torch.float, device=self.device
        )
        self._goal_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_timeout_steps = max(1, int(round(float(cfg.goal_timeout_s) / float(self.step_dt))))
        self._reset_target_pose(torch.arange(self.num_envs, device=self.device))

    def _reset_target_pose(self, env_ids):
        super()._reset_target_pose(env_ids)
        if hasattr(self, "_goal_elapsed_steps"):
            self._goal_elapsed_steps[env_ids] = 0
            self._ema_alpha[env_ids] = sample_uniform(
                float(self.cfg.ema_alpha_min), float(self.cfg.ema_alpha_max), (len(env_ids),), device=self.device
            )
            curriculum = ((self.consecutive_successes.mean() - self.cfg.curriculum_start_successes)
                           / max(self.cfg.curriculum_full_successes - self.cfg.curriculum_start_successes, 1e-6)).clamp(0.0, 1.0)
            effective_delay_max = int(round(
                float(self.cfg.action_delay_min_steps)
                + curriculum.item() * float(self.cfg.action_delay_max_steps - self.cfg.action_delay_min_steps)
            ))
            self._action_delay_steps[env_ids] = torch.randint(
                int(self.cfg.action_delay_min_steps),
                effective_delay_max + 1,
                (len(env_ids),),
                device=self.device,
            )
            self._action_queue[env_ids] = 0.0

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_action_queue"):
            self._action_queue[env_ids] = 0.0
            self._action_history[env_ids] = 0.0
            self._goal_elapsed_steps[env_ids] = 0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # The policy action is held for all physics substeps.  The queue adds a
        # sampled policy-step latency before the randomized EMA is applied.
        self.prev_actions = self.actions.clone()
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._action_history[:, -1] = actions
        self._action_queue = torch.roll(self._action_queue, shifts=-1, dims=1)
        self._action_queue[:, -1] = actions
        env_ids = torch.arange(self.num_envs, device=self.device)
        # The queue is ordered oldest -> newest.  A zero-step delay must pick
        # the newest action; the previous implementation selected the oldest
        # entry and silently inverted the sampled latency.
        queue_index = int(self.cfg.action_delay_max_steps) - self._action_delay_steps
        self.actions = self._action_queue[env_ids, queue_index].clone()

    def compute_full_observations(self):
        """Expose Revo3 state plus VisERDex action/property context."""

        base = super().compute_full_observations()
        if not hasattr(self, "_action_history"):
            history = base.new_zeros((self.num_envs, 4 * self.cfg.action_space))
            props = base.new_zeros((self.num_envs, 2))
        else:
            history = self._action_history.reshape(self.num_envs, -1)
            props = torch.stack(
                (
                    self._ema_alpha,
                    self._action_delay_steps.to(base.dtype) / max(float(self.cfg.action_delay_max_steps), 1.0),
                ),
                dim=-1,
            )
        # ``base`` already includes the six-dimensional semantic face token
        # from SemanticReorientEnv.
        return torch.cat((base, history, props), dim=-1)

    def _apply_action(self) -> None:
        raw_targets = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        alpha = self._ema_alpha.unsqueeze(-1)
        targets = alpha * raw_targets + (1.0 - alpha) * self.prev_targets[:, self.actuated_dof_indices]
        targets = saturate(
            targets,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = targets
        self.prev_targets[:, self.actuated_dof_indices] = targets
        self.hand.set_joint_position_target(targets, joint_ids=self.actuated_dof_indices)

    def _get_dones(self):
        self._compute_intermediate_values()
        self._step_orientation_error = self._step_orientation_error.new_zeros(self.num_envs)
        from .inhand_manipulation_env import rotation_distance

        self._step_orientation_error = rotation_distance(self.object_rot, self.goal_rot)
        self._step_object_distance = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1)
        self._step_target_face = self.target_face.clone()
        self._step_goal_rotation = self.goal_rot.clone()
        self._step_goal_reached = self._step_orientation_error <= self.cfg.success_tolerance
        self._step_dropped = self._step_object_distance >= self.cfg.fall_dist
        required_steps = required_goal_hold_steps(float(self.cfg.goal_hold_time_s), float(self.step_dt))
        self._goal_elapsed_steps += 1
        (
            self.goal_hold_steps,
            self.goal_success_latched,
            self._step_hold_complete,
            self._step_success_event,
        ) = advance_goal_hold(
            self.goal_hold_steps,
            self.goal_success_latched,
            self._step_goal_reached,
            self._step_dropped,
            required_steps,
        )
        timeout = self._goal_elapsed_steps >= self._goal_timeout_steps
        # Keep the ordinary environment horizon as a second safety bound.
        # This is also what makes short diagnostic/evaluation horizons
        # effective instead of relying only on the per-goal clock.
        timeout = timeout | (self.episode_length_buf >= self.max_episode_length - 1)
        if self.cfg.max_consecutive_success > 0:
            timeout = timeout | (
                self.successes + self._step_success_event.to(self.successes.dtype)
                >= self.cfg.max_consecutive_success
            )
        return self._step_dropped, timeout

    def _get_rewards(self):
        rot_dist = self._step_orientation_error
        goal_dist = self._step_object_distance
        dist_rew = goal_dist * self.cfg.dist_reward_scale
        rot_rew = (torch.abs(rot_dist) + self.cfg.rot_eps).reciprocal() * self.cfg.rot_reward_scale
        reward = dist_rew + rot_rew
        action_mag = torch.linalg.vector_norm(self.actions, dim=-1)
        action_rate = torch.linalg.vector_norm(self.actions - self.prev_actions, dim=-1)
        joint_vel = torch.linalg.vector_norm(self.hand_dof_vel[:, self.actuated_dof_indices], dim=-1)
        object_linvel = torch.linalg.vector_norm(self.object_linvel, dim=-1)
        object_angvel = torch.linalg.vector_norm(self.object_angvel, dim=-1)
        target_error = self.cur_targets[:, self.actuated_dof_indices] - self.hand_dof_pos[:, self.actuated_dof_indices]
        torque_proxy = self.cfg.torque_proxy_stiffness * target_error - self.cfg.torque_proxy_damping * self.hand_dof_vel[:, self.actuated_dof_indices]
        dof_normalizer = float(len(self.actuated_dof_indices)) ** 0.5
        # The published terms use an L2 torque norm and absolute mechanical
        # work.  Normalize the Revo3 PD proxy by sqrt(DoF) because it is a
        # position-control proxy rather than a measured actuator torque.
        torque_proxy_l2 = torch.linalg.vector_norm(torque_proxy, dim=-1)
        torque_mag = torque_proxy_l2 / dof_normalizer
        joint_work_abs = torch.sum(torch.abs(torque_proxy * self.hand_dof_vel[:, self.actuated_dof_indices]), dim=-1)
        joint_work = joint_work_abs / dof_normalizer
        curriculum = ((self.consecutive_successes.mean() - self.cfg.curriculum_start_successes)
                       / max(self.cfg.curriculum_full_successes - self.cfg.curriculum_start_successes, 1e-6)).clamp(0.0, 1.0)
        # Start with task completion and progressively turn on the full
        # stability regularization as the moving success count improves.
        reg_scale = curriculum
        reward = reward + reg_scale * self.cfg.action_penalty_scale * action_mag
        reward = reward + reg_scale * self.cfg.action_slew_penalty_scale * action_rate
        reward = reward + reg_scale * self.cfg.joint_velocity_penalty_scale * joint_vel
        reward = reward + reg_scale * self.cfg.object_linear_velocity_penalty_scale * object_linvel
        reward = reward + reg_scale * self.cfg.object_angular_velocity_penalty_scale * object_angvel
        reward = reward + reg_scale * self.cfg.joint_torque_penalty_scale * torque_mag
        reward = reward + reg_scale * self.cfg.joint_work_penalty_scale * joint_work

        self.successes = self.successes + self._step_success_event.to(self.successes.dtype)
        reward = torch.where(self._step_success_event, reward + self.cfg.reach_goal_bonus, reward)
        reward = torch.where(self._step_dropped, reward + self.cfg.fall_penalty, reward)

        # Keep the standard RSL-RL moving success statistic in sync with the
        # base task.  ``reset_buf`` still describes the transition just
        # evaluated here, before DirectRLEnv resets terminal slots.
        num_resets = torch.sum(self.reset_buf)
        finished_successes = torch.sum(self.successes * self.reset_buf.to(self.successes.dtype))
        self.consecutive_successes[:] = torch.where(
            num_resets > 0,
            self.cfg.av_factor * finished_successes / num_resets
            + (1.0 - self.cfg.av_factor) * self.consecutive_successes,
            self.consecutive_successes,
        )
        # Snapshot before goal reset clears hold/success event buffers.
        if self.cfg.record_eval_metrics:
            self.extras["semantic_metrics"] = {
                "orientation_error": self._step_orientation_error.clone(),
                "object_distance": self._step_object_distance.clone(),
                "goal_reached": self._step_goal_reached.clone(),
                "dropped": self._step_dropped.clone(),
                "hold_complete": self._step_hold_complete.clone(),
                "hold_steps": self.goal_hold_steps.clone(),
                "success_event": self._step_success_event.clone(),
                "target_face": self._step_target_face.clone(),
                "goal_rotation": self._step_goal_rotation.clone(),
            }

        self.reset_goal_buf[:] = self._step_success_event
        if self.cfg.freeze_goal_for_episode:
            self.reset_goal_buf.zero_()
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["active_episode_successes"] = self.successes.mean()
        self.extras["log"]["drop_fraction"] = self._step_dropped.float().mean()
        self.extras["log"]["orientation_error_rad"] = self._step_orientation_error.mean()
        self.extras["log"]["object_angular_velocity_rad_s"] = torch.linalg.vector_norm(self.object_angvel, dim=-1).mean()
        self.extras["log"]["action_delay_steps"] = self._action_delay_steps.float().mean()
        self.extras["log"]["ema_alpha"] = self._ema_alpha.mean()
        self.extras["log"]["curriculum"] = curriculum
        # Keep raw reward components visible during smoke/debug runs.  These
        # are averages before any PPO normalization and expose unit mistakes
        # that can otherwise be hidden by the aggregate reward.
        self.extras["log"]["reward_task"] = (dist_rew + rot_rew).mean()
        self.extras["log"]["reward_action"] = (reg_scale * self.cfg.action_penalty_scale * action_mag).mean()
        self.extras["log"]["reward_joint_velocity"] = (reg_scale * self.cfg.joint_velocity_penalty_scale * joint_vel).mean()
        self.extras["log"]["reward_torque"] = (reg_scale * self.cfg.joint_torque_penalty_scale * torque_mag).mean()
        self.extras["log"]["reward_work"] = (reg_scale * self.cfg.joint_work_penalty_scale * joint_work).mean()
        self.extras["log"]["torque_proxy_l2"] = torque_proxy_l2.mean()
        self.extras["log"]["torque_proxy_l2_normalized"] = torque_mag.mean()
        self.extras["log"]["joint_work_abs_sum"] = joint_work_abs.mean()
        self.extras["log"]["joint_work_normalized"] = joint_work.mean()
        return reward
