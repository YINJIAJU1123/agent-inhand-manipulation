"""Goal-conditioned semantic-surface teacher for Revo3.

This task keeps the original RevoLab 21-DoF action interface, but samples a
target cube face (plus a random in-plane yaw) instead of an unconstrained
orientation.  The face id is exposed as a six-dimensional goal token for the
state-based teacher.  A future visual-language student can replace this token
with an image/language embedding without changing the action interface.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from .inhand_manipulation_env import InHandManipulationEnv, rotation_distance
from BrainCo_DexHand.algo.agentic.language_goal import (
    FACE_NAMES,
    batch_instructions,
    encode_face_goal,
)


def required_goal_hold_steps(goal_hold_time_s: float, step_dt: float) -> int:
    """Convert a continuous hold duration to environment steps."""

    if goal_hold_time_s < 0.0:
        raise ValueError("goal_hold_time_s must be non-negative")
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    if goal_hold_time_s == 0.0:
        return 0
    return max(1, int(math.ceil(goal_hold_time_s / step_dt)))


def advance_goal_hold(
    hold_steps: torch.Tensor,
    success_latched: torch.Tensor,
    goal_reached: torch.Tensor,
    dropped: torch.Tensor,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance continuous-goal-hold state for a batch of environments.

    Returns next hold count, success latch, completed-hold predicate, and a
    one-step success event.  The event fires only on entry to the completed
    state, preventing duplicate bonuses while a frozen goal remains visible.
    """

    if required_steps < 0:
        raise ValueError("required_steps must be non-negative")
    if hold_steps.shape != success_latched.shape:
        raise ValueError("hold_steps and success_latched must have the same shape")
    if goal_reached.shape != hold_steps.shape or dropped.shape != hold_steps.shape:
        raise ValueError("goal predicates must match hold state shape")

    qualified = goal_reached & ~dropped
    next_steps = torch.where(qualified, hold_steps + 1, torch.zeros_like(hold_steps))
    hold_complete = qualified if required_steps == 0 else qualified & (next_steps >= required_steps)
    success_event = hold_complete & ~success_latched
    next_latched = torch.where(qualified, hold_complete, torch.zeros_like(success_latched))
    return next_steps, next_latched, hold_complete, success_event


class SemanticReorientEnv(InHandManipulationEnv):
    """Revo3 cube reorientation conditioned on a target surface id."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.target_face = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_face_onehot = torch.zeros((self.num_envs, len(FACE_NAMES)), dtype=torch.float, device=self.device)
        self.goal_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_hold_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_orientation_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._step_object_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._step_target_face = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._step_goal_rotation = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # The parent constructor performs an initial reset before these buffers
        # exist.  Initialize the semantic goal for that first rollout too.
        self._reset_target_pose(torch.arange(self.num_envs, device=self.device))

    def _reset_target_pose(self, env_ids):
        # DirectRLEnv may reset once during the parent constructor, before our
        # goal-token buffers have been allocated.
        if not hasattr(self, "target_face"):
            return super()._reset_target_pose(env_ids)
        # Six cube face normals: +x, -x, +y, -y, +z, -z.  The base
        # orientation maps the selected local normal to world +z; a random
        # yaw makes the task genuinely SO(3)-valued rather than a face lookup.
        n = len(env_ids)
        if self.cfg.fixed_target_face is None:
            face = torch.randint(0, 6, (n,), device=self.device)
        else:
            face = torch.full((n,), int(self.cfg.fixed_target_face), dtype=torch.long, device=self.device)
        zero = torch.zeros(n, device=self.device)
        half_pi = torch.full((n,), 1.5707963267948966, device=self.device)
        neg_half_pi = -half_pi
        pi = torch.full((n,), 3.141592653589793, device=self.device)
        # Quaternion convention is (w, x, y, z).
        base = torch.zeros((n, 4), device=self.device)
        base[:, 0] = 1.0
        base[face == 0] = quat_from_euler_xyz(zero[face == 0], neg_half_pi[face == 0], zero[face == 0])
        base[face == 1] = quat_from_euler_xyz(zero[face == 1], half_pi[face == 1], zero[face == 1])
        base[face == 2] = quat_from_euler_xyz(half_pi[face == 2], zero[face == 2], zero[face == 2])
        base[face == 3] = quat_from_euler_xyz(neg_half_pi[face == 3], zero[face == 3], zero[face == 3])
        base[face == 5] = quat_from_euler_xyz(pi[face == 5], zero[face == 5], zero[face == 5])
        if self.cfg.goal_yaw is None:
            yaw = sample_uniform(-3.141592653589793, 3.141592653589793, (n,), device=self.device)
        else:
            yaw = torch.full((n,), float(self.cfg.goal_yaw), device=self.device)
        yaw_q = quat_from_euler_xyz(zero, zero, yaw)
        self.goal_rot[env_ids] = quat_mul(yaw_q, base)
        self.target_face[env_ids] = face
        # Advanced indexing returns a copy; .zero_() on that copy would leave
        # previous target bits set after repeated goal resets.
        self.target_face_onehot[env_ids] = encode_face_goal(face)

        # Keep the existing visual goal marker for debugging and evaluation.
        goal_pos = self.goal_pos + self.scene.env_origins
        dot_pos = self.in_hand_pos + self.scene.env_origins
        dot_pos[:, 2] += 0.02
        marker_pos = torch.cat((goal_pos, dot_pos), dim=0)
        marker_rot = torch.cat((self.goal_rot, self.goal_rot), dim=0)
        marker_indices = torch.cat(
            (
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs, dtype=torch.long, device=self.device),
            ),
            dim=0,
        )
        self.goal_markers.visualize(marker_pos, marker_rot, marker_indices=marker_indices)
        self.reset_goal_buf[env_ids] = 0
        if hasattr(self, "goal_hold_steps"):
            self.goal_hold_steps[env_ids] = 0
            self.goal_success_latched[env_ids] = False
            self._step_goal_reached[env_ids] = False
            self._step_dropped[env_ids] = False
            self._step_hold_complete[env_ids] = False
            self._step_success_event[env_ids] = False

    def compute_full_observations(self):
        # Preserve the exact RevoLab state observation and append an explicit
        # semantic goal token for the teacher policy.
        obs = super().compute_full_observations()
        return torch.cat((obs, self.target_face_onehot), dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update hold state and compute done flags before reward/reset."""

        self._compute_intermediate_values()
        self._step_orientation_error = rotation_distance(self.object_rot, self.goal_rot)
        self._step_object_distance = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1)
        # Snapshot the goal before `_get_rewards` can reset successful slots.
        self._step_target_face = self.target_face.clone()
        self._step_goal_rotation = self.goal_rot.clone()
        self._step_goal_reached = self._step_orientation_error <= self.cfg.success_tolerance
        self._step_dropped = self._step_object_distance >= self.cfg.fall_dist
        required_steps = required_goal_hold_steps(float(self.cfg.goal_hold_time_s), float(self.step_dt))
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

        out_of_reach = self._step_dropped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0 and not self.cfg.freeze_goal_for_episode:
            time_out = time_out | (
                self.successes + self._step_success_event.to(self.successes.dtype)
                >= self.cfg.max_consecutive_success
            )
        return out_of_reach, time_out

    def _get_rewards(self):
        """Compute rewards using the current state before DirectRLEnv resets."""

        goal_dist = self._step_object_distance
        rot_dist = self._step_orientation_error
        dist_rew = goal_dist * self.cfg.dist_reward_scale
        rot_rew = 1.0 / (torch.abs(rot_dist) + self.cfg.rot_eps) * self.cfg.rot_reward_scale
        action_penalty = torch.sum(self.actions**2, dim=-1)
        action_slew_penalty = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)
        reward = (
            dist_rew
            + rot_rew
            + action_penalty * self.cfg.action_penalty_scale
            + action_slew_penalty * self.cfg.action_slew_penalty_scale
        )

        # Give the actor credit for each qualified hold step.  The progress
        # term is normalized by the required dwell, so its maximum per-step
        # contribution is the configured scale and it cannot grow with a
        # longer hold requirement.  A zero scale preserves the sparse reward
        # used by the original baseline.
        if self.cfg.hold_progress_reward_scale != 0.0:
            required_steps = required_goal_hold_steps(float(self.cfg.goal_hold_time_s), float(self.step_dt))
            hold_progress = (self.goal_hold_steps.to(reward.dtype) / float(required_steps)).clamp(0.0, 1.0)
            reward = reward + hold_progress * self.cfg.hold_progress_reward_scale

        self.successes = self.successes + self._step_success_event.to(self.successes.dtype)
        reward = torch.where(
            self._step_success_event,
            reward + self.cfg.reach_goal_bonus,
            reward,
        )
        reward = torch.where(
            self._step_dropped,
            reward + self.cfg.fall_penalty,
            reward,
        )

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

        # Only completed holds trigger target changes.  A frozen-goal run
        # keeps the target and continues until timeout/drop for stability eval.
        self.reset_goal_buf[:] = self._step_success_event
        if self.cfg.freeze_goal_for_episode:
            self.reset_goal_buf.zero_()
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        # Keep the base task's exponentially averaged success statistic for
        # existing RSL-RL dashboards.  `reset_buf` was computed immediately
        # before this reward call and therefore still refers to the current
        # (pre-reset) transition.
        num_resets = torch.sum(self.reset_buf)
        finished_successes = torch.sum(self.successes * self.reset_buf.to(self.successes.dtype))
        self.consecutive_successes[:] = torch.where(
            num_resets > 0,
            self.cfg.av_factor * finished_successes / num_resets
            + (1.0 - self.cfg.av_factor) * self.consecutive_successes,
            self.consecutive_successes,
        )
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.successes.mean()
        return reward

    @property
    def language_goal(self) -> torch.Tensor:
        """Structured language goal consumed by the teacher and student."""
        return self.target_face_onehot

    def current_instructions(self) -> list[str]:
        """Human-readable instructions for logs and rollout metadata."""
        return batch_instructions(self.target_face)
