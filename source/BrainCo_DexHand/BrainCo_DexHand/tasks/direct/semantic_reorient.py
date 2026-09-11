"""Goal-conditioned semantic-surface teacher for Revo3.

This task keeps the original RevoLab 21-DoF action interface, but samples a
target cube face (plus a random in-plane yaw) instead of an unconstrained
orientation.  The face id is exposed as a six-dimensional goal token for the
state-based teacher.  A future visual-language student can replace this token
with an image/language embedding without changing the action interface.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_mul, sample_uniform

from .inhand_manipulation_env import InHandManipulationEnv


class SemanticReorientEnv(InHandManipulationEnv):
    """Revo3 cube reorientation conditioned on a target surface id."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.target_face = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_face_onehot = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

    def _reset_target_pose(self, env_ids):
        # DirectRLEnv may reset once during the parent constructor, before our
        # goal-token buffers have been allocated.
        if not hasattr(self, "target_face"):
            return super()._reset_target_pose(env_ids)
        # Six cube face normals: +x, -x, +y, -y, +z, -z.  The base
        # orientation maps the selected local normal to world +z; a random
        # yaw makes the task genuinely SO(3)-valued rather than a face lookup.
        n = len(env_ids)
        face = torch.randint(0, 6, (n,), device=self.device)
        zero = torch.zeros(n, device=self.device)
        half_pi = torch.full((n,), 1.5707963267948966, device=self.device)
        neg_half_pi = -half_pi
        pi = torch.full((n,), 3.141592653589793, device=self.device)
        # Quaternion convention is (w, x, y, z).
        base = torch.zeros((n, 4), device=self.device)
        base[:, 0] = 1.0
        base[face == 0] = quat_from_euler_xyz(zero[face == 0], half_pi[face == 0], zero[face == 0])
        base[face == 1] = quat_from_euler_xyz(zero[face == 1], neg_half_pi[face == 1], zero[face == 1])
        base[face == 2] = quat_from_euler_xyz(neg_half_pi[face == 2], zero[face == 2], zero[face == 2])
        base[face == 3] = quat_from_euler_xyz(half_pi[face == 3], zero[face == 3], zero[face == 3])
        base[face == 5] = quat_from_euler_xyz(pi[face == 5], zero[face == 5], zero[face == 5])
        yaw = sample_uniform(-3.141592653589793, 3.141592653589793, (n,), device=self.device)
        yaw_q = quat_from_euler_xyz(zero, zero, yaw)
        self.goal_rot[env_ids] = quat_mul(yaw_q, base)
        self.target_face[env_ids] = face
        self.target_face_onehot[env_ids].zero_()
        self.target_face_onehot[env_ids, face] = 1.0

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

    def compute_full_observations(self):
        # Preserve the exact RevoLab state observation and append an explicit
        # semantic goal token for the teacher policy.
        obs = super().compute_full_observations()
        return torch.cat((obs, self.target_face_onehot), dim=-1)
