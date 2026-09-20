"""Image-conditioned search with randomized marker placement and stable display.

No teacher observation is exposed as actor input. True object geometry is
restricted to scoring/critic. Frame capture happens after moving visual patches
and again after automatic reset, avoiding pre-reset image/post-reset goal pairs.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .visual_semantic_reorient import VisualSemanticReorientEnv
from BrainCo_DexHand.algo.agentic.search_protocol import (
    NORMALS, layout_bank, image_features, image_color_masks, instruction,
    quat_rotate, project_points, update_dwell,
)


class SemanticSearchEnv(VisualSemanticReorientEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.initial_visibility not in ("visible", "hidden", "mixed"):
            raise ValueError("initial_visibility must be visible, hidden, or mixed")
        self.layouts = layout_bank(cfg.layout_split, self.device)
        self.layout_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.face_colors = torch.arange(6, device=self.device).repeat(self.num_envs, 1)
        self.target_color = torch.zeros_like(self.layout_index)
        self.initial_hidden = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending = torch.ones_like(self.initial_hidden)
        self.episode_id = torch.zeros_like(self.layout_index)
        self.dwell = torch.zeros_like(self.layout_index)
        self.ever_display = torch.zeros_like(self.initial_hidden)
        self.ever_exposed = torch.zeros_like(self.initial_hidden)
        self.first_exposure = torch.full_like(self.layout_index, -1)
        self.first_display = torch.full_like(self.layout_index, -1)
        self._frames = None
        self._cached_observation = None
        self.reset_rejections = 0
        # 7x7 samples across the actual 28 mm patch. Surface z is 37 mm:
        # parent marker center is 35.5 mm with thickness 3 mm.
        uv = torch.linspace(-.012, .012, 7, device=self.device)
        xx, yy = torch.meshgrid(uv, uv, indexing="ij")
        patch = torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), .037)), -1)
        self.patch_points = quat_rotate(self._face_marker_rots[:, None], patch[None])

    def _reset_target_pose(self, env_ids):
        # Preserve parent initialization, but keep one semantic target per episode.
        if not hasattr(self, "face_colors"):
            super()._reset_target_pose(env_ids)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "face_colors"):
            return
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        n = len(env_ids)
        indices = torch.randint(len(self.layouts), (n,), device=self.device)
        self.layout_index[env_ids] = indices
        self.face_colors[env_ids] = self.layouts[indices]
        self.initial_hidden[env_ids] = (torch.rand(n, device=self.device) < .5
                                       if self.cfg.initial_visibility == "mixed"
                                       else self.cfg.initial_visibility == "hidden")
        self.pending[env_ids] = True
        self.episode_id[env_ids] += 1
        self.dwell[env_ids] = 0
        self.ever_display[env_ids] = False
        self.ever_exposed[env_ids] = False
        self.first_exposure[env_ids] = -1
        self.first_display[env_ids] = -1
        self._frames = None
        self._cached_observation = None

    def _update_face_markers(self):
        if not hasattr(self, "face_colors"):
            return super()._update_face_markers()
        from isaaclab.utils.math import quat_mul
        rot = self.object_rot[:, None]
        offsets = quat_rotate(rot, .0355 * self._face_normals[None])
        positions = self.object_pos[:, None] + offsets + self.scene.env_origins[:, None]
        rotations = quat_mul(rot.expand(-1, 6, -1), self._face_marker_rots[None].expand(self.num_envs, -1, -1))
        self.face_markers.visualize(positions.reshape(-1, 3), rotations.reshape(-1, 4),
                                    marker_indices=self.face_colors.flatten())

    def _pre_physics_step(self, actions):
        self._frames = None
        self._cached_observation = None
        super()._pre_physics_step(actions)

    def _refresh_frames(self):
        if self._frames is None:
            self._compute_intermediate_values()
            self._update_face_markers()
            self.sim.render()
            if self.pending.any():
                self.sim.render()
            self.camera.update(self.step_dt, force_recompute=True)
            out = self.camera.data.output
            self._frames = {"rgb": out["rgb"][..., :3].clone(),
                            "depth": out["distance_to_image_plane"].clone()}
        return self._frames

    def capture_camera(self, clone=True, refresh=True):
        frames = self._refresh_frames()
        return {k: v.clone() if clone else v for k, v in frames.items()}

    def _surface_metrics(self):
        frames = self._refresh_frames()
        pos_w = self.object_pos + self.scene.env_origins
        points = quat_rotate(self.object_rot[:, None], self.patch_points.flatten(0, 1)[None]) + pos_w[:, None]
        cam = self.camera.data
        pixels, depth = project_points(points, cam.pos_w, cam.quat_w_ros, cam.intrinsic_matrices)
        n, h, w = frames["depth"].shape[:3]
        x, y = pixels.unbind(-1)
        inside = (depth > .05) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        xi, yi = x.round().long().clamp(0, w - 1), y.round().long().clamp(0, h - 1)
        observed = frames["depth"].reshape(n, h, w)[torch.arange(n, device=self.device)[:, None], yi, xi]
        valid = inside & torch.isfinite(observed) & ((observed - depth).abs() <= self.cfg.depth_tolerance)
        fraction = valid.reshape(n, 6, -1).float().mean(-1)
        normals = quat_rotate(self.object_rot[:, None], self._face_normals[None])
        toward_camera = F.normalize(cam.pos_w - pos_w, dim=-1)
        facing = (normals * toward_camera[:, None]).sum(-1)
        center_depth = depth.reshape(n, 6, -1).mean(-1).clamp_min(.05)
        projected = .028**2 * cam.intrinsic_matrices[:, 0, 0, None] * cam.intrinsic_matrices[:, 1, 1, None]
        projected = projected * facing.clamp_min(0) / center_depth.square()
        fraction = torch.where(facing > 0, fraction, 0.)
        return fraction, projected, facing, normals

    def _choose_pending_targets(self):
        # Rejection sample resets that have no genuinely visible patch; hidden
        # samples must be back-facing. Conditions use post-reset camera frames.
        for _ in range(25):
            fraction, area, facing, _ = self._surface_metrics()
            counts = image_color_masks(self._frames["rgb"]).sum((1, 2))
            visible = (fraction >= self.cfg.minimum_visible_fraction) & (area >= self.cfg.minimum_projected_area)
            visible &= counts.gather(1, self.face_colors) >= 8
            hidden = (fraction <= .02) & (facing < -.2)
            pending_ids = self.pending.nonzero(as_tuple=False).flatten()
            if not len(pending_ids):
                break
            want_hidden = self.initial_hidden[pending_ids].clone()
            candidates = torch.where(want_hidden[:, None], hidden[pending_ids], visible[pending_ids])
            accepted = candidates.any(-1)
            good_ids = pending_ids[accepted]
            if len(good_ids):
                scores = torch.rand((len(good_ids), 6), device=self.device).masked_fill(~candidates[accepted], -1)
                face = scores.argmax(-1)
                self.target_face[good_ids] = face
                colors = self.face_colors[good_ids, face]
                # Roundtrip through the declared language grammar: semantic token,
                # not a privileged face token.
                self.target_color[good_ids] = colors
                self.initial_hidden[good_ids] = want_hidden[accepted]
                self.pending[good_ids] = False
                exposed = fraction[good_ids, face] >= self.cfg.minimum_visible_fraction
                self.ever_exposed[good_ids] = exposed
                self.first_exposure[good_ids] = torch.where(exposed, 0, -1)
            bad_ids = pending_ids[~accepted]
            if not len(bad_ids):
                return
            self.reset_rejections += len(bad_ids)
            self._reset_idx(bad_ids)
            self.initial_hidden[bad_ids] = want_hidden[~accepted]
            self.scene.write_data_to_sim()
            self.sim.forward()
        if self.pending.any():
            ids = self.pending.nonzero(as_tuple=False).flatten()
            print("RESET_DIAGNOSTICS", {"ids": ids.tolist(), "fraction": fraction[ids].tolist(),
                                        "area": area[ids].tolist(), "facing": facing[ids].tolist(),
                                        "counts": counts[ids].tolist()}, flush=True)
            raise RuntimeError("Could not create requested visible/hidden resets; inspect rendering and thresholds")

    def _get_observations(self):
        if not hasattr(self, "face_colors"):
            return super()._get_observations()
        if self.pending.any():
            self._choose_pending_targets()
        if self._cached_observation is not None:
            return self._cached_observation
        frames = self._refresh_frames()
        ids = self.actuated_dof_indices
        lower, upper = self.hand_dof_lower_limits[:, ids], self.hand_dof_upper_limits[:, ids]
        proprio = torch.cat((2 * (self.hand_dof_pos[:, ids] - lower) / (upper - lower) - 1,
                             .2 * self.hand_dof_vel[:, ids], self.actions), -1)
        actor = torch.cat((image_features(frames["rgb"], frames["depth"]), proprio,
                           F.one_hot(self.target_color, 6).float()), -1)
        normals = quat_rotate(self.object_rot, self._face_normals[self.target_face])
        critic = torch.cat((actor, self.object_pos, self.object_rot,
                            self.object_linvel, self.object_angvel, normals), -1)
        assert actor.shape[-1] == self.cfg.observation_space
        assert critic.shape[-1] == self.cfg.state_space
        self._cached_observation = {"policy": actor, "critic": critic}
        return self._cached_observation

    def _get_dones(self):
        self._compute_intermediate_values()
        fraction, area, facing, _ = self._surface_metrics()
        row = torch.arange(self.num_envs, device=self.device)
        visible = fraction[row, self.target_face]
        projected = area[row, self.target_face]
        cosine = facing[row, self.target_face]
        speed = self.object_linvel.norm(dim=-1)
        angular = self.object_angvel.norm(dim=-1)
        exposed = (visible >= self.cfg.minimum_visible_fraction) & (projected >= self.cfg.minimum_projected_area)
        exposed &= cosine >= math.cos(math.radians(self.cfg.maximum_facing_angle_deg))
        qualified = exposed & (speed <= self.cfg.maximum_linear_speed) & (angular <= self.cfg.maximum_angular_speed)
        distance = (self.object_pos - self.in_hand_pos).norm(dim=-1)
        dropped = distance >= self.cfg.fall_dist
        self.dwell, complete = update_dwell(self.dwell, qualified, dropped, self.step_dt, self.cfg.goal_hold_time_s)
        first_success = complete & ~self.ever_display
        self.first_exposure = torch.where(exposed & ~self.ever_exposed, self.episode_length_buf, self.first_exposure)
        self.first_display = torch.where(first_success, self.episode_length_buf, self.first_display)
        self.ever_exposed |= exposed
        self.ever_display |= complete
        self._search_metrics = {
            "visible_fraction": visible.clone(), "projected_area": projected.clone(),
            "facing_cosine": cosine.clone(), "linear_speed": speed.clone(), "angular_speed": angular.clone(),
            "exposed": exposed.clone(), "qualified": qualified.clone(), "held_at_end": complete.clone(),
            "ever_display": self.ever_display.clone(), "ever_exposed": self.ever_exposed.clone(),
            "first_exposure_step": self.first_exposure.clone(), "first_display_step": self.first_display.clone(),
            "dropped": dropped.clone(), "initial_hidden": self.initial_hidden.clone(),
            "target_color": self.target_color.clone(), "target_face": self.target_face.clone(),
            "layout_index": self.layout_index.clone(), "layout": self.face_colors.clone(),
            "episode_id": self.episode_id.clone(), "steps": self.episode_length_buf.clone(),
        }
        # Same task reward for every PPO mechanism/control comparison. Object
        # target geometry is privileged reward supervision, never actor input.
        self._search_reward = (1.5 * cosine.clamp_min(0) + 3. * visible
                               + 2. * qualified.float() + 4. * complete.float()
                               + 20. * first_success.float() - 10. * distance
                               - .0004 * self.actions.square().sum(-1)
                               - .01 * (self.actions - self.prev_actions).square().sum(-1)
                               - 25. * dropped.float())
        return dropped, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        self.extras["search_metrics"] = self._search_metrics
        self.extras["log"] = {
            "search/visible_fraction": self._search_metrics["visible_fraction"].mean(),
            "search/display": self._search_metrics["held_at_end"].float().mean(),
            "search/reset_rejections": float(self.reset_rejections),
        }
        return self._search_reward

    def current_instructions(self):
        return [instruction(c) for c in self.target_color.tolist()]
