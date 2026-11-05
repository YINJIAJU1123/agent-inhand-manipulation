"""Explicit benchmark-only adapter for the released dice dynamics.

Copied init/read_state/step retain source audits with three documented changes:
120-second horizon, no success goal resampling, and original 20-second feature
clock. Source physics, sensors, actions, rewards and geometry are unchanged.
"""
from viserdex.supermimic.simulation import *
from viserdex.supermimic.hold_command import HoldReorientCommand

class BenchmarkHoldCommand(HoldReorientCommand):
    def _update_command(self):
        if self.cfg.update_goal_on_success:
            raise ValueError('Benchmark command must keep a goal until an explicit skill switch')
        # Release the reset guard, but do not call _resample on earned events.
        self._success.earned(self.just_completed)

class BenchmarkAdapter(Adapter):
    def __init__(self, env, noise_cfg, history_length=10):
        self.env, self.noise_cfg = env, noise_cfg
        command_cfg = env.command_manager.get_term('object_pose').cfg
        self.hold_cfg = config20()
        if (command_cfg.orientation_success_threshold != self.hold_cfg.hold_angle_threshold
                or command_cfg.hold_speed_threshold != self.hold_cfg.hold_speed_threshold
                or command_cfg.hold_duration != self.hold_cfg.hold_duration
                or command_cfg.update_goal_on_success):
            raise ValueError('Hold success protocol differs from the task contract')
        self.robot, self.obj = env.scene["robot"], env.scene["object"]
        ids, names = self.robot.find_joints(list(REVO3_JOINT_NAMES), preserve_order=True)
        if list(names) != list(REVO3_JOINT_NAMES):
            raise ValueError("Joint order changed")
        self.ids = ids
        self.limits = tensor(self.robot.data.soft_joint_pos_limits)[:, ids].clone()
        self.base_inverse = quat_conjugate(tensor(self.robot.data.root_quat_w)).clone()
        self.raw_history = torch.zeros(env.num_envs, 84, device=env.device)
        self.ema = tensor(self.robot.data.default_joint_pos)[:, ids].clone()
        self.sensors = initial_sensors(env.num_envs, env.device, noise_cfg)
        self.sensors = reset_sensors(
            self.sensors, torch.ones(env.num_envs, device=env.device, dtype=torch.bool), noise_cfg
        )
        self.state = self.read_state()
        obs, self.sensors = observe(self.state, self.sensors, self.limits, noise_cfg)
        self.history = obs[:, None].expand(-1, history_length, -1).clone()
        self.context, self.context_schema = self.read_context()
        self.captured_physical = None
        self.captured_target = None
        self.captured_second_diff = None
        self.captured_hand_pose = None
        self.captured_orientation_error = None
        self.max_errors = dict(reward=0.0, ema=0.0, clean_student_observation=0.0)
        self.geometry_spec = build_geometry_spec(
            env.cfg.scene.robot.spawn.usd_path, env.cfg.scene.object.spawn.usd_path,
            getattr(env.cfg.scene.robot.spawn, "scale", None), getattr(env.cfg.scene.object.spawn, "scale", None),
        )
        if self.geometry_spec["object_shape"] != "convex_mesh":
            raise ValueError("RoundedDie requires its authored convex mesh; sphere checkpoints are incompatible")
        self.geometry = HandObjectGeometry(self.geometry_spec).to(env.device)
        from viserdex.supermimic.convex_audit import audit_live_convex
        self.convex_audit = audit_live_convex(self.obj, env, self.geometry_spec)
        self.max_errors.update(self.convex_audit["errors"])
        self.geometry_body_ids, found = self.robot.find_bodies(self.geometry.body_names, preserve_order=True)
        if list(found) != self.geometry.body_names:
            raise ValueError("Geometry body ordering differs from simulator")
        self.geometry_audit_steps = 0
        self.audit_geometry(self.state)
        self.reset_count = 0
        env.supermimic_adapter = self
        if abs(env.step_dt - 1 / 20) > 1e-8 or abs(env.max_episode_length * env.step_dt - 120) > 1e-8:
            raise ValueError("Benchmark requires 20 Hz and 120 seconds")
        action_cfg = env.action_manager.get_term("joint_pos").cfg
        self.action_alpha = action_cfg.alpha
        self.success_bonus = env.cfg.rewards.success_bonus.weight
        self.action_rate_weight = -env.cfg.rewards.action_rate_l2.weight
        self.joint_vel_weight = -env.cfg.rewards.joint_vel_l2.weight
        self.action_l2_weight = -env.cfg.rewards.action_l2.weight
        self.action_second_diff_weight = -env.cfg.rewards.action_second_diff_l2.weight
        torch.testing.assert_close(tensor(self.robot.data.default_joint_pos)[:, ids],
            self.limits.new_tensor(self.hold_cfg.action_home).expand(env.num_envs, -1), atol=1e-7, rtol=0)
        if action_cfg.rescale_to_limits or action_cfg.offset_scale != self.hold_cfg.action_offset_scale or action_cfg.warmup_time_s != self.hold_cfg.action_warmup_s:
            raise ValueError("Wuji home-offset action contract changed")
        if abs(env.cfg.rewards.hand_pose.weight + self.hold_cfg.hand_pose_weight) > 1e-9:
            raise ValueError("Hand-pose reward weight differs between simulator and imagination")
        expected_alpha = .5
        if abs(action_cfg.alpha - expected_alpha) > 1e-8 or action_cfg.scale != 1.0 or action_cfg.clip is not None:
            raise ValueError("Action contract changed")

    def read_state(self):
        env = self.env
        command = env.command_manager.get_command("object_pose")
        base_local = tensor(self.robot.data.root_pos_w) - env.scene.env_origins
        goal_pos = quat_apply(self.base_inverse, command[:, :3] - base_local)
        goal_quat = quat_unique(quat_mul(self.base_inverse, command[:, 3:7]))
        count = env.command_manager.get_term("object_pose").metrics["consecutive_success"][:, None]
        return torch.cat(
            (
                self.physical(),
                self.ema,
                self.raw_history,
                goal_pos,
                goal_quat,
                (1 - env.episode_length_buf[:, None].float() / round(20 / env.step_dt)).clamp(0, 1),
                count,
                base_local,
                env.command_manager.get_term("object_pose").hold_ticks[:, None].clone(),
            ),
            -1,
        )

    def step(self, action):
        state, history, sensors = self.state, self.history, self.sensors
        expected_target = action_target(state, action, self.limits, self.hold_cfg)
        self.captured_physical = None
        self.captured_orientation_error = None
        obs, actual_reward, terminated, truncated, info = self.env.step(action)
        if self.captured_physical is None or self.captured_orientation_error is None:
            raise RuntimeError("Official pre-reset recorder did not execute")
        done = terminated | truncated
        actual_reward = actual_reward.clone()
        done = done.clone()
        expected_pose = (self.captured_physical[:, Q] - state.new_tensor(self.hold_cfg.action_home)).square().sum(-1)
        torch.testing.assert_close(self.captured_hand_pose, expected_pose, atol=2e-6, rtol=1e-5)
        self.max_errors["hand_pose"] = max(self.max_errors.get("hand_pose", 0.), float((self.captured_hand_pose - expected_pose).abs().max()))
        expected_second = (action - 2 * state[:, 76:97] + state[:, 97:118]).square().sum(-1)
        torch.testing.assert_close(self.captured_second_diff, expected_second, atol=2e-6, rtol=1e-5)
        self.max_errors["action_second_diff"] = max(self.max_errors.get("action_second_diff", 0.), float((self.captured_second_diff - expected_second).abs().max()))
        torch.testing.assert_close(self.captured_angular_speed, self.captured_physical[:, OMEGA].norm(dim=-1), atol=2e-6, rtol=1e-5)
        expected_hold, expected_event = goal_event(state, self.captured_physical, self.env.step_dt,
            error=self.captured_orientation_error, speed=self.captured_angular_speed,
            hold_cfg=self.hold_cfg, blocked=done)
        torch.testing.assert_close(expected_event, self.captured_completion)
        torch.testing.assert_close(expected_hold, self.captured_hold, atol=0, rtol=0)
        self.max_errors['hold_counter'] = float((expected_hold-self.captured_hold).abs().max())
        audit = audit_real_reward(
            state, self.captured_physical, action, self.captured_orientation_error, actual_reward, self.env.step_dt, self.success_bonus, self.action_rate_weight, self.joint_vel_weight, self.action_l2_weight, self.action_second_diff_weight, hold_cfg=self.hold_cfg, success_event=expected_event
        )
        for key in ("reward", "orientation_frame"):
            self.max_errors[key] = max(self.max_errors.get(key, 0.0), audit[key])
        self.max_errors["success_boundary_disagreements"] = (
            self.max_errors.get("success_boundary_disagreements", 0) + audit["success_boundary_disagreements"]
        )
        self.max_errors["ema"] = max(
            self.max_errors["ema"], float((expected_target - self.captured_target).abs().max())
        )
        torch.testing.assert_close(expected_target, self.captured_target, atol=2e-6, rtol=1e-5)
        record = dict(
            state=state,
            history=history,
            action=action.clone(),
            next_physical=self.captured_physical,
            done=done,
            terminated=terminated.clone(),
            reward=actual_reward,
            success=self.captured_completion.clone(),
        )
        record.update({"sensor/" + k: v for k, v in sensors.items()})
        self.ema = torch.where(done[:, None], tensor(self.robot.data.default_joint_pos)[:, self.ids], self.captured_target)
        self.raw_history = torch.cat((action, self.raw_history[:, :63]), -1)
        self.raw_history = torch.where(done[:, None], 0.0, self.raw_history)
        self.sensors = reset_sensors(sensors, done, self.noise_cfg)
        self.state = self.read_state()
        terminal_state = advance(state, self.captured_physical, action, self.captured_target,
                                 self.state[:, GOAL_ROT], round(20 / self.env.step_dt), hold_cfg=self.hold_cfg,
                                 event_override=(expected_hold, expected_event))
        torch.testing.assert_close(self.state[:, HOLD].squeeze(-1), expected_hold, atol=0, rtol=0)
        if (~done).any():
            torch.testing.assert_close(self.state[~done, COUNT].squeeze(-1),
                                       state[~done, COUNT].squeeze(-1) + expected_event[~done])
        unchanged = ~done & ~expected_event
        torch.testing.assert_close(self.state[unchanged, GOAL_ROT], state[unchanged, GOAL_ROT])
        record["next_aux"] = torch.where(done[:, None], terminal_state, self.state)[:, 55:].clone()
        observation, self.sensors = observe(self.state, self.sensors, self.limits, self.noise_cfg)
        self.history = push_history(history, observation, done)
        self.reset_count += int(done.sum())
        if done.any():
            separation = (self.captured_physical[done, :45] - self.state[done, :45]).abs().max()
            self.max_errors["terminal_reset_state_separation"] = max(
                self.max_errors.get("terminal_reset_state_separation", 0.0), float(separation)
            )
        return record, expected_target, observation, obs
