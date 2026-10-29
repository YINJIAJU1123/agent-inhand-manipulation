"""Paired single-environment diagnostics for released and direct-latent students.

This runner measures fixed-goal reach before drop using simulated perception
noise or calibrated rendered RGB. It is not an autonomous semantic-agent benchmark.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time
import types

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--seed', type=int, default=182901)
parser.add_argument('--policy', choices=('released', 'latent'), required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--student_checkpoint', type=Path, required=True)
parser.add_argument('--text_cache', type=Path, required=True)
parser.add_argument('--original_asset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--formal_release', type=Path)
parser.add_argument('--formal_condition')
parser.add_argument('--prepare_only',action='store_true',help='Initialize the measured-pose runtime and export its contract without scheduled trials.')
parser.add_argument('--max_steps', type=int, default=3600)
parser.add_argument('--pose_source', choices=('simulated', 'gs_cnn'), default='simulated')
parser.add_argument('--pose_checkpoint', type=Path)
parser.add_argument('--camera_calibration', type=Path)
add_runtime_args(parser)
args = parse_runtime_args(parser)
args.num_envs = 1
if not 0 < args.episodes < 1000 or args.seed < 0 or not 0 < args.max_steps <= 3600:
    parser.error('Require 1..999 episodes, nonnegative seed, and 1..3600 maximum steps')
if args.output.exists():
    raise FileExistsError(f'Refusing to overwrite {args.output}')
if args.pose_source == 'gs_cnn' and (not args.pose_checkpoint or not args.camera_calibration):
    parser.error('RGB feedback requires a frozen CNN checkpoint and camera calibration')

from formal_release import load_release,verify_runtime,save_preparation
formal_release=load_release(args,Path(__file__).parent,kind='diagnostic')


import torch
from viserdex.distillation.environment import SavedEnvironmentBinding, environment_contract
from viserdex.distillation.trainer import load_student_checkpoint
from goals import load_bank, COLORS, SYMBOLS
from model import LatentStudent, PROP_KEEP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = load_student_checkpoint(args.student_checkpoint)
binding = SavedEnvironmentBinding(args.student_checkpoint.parent / 'env.yaml',
                                  source['metadata']['rollout_environment_sha256'])
if binding.contract_sha != source['metadata']['environment_binding']['contract_sha256']:
    raise ValueError('Original student environment contract changed')
physical_binding = {}


def verify_physical_configuration(cfg):
    if args.pose_source == 'simulated':
        return binding.verify(cfg)
    # Remove only explicitly added visual sensors from the comparison copy.
    # Robot/object physics, actions, events, rewards and terminations stay checked.
    actual = environment_contract(cfg.to_dict())
    for key in ('base', 'gsplat_camera', 'hand_depth_camera'):
        if key in binding.contract['scene']:
            actual['scene'][key] = binding.contract['scene'][key]
        else:
            actual['scene'].pop(key, None)
    if 'pose' in binding.contract['observations']:
        actual['observations']['pose'] = binding.contract['observations']['pose']
    else:
        actual['observations'].pop('pose', None)
    if actual != binding.contract:
        changed = [key for key in set(actual) | set(binding.contract)
                   if actual.get(key) != binding.contract.get(key)]
        raise ValueError(f'Adding RGB changed the bound physical configuration: {changed}')
    return physical_binding


def configure(cfg):
    binding.configure(cfg, args)
    physical_binding.update(binding.verify(cfg))
    if args.pose_source == 'gs_cnn':
        from viserdex.renderer.hikrobot_occlusion import configure_original_capture
        configure_original_capture(cfg, args.camera_calibration)
        cfg.observations.pose.image_features.params['debug_vis'] = False
        verify_physical_configuration(cfg)


app = SimulationRuntime(args, configure=configure)

import gymnasium as gym
from tensordict import TensorDict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import viserdex.tasks  # noqa: F401
from viserdex.distillation.config import PerceptionNoiseCfg, StudentCfg
from viserdex.distillation.frames import validate_pose_metadata
from viserdex.distillation.model import BeliefStudent
from viserdex.distillation.observations import StudentObservations
from viserdex.distillation.teacher import configure_clean_state_observations, audit_clean_state_observations


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def angle(quaternion, target):
    q, t = quaternion.double(), target.double()
    cosine = ((q * t).sum(-1) / (q.norm(dim=-1) * t.norm(dim=-1))).abs().clamp(0, 1)
    error = 2 * torch.acos(cosine)
    if not torch.isfinite(error).all():
        raise FloatingPointError('Invalid diagnostic orientation')
    return float(error[0])


def initial_state(raw):
    """Actual reset state and action buffers, not a seed-based claim of equality."""
    robot, obj = raw.scene['robot'], raw.scene['object']
    values = {
        'robot_root_state_w': robot.data.root_state_w.torch,
        'object_root_state_w': obj.data.root_state_w.torch,
        'joint_position': robot.data.joint_pos.torch,
        'joint_velocity': robot.data.joint_vel.torch,
        'manager_action': raw.action_manager.action,
        'manager_previous_action': raw.action_manager.prev_action,
    }
    term = raw.action_manager.get_term('joint_pos')
    values['raw_joint_action'] = term.raw_actions
    values['processed_joint_action'] = term.processed_actions
    values['ema_previous_applied_joint_action'] = term._prev_applied_actions
    torch.testing.assert_close(term._prev_applied_actions,
        robot.data.joint_pos.torch[:, term._joint_ids], atol=0, rtol=0)
    record = {name: value.detach().cpu().tolist() for name, value in values.items()}
    # Isaac Lab retains the old processed output cache across reset, but replaces
    # it before the next apply_action. EMA memory is the actual causal state.
    causal = {key:value for key,value in record.items() if key != 'processed_joint_action'}
    digest = hashlib.sha256(json.dumps(causal, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    return record, digest


def main():
    device = args.device or 'cuda:0'
    if args.pose_source == 'gs_cnn':
        torch.backends.cuda.matmul.allow_tf32 = False
    validate_pose_metadata(source['metadata'])
    if source['metadata']['object'] != 'RSLCube':
        raise ValueError('This diagnostic is bound to the released RSLCube')
    cfg_student = StudentCfg(**source['student_cfg'])
    vectors, quaternions, goal_manifest = load_bank(args.text_cache, args.original_asset, device)
    if args.policy == 'released':
        if sha(args.checkpoint) != sha(args.student_checkpoint):
            raise ValueError('Released comparator must be the supplied original student')
        policy = BeliefStudent(cfg_student)
        policy.load_state_dict(source['student'], strict=True)
        checkpoint_details = {'iteration': source['iteration'], 'format': source['format']}
    else:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if payload['format'] != 'pose_latent_student_v1' or payload['original_student_cfg'] != asdict(cfg_student):
            raise ValueError('Wrong latent checkpoint architecture or format')
        if payload['metadata']['source_student_sha256'] != sha(args.student_checkpoint):
            raise ValueError('Latent checkpoint was trained from a different released student')
        if payload['noise_cfg'] != source['noise_cfg']:
            raise ValueError('Latent and released perception-noise protocols differ')
        for key in ('asset_sha256', 'cache_sha256'):
            if payload['metadata']['goal_bank'][key] != goal_manifest[key]:
                raise ValueError(f'Latent goal bank changed: {key}')
        policy = LatentStudent(cfg_student, payload['latent_dim'], payload['goal_width'])
        policy.load_state_dict(payload['model'], strict=True)
        checkpoint_details = {key: payload[key] for key in ('format', 'stage', 'completed_updates')}
    policy = policy.to(device).eval().requires_grad_(False)
    noise_cfg = PerceptionNoiseCfg(**source['noise_cfg'])
    if 'joint_delay_max_steps' not in source['noise_cfg']:
        noise_cfg.joint_delay_max_steps = 0
    cfg = app.env_cfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = device
    cfg.log_dir = str(args.output.resolve())
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    configure_clean_state_observations(cfg)
    verified_binding = verify_physical_configuration(cfg)
    args.output.mkdir(parents=True)
    dump_yaml(str(args.output / 'env.yaml'), cfg)
    schedule = [dict(episode=i, trial_seed=args.seed * 1000 + i,
                     goal_index=(i + 2 * (args.seed % 2)) % 6) for i in range(args.episodes)]
    manifest = {
        'version': 'pose_latent_student_diagnostic_v1', 'policy': args.policy,
        'checkpoint_sha256': sha(args.checkpoint), 'checkpoint': checkpoint_details,
        'original_student_sha256': sha(args.student_checkpoint),
        'physical_environment_binding': verified_binding, 'goal_bank': goal_manifest,
        'schedule': schedule, 'startup_randomization_seed': args.seed,
        'reset_seed_rule': 'base_seed * 1000 + episode_index; same schedule required for both policies',
        'initial_state_digest_scope': 'actual roots, joints, raw actions and EMA memory; stale processed output cache excluded; startup dynamics use matched seed and saved binding',
        'goal_distribution_override': {
            'distribution': 'six audited cardinal goals in deterministic cyclic order',
            'fixed_during_trial': True, 'update_goal_on_success': False,
            'time_resampling': 'time_left set beyond external horizon after each reset',
            'reset_sampler': 'original resampler runs, then its orientation is replaced by the assigned goal',
        },
        'metric': 'full-quaternion reach <=0.16 rad before any object_out_of_reach termination',
        'stopping': 'diagnostic trial ends at first reach, drop, original environment termination, or external cap',
        'initial_reach': 'counts as diagnostic reach and is separately recorded',
        'simultaneous_drop_and_reach': 'drop takes precedence',
        'not_autonomous_semantic_success': True, 'teacher_used': False,
        'formal_trials_used': formal_release is not None,
        'formal_condition': args.formal_condition,
        'formal_release_sha256': sha(args.formal_release) if formal_release is not None else None,
        'pose_source': ('released simulated perception noise; no rendered CNN inference'
                        if args.pose_source == 'simulated' else 'calibrated GS object plus black hand occlusion, frozen RGB CNN'),
        'full_scene_rgb': False, 'visual_noise_added_to_cnn_pose': False,
        'noise_cfg': asdict(noise_cfg), 'physical_randomization': 'unchanged startup and reset events',
        'step_dt': cfg.sim.dt * cfg.decimation, 'external_max_steps': args.max_steps,
        'external_horizon_seconds': args.max_steps * cfg.sim.dt * cfg.decimation,
        'preserved_environment_episode_seconds': cfg.episode_length_s,
        'policy_inputs': ['proprio109', 'current_noisy_pose7', 'continuous_latent768'] if args.policy == 'latent'
                         else ['released_proprio113_including_goal_quaternion', 'released_extero11'],
        'ground_truth_scope': 'simulation sensor generation and evaluator diagnostics only; collect-only pose observation is excluded from policy inputs',
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('evaluate.py', 'model.py', 'goals.py', 'perception.py')},
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    save_json(args.output / 'manifest.json', manifest)
    stop = {'requested': False}
    handlers = {s: signal.signal(s, lambda *_: stop.update(requested=True)) for s in (signal.SIGTERM, signal.SIGINT)}
    env = None
    records = []
    started = time.monotonic()
    status = 'interrupted'
    try:
        # Model construction consumes RNG. Seed again immediately before the
        # environment creates its randomized physical properties in both arms.
        torch.manual_seed(args.seed)
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        raw = env.unwrapped
        if raw.num_envs != 1 or abs(raw.step_dt - 1 / 30) > 1e-9:
            raise ValueError('Expected one environment with the released 30 Hz control period')
        command = raw.command_manager.get_term('object_pose')
        vision_term = estimator = None
        if args.pose_source == 'gs_cnn':
            from isaaclab.utils.math import matrix_from_quat
            from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func
            from viserdex.renderer.hikrobot_occlusion import audit_original_capture
            from perception import ReleasedPoseEstimator
            vision_term = _get_observation_term_func(raw.observation_manager, 'pose', 'image_features')
            vision_term.capture_enabled = True
            estimator = ReleasedPoseEstimator(args.pose_checkpoint, vision_term.pts_path, device)
            manifest['estimator'] = estimator.metadata
            manifest['camera'] = audit_original_capture(vision_term, vision_term._hand_depth_camera,
                                                       raw.scene['robot'], args.camera_calibration)
            save_json(args.output / 'manifest.json', manifest)
        if formal_release is not None:
            verify_runtime(formal_release,args.formal_condition,manifest,kind='diagnostic')
            save_json(args.output / 'formal_release.json',formal_release)
        if args.prepare_only:
            manifest.update(preparation_only=True)
            save_json(args.output / 'manifest.json',manifest)
            save_preparation(args.output,Path(__file__).parent,'diagnostic')
            status='prepared'
            return
        assigned = {'goal': 0}
        original_resample = command._resample_command

        def resample(self, env_ids):
            original_resample(env_ids)
            self.quat_command_w[env_ids] = quaternions[assigned['goal']]

        command._resample_command = types.MethodType(resample, command)
        command.cfg.update_goal_on_success = False
        manager = raw.termination_manager
        original_compute = manager.compute
        observed = {}

        def compute(self):
            result = original_compute()
            # Isaac Lab resets done environments before env.step returns.
            # Capture the exact scored state before that reset, without changing
            # any termination or reward values.
            observed['quaternion'] = raw.scene['object'].data.root_quat_w.torch.detach().clone()
            observed['terms'] = {name: self.get_term(name).detach().clone() for name in self.active_terms}
            observed['done'] = result.detach().clone()
            return result

        manager.compute = types.MethodType(compute, manager)
        drop_cfg = manager.get_term_cfg('object_out_of_reach')
        with torch.inference_mode(), (args.output / 'steps.jsonl').open('w', buffering=1) as step_stream:
            for case in schedule:
                if stop['requested'] or not app.is_running():
                    break
                assigned['goal'] = case['goal_index']
                raw.reset(seed=case['trial_seed'])
                command.quat_command_w[:] = quaternions[assigned['goal']]
                command.time_left.fill_((args.max_steps + 1) * raw.step_dt)
                # Only control groups contain the changed goal. Re-rendering the
                # pose group here would read a camera already consumed by reset.
                for group in ('policy', 'privileged'):
                    raw.obs_buf[group] = raw.observation_manager.compute_group(group)
                obs = TensorDict(raw.obs_buf, batch_size=[1])
                audit_clean_state_observations(raw, obs)
                reset_state, state_digest = initial_state(raw)
                adapter = StudentObservations(raw, cfg_student, noise_cfg)
                hidden = policy.initial_state(1, device)
                target = quaternions[assigned['goal']:assigned['goal'] + 1]
                error = angle(raw.scene['object'].data.root_quat_w.torch, target)
                dropped = bool(drop_cfg.func(raw, **drop_cfg.params)[0])
                initial_error = error
                initial_reached = error <= .16 and not dropped
                minimum_error = error
                reason = 'drop' if dropped else 'reach' if error <= .16 else None
                steps = 0
                terminal_terms = {}
                last_quaternion = raw.scene['object'].data.root_quat_w.torch.detach().clone()
                while reason is None and steps < args.max_steps:
                    if stop['requested'] or not app.is_running():
                        reason = 'interrupted'
                        break
                    perception_fit = None
                    if estimator is None:
                        prop, extero, _ = adapter(obs)
                    else:
                        robot = raw.scene['robot']
                        measurement = estimator.predict_rectified(
                            vision_term.masked_rgb.float() / 255,
                            vision_term.camera.data.intrinsic_matrices.torch[0],
                            matrix_from_quat(vision_term.camera_quat_ros), vision_term.camera_pos,
                            matrix_from_quat(robot.data.root_quat_w.torch), robot.data.root_pos_w.torch,
                        )
                        prop, extero, _ = adapter(obs, estimated_pose=(measurement['position'], measurement['quaternion']))
                        perception_fit = float(measurement['fit_error_m'][0])
                    if args.policy == 'latent':
                        action, _, hidden = policy(prop[:, PROP_KEEP], extero[:, :7],
                                                   vectors[assigned['goal']:assigned['goal'] + 1], hidden)
                    else:
                        action, _, hidden = policy(prop, extero, hidden)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('Non-finite student action')
                    observed.clear()
                    obs, _, dones, _ = env.step(action)
                    steps += 1
                    adapter.record_action(action, dones.bool())
                    last_quaternion = observed['quaternion']
                    terminal_terms = {name: bool(value[0]) for name, value in observed['terms'].items()}
                    error = angle(last_quaternion, target)
                    minimum_error = min(minimum_error, error)
                    dropped = terminal_terms['object_out_of_reach']
                    reason = ('drop' if dropped else 'reach' if error <= .16 else
                              'environment_termination' if bool(observed['done'][0]) else None)
                    if reason is None and steps == args.max_steps:
                        reason = 'external_horizon'
                    step_stream.write(json.dumps(dict(episode=case['episode'], step=steps,
                        orientation_error_rad=error, dropped=dropped, reason=reason,
                        object_quaternion_world_xyzw=last_quaternion[0].cpu().tolist(),
                        perception_fit_error_m=perception_fit,
                        raw_action_rms=float(action.square().mean().sqrt())), allow_nan=False) + '\n')
                    if steps % 100 == 0:
                        save_json(args.output / 'progress.json', dict(status='running', completed_episodes=len(records),
                            requested_episodes=args.episodes, current_episode=case['episode'], step=steps,
                            minimum_error_rad=minimum_error, elapsed_seconds=time.monotonic() - started))
                record = dict(**case, color=COLORS[assigned['goal']], symbol=SYMBOLS[assigned['goal']],
                    initial_state_sha256=state_digest, initial_state=reset_state,
                    initial_error_rad=initial_error, initially_reached=initial_reached,
                    steps=steps, seconds=steps * raw.step_dt, reason=reason,
                    reached_before_drop=reason == 'reach', dropped=dropped,
                    final_error_rad=error, minimum_error_rad=minimum_error,
                    target_quaternion_world_xyzw=target[0].cpu().tolist(),
                    final_object_quaternion_world_xyzw=last_quaternion[0].cpu().tolist(),
                    termination_terms=terminal_terms)
                records.append(record)
                with (args.output / 'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                progress = dict(status='running', completed_episodes=len(records), requested_episodes=args.episodes,
                                reached=sum(r['reached_before_drop'] for r in records),
                                drops=sum(r['dropped'] for r in records), elapsed_seconds=time.monotonic() - started)
                save_json(args.output / 'progress.json', progress)
                print('[POSE_LATENT_EVAL] ' + json.dumps(progress), flush=True)
                if reason == 'interrupted':
                    break
        status = 'completed' if len(records) == args.episodes and all(r['reason'] != 'interrupted' for r in records) else 'interrupted'
    except Exception as exc:
        status = 'failed'
        save_json(args.output / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        reached = sum(r['reached_before_drop'] for r in records)
        summary = dict(status=status, policy=args.policy, requested_episodes=args.episodes,
            recorded_episodes=len(records), reached=reached, drops=sum(r['dropped'] for r in records),
            reach_before_drop_rate=reached / len(records) if records else None,
            formal_autonomous_semantic_success_rate=None,
            formal_diagnostic_result=formal_release is not None,
            formal_condition=args.formal_condition,
            formal_release_sha256=sha(args.formal_release) if formal_release is not None else None,
            evaluation_scope='runtime_preparation' if args.prepare_only else f"{args.pose_source} low-level {'formal' if formal_release is not None else 'development'} diagnostics; no semantic supervisor",
            termination_counts={reason: sum(r['reason'] == reason for r in records)
                                for reason in sorted({r['reason'] for r in records})},
            per_goal={COLORS[i]: dict(episodes=sum(r['goal_index'] == i for r in records),
                      reached=sum(r['goal_index'] == i and r['reached_before_drop'] for r in records)) for i in range(6)},
            elapsed_seconds=time.monotonic() - started)
        save_json(args.output / 'result.json', summary)
        save_json(args.output / 'progress.json', summary)
        for s, handler in handlers.items():
            signal.signal(s, handler)
        if env is not None:
            env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
