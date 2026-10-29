"""Newton rollout distillation of the direct-latent pose student.

Run only after the released student passes the unmodified runtime smoke check.
Choose released simulated pose noise or the frozen calibrated RGB CNN.
Goal and object ground truth remain teacher/auxiliary labels only.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--student_checkpoint', type=Path, required=True, help='Relocated original release')
parser.add_argument('--initial_checkpoint', type=Path, help='Previous migrated checkpoint, for a new stage')
parser.add_argument('--teacher_checkpoint', type=Path)
parser.add_argument('--text_cache', type=Path, required=True)
parser.add_argument('--original_asset', type=Path, required=True)
parser.add_argument('--pose_source', choices=('simulated', 'gs_cnn'), default='simulated')
parser.add_argument('--pose_checkpoint', type=Path)
parser.add_argument('--camera_calibration', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--stage', choices=('interface', 'adapt'), required=True)
parser.add_argument('--num_envs', type=int, default=128)
parser.add_argument('--max_iterations', type=int, default=200)
parser.add_argument('--rollout_steps', type=int, default=45)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--env_batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--save_interval', type=int, default=20)
parser.add_argument('--seed', type=int, default=182801)
add_runtime_args(parser)
args = parse_runtime_args(parser)
if min(args.num_envs, args.max_iterations, args.rollout_steps, args.epochs,
       args.env_batch_size, args.save_interval, args.learning_rate) <= 0:
    parser.error('Training counts and learning rate must be positive')
if args.stage == 'adapt' and (not args.initial_checkpoint or not args.teacher_checkpoint):
    parser.error('Adaptation requires a migrated initialization and the released teacher')
if args.pose_source == 'gs_cnn' and (not args.pose_checkpoint or not args.camera_calibration):
    parser.error('RGB training requires the frozen CNN and camera calibration')
if args.output.exists():
    raise FileExistsError(f'Refusing to overwrite {args.output}')

import torch
from viserdex.distillation.environment import SavedEnvironmentBinding, environment_contract
from viserdex.distillation.trainer import load_student_checkpoint
from viserdex.distillation.teacher import teacher_metadata
from goals import load_bank, install_training_goals
from model import LatentStudent, PROP_KEEP, configure_training, migrate

source = load_student_checkpoint(args.student_checkpoint)
binding = SavedEnvironmentBinding(args.student_checkpoint.parent / 'env.yaml',
                                  source['metadata']['rollout_environment_sha256'])
if binding.contract_sha != source['metadata']['environment_binding']['contract_sha256']:
    raise ValueError('Original student environment contract changed')
teacher_info = teacher_metadata(args.teacher_checkpoint, 'RSLCube') if args.stage == 'adapt' else None
if teacher_info and teacher_info['sha256'] != source['metadata']['teacher']['sha256']:
    raise ValueError('Adaptation must use the original released teacher')
physical_binding = {}


def verify_physical_configuration(cfg):
    if args.pose_source == 'simulated':
        return binding.verify(cfg)
    actual = environment_contract(cfg.to_dict())
    for section, names in (('scene', ('base', 'gsplat_camera', 'hand_depth_camera')),
                           ('observations', ('pose',))):
        for name in names:
            if name in binding.contract[section]:
                actual[section][name] = binding.contract[section][name]
            else:
                actual[section].pop(name, None)
    if actual != binding.contract:
        raise ValueError('Adding RGB capture changed the released physical configuration')
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
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.distillation.frames import validate_pose_metadata
from viserdex.distillation.model import BeliefStudent
from viserdex.distillation.observations import StudentObservations
from viserdex.distillation.teacher import (
    configure_clean_state_observations, audit_clean_state_observations,
    load_teacher, teacher_observations,
)


def main():
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    validate_pose_metadata(source['metadata'])
    if source['metadata']['object'] != 'RSLCube':
        raise ValueError('Only the released cube is admitted to this migration')
    student, migration = migrate(source, args.seed)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)
        if initial['format'] != 'pose_latent_student_v1':
            raise ValueError('Wrong migrated checkpoint format')
        if initial['metadata']['source_student_sha256'] != hashlib.sha256(args.student_checkpoint.read_bytes()).hexdigest():
            raise ValueError('Initialization was trained from a different released student')
        if initial['original_student_cfg'] != asdict(student.original_cfg):
            raise ValueError('Student architecture changed between stages')
        student.load_state_dict(initial['model'], strict=True)
    device = args.device or 'cuda:0'
    student = student.to(device).train()
    hooks = configure_training(student, args.stage)
    # No weight decay: inherited input columns must remain unchanged in warmup.
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=0)
    reference = BeliefStudent(student.original_cfg).to(device).eval()
    reference.load_state_dict(source['student'], strict=True)
    reference.requires_grad_(False)
    noise_cfg = PerceptionNoiseCfg(**source['noise_cfg'])
    if 'joint_delay_max_steps' not in source['noise_cfg']:
        noise_cfg.joint_delay_max_steps = 0
    vectors, target_quaternions, goal_manifest = load_bank(args.text_cache, args.original_asset, device)
    if args.initial_checkpoint:
        for key in ('asset_sha256', 'cache_sha256'):
            if initial['metadata']['goal_bank'][key] != goal_manifest[key]:
                raise ValueError('Initialization goal bank changed: ' + key)
    cfg = app.env_cfg()
    cfg.scene.num_envs = args.num_envs
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
    metadata = {
        'format': 'pose_latent_student_v1', 'stage': args.stage,
        'label_source': 'frozen_released_student' if args.stage == 'interface' else 'frozen_released_ppo_teacher',
        'source_student_sha256': hashlib.sha256(args.student_checkpoint.read_bytes()).hexdigest(),
        'initial_checkpoint_sha256': hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest() if args.initial_checkpoint else None,
        'source_metadata': source['metadata'], 'physical_environment_binding': verified_binding,
        'training_goal_distribution_override': 'six audited cardinal goals, uniform per resample',
        'goal_bank': goal_manifest, 'migration': migration,
        'pose_source': ('released simulated perception noise; no CNN inference in this stage'
                        if args.pose_source == 'simulated' else 'calibrated GS object plus black hand occlusion, frozen RGB CNN'),
        'visual_noise_added_to_cnn_pose': False, 'ground_truth_pose_fallback': False,
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('train.py', 'model.py', 'goals.py', 'perception.py')},
        'normalization': 'frozen inherited physical statistics and identity latent statistics',
        'evaluation_scope': 'training diagnostics only, not the frozen 40-episode benchmark',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    stop = {'requested': False}
    old_handlers = {s: signal.signal(s, lambda *_: stop.update(requested=True)) for s in (signal.SIGTERM, signal.SIGINT)}
    env = None
    completed = 0

    def save():
        path = args.output / f'latent_student_{completed:05d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save({
            'format': 'pose_latent_student_v1', 'stage': args.stage,
            'completed_updates': completed, 'model': student.state_dict(),
            'optimizer': optimizer.state_dict(), 'metadata': metadata,
            'original_student_cfg': asdict(student.original_cfg),
            'latent_dim': student.latent_dim, 'goal_width': student.goal_width,
            'noise_cfg': source['noise_cfg'], 'rng_state': torch.get_rng_state(),
        }, temp)
        temp.replace(path)
        return path

    try:
        # Reset after model construction so both sensor variants share the startup seed.
        torch.manual_seed(args.seed)
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        raw = env.unwrapped
        if abs(raw.step_dt - 1 / 30) > 1e-9:
            raise ValueError('Released control period changed')
        from collections import Counter
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton import GeoType, ShapeFlags
        physics_model = NewtonManager.get_model()
        collision = (physics_model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        geometry = dict(world_count=physics_model.world_count, body_count=physics_model.body_count,
                        collision_shape_count=int(collision.sum()),
                        collision_types=dict(Counter(GeoType(int(t)).name for t in physics_model.shape_type.numpy()[collision])))
        if geometry['world_count'] != raw.num_envs or geometry['collision_types'] != {'CONVEX_MESH':116*raw.num_envs, 'BOX':raw.num_envs}:
            raise ValueError('Unexpected live collision geometry: ' + str(geometry))
        metadata['live_geometry'] = geometry
        print('LIVE_COLLISION_GEOMETRY', json.dumps(geometry), flush=True)
        estimator = None
        if args.pose_source == 'gs_cnn':
            from isaaclab.utils.math import matrix_from_quat
            from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func
            from viserdex.renderer.hikrobot_occlusion import audit_original_capture
            from perception import ReleasedPoseEstimator
            vision = _get_observation_term_func(raw.observation_manager, 'pose', 'image_features')
            vision.capture_enabled = True
            estimator = ReleasedPoseEstimator(args.pose_checkpoint, vision.pts_path, device)
            metadata['pose_estimator'] = estimator.metadata
            metadata['camera'] = audit_original_capture(vision, vision._hand_depth_camera, raw.scene['robot'], args.camera_calibration)
        raw.reset(seed=args.seed)
        command = raw.command_manager.get_term('object_pose')
        ids = install_training_goals(command, target_quaternions)
        for group in ('policy', 'privileged'):
            raw.obs_buf[group] = raw.observation_manager.compute_group(group)
        (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        obs = TensorDict(raw.obs_buf, batch_size=[env.num_envs])
        audit_clean_state_observations(raw, obs)
        teacher = load_teacher(teacher_info, obs, device) if teacher_info else None
        adapter = StudentObservations(raw, student.original_cfg, noise_cfg)
        hidden = student.initial_state(env.num_envs, device)
        reference_hidden = reference.initial_state(env.num_envs, device)
        reset = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        while completed < args.max_iterations and app.is_running() and not stop['requested']:
            started = time.monotonic()
            # Reference actions collect warmup data; DAgger transitions to the
            # current student, with its teacher still labeling every state.
            beta = 1. if args.stage == 'interface' else .9 * .95 ** completed
            batch = {k: [] for k in ('prop', 'pose', 'latent', 'label', 'target', 'reset')}
            initial_hidden = hidden.detach().clone()
            rewards, drops, dones_count = 0., 0, 0
            position_errors, orientation_errors, fit_errors = [], [], []
            with torch.no_grad():
                for _ in range(args.rollout_steps):
                    if estimator is None:
                        prop, pose, target = adapter(obs)
                    else:
                        robot = raw.scene['robot']
                        measure = estimator.predict_rectified(
                            vision.masked_rgb.float()/255,
                            vision.camera.data.intrinsic_matrices.torch[0],
                            matrix_from_quat(vision.camera_quat_ros), vision.camera_pos,
                            matrix_from_quat(robot.data.root_quat_w.torch), robot.data.root_pos_w.torch)
                        prop, pose, target = adapter(obs, estimated_pose=(measure['position'], measure['quaternion']))
                        fit_errors.append(float(measure['fit_error_m'].mean()))
                    position_errors.append(float((pose[:, :3]-target[:, :3]).norm(dim=-1).mean()))
                    cosine = torch.nn.functional.cosine_similarity(pose[:, 3:7].double(), target[:, 3:7].double()).abs().clamp(0, 1)
                    orientation_errors.append(float((2*torch.acos(cosine)).mean()))
                    latent = vectors[ids]
                    action, _, hidden = student(prop[:, PROP_KEEP], pose[:, :7], latent, hidden)
                    if teacher is None:
                        label, _, reference_hidden = reference(prop, pose, reference_hidden)
                    else:
                        label = teacher(teacher_observations(obs, prop, source['metadata']['teacher_quaternion_sign']))
                    selected = torch.rand(env.num_envs, 1, device=device) < beta
                    executed = torch.where(selected, label, action)
                    values = (prop[:, PROP_KEEP], pose[:, :7], latent, label,
                              torch.cat((target[:, :7], target[:, 11:]), -1), reset)
                    for key, value in zip(batch, values):
                        batch[key].append(value.detach().clone())
                    obs, reward, dones, _ = env.step(executed)
                    reset = dones.bool()
                    adapter.record_action(executed, reset)
                    hidden *= (~reset)[None, :, None]
                    reference_hidden *= (~reset)[None, :, None]
                    rewards += float(reward.mean())
                    dones_count += int(reset.sum())
                    if 'object_out_of_reach' in raw.termination_manager.active_terms:
                        drops += int(raw.termination_manager.get_term('object_out_of_reach').sum())
            batch = {k: torch.stack(v) for k, v in batch.items()}
            if not all(torch.isfinite(v).all() for v in batch.values()):
                raise FloatingPointError('Non-finite rollout')
            losses, action_losses = [], []
            for _ in range(args.epochs):
                order = torch.randperm(env.num_envs, device=device)
                for indices in order.split(args.env_batch_size):
                    prediction, reconstruction, _ = student.sequence(
                        batch['prop'][:, indices], batch['pose'][:, indices], batch['latent'][:, indices],
                        initial_hidden[:, indices], batch['reset'][:, indices])
                    action_loss = (prediction - batch['label'][:, indices]).square().mean()
                    loss = action_loss
                    if args.stage == 'adapt':
                        target = student.core.target_norm(batch['target'][:, indices])
                        loss = loss + .2 * (reconstruction - target).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite optimization loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    action_losses.append(float(action_loss.detach()))
            with torch.no_grad():
                _, _, hidden = student.sequence(batch['prop'], batch['pose'], batch['latent'], initial_hidden, batch['reset'])
                hidden *= (~reset)[None, :, None]
            completed += 1
            metrics = dict(iteration=completed, stage=args.stage, teacher_action_fraction=beta,
                           training_action_mse=sum(action_losses)/len(action_losses),
                           training_loss=sum(losses)/len(losses), rollout_terminations=dones_count,
                           rollout_out_of_reach=drops, training_mean_reward=rewards/args.rollout_steps,
                           environment_steps=completed*env.num_envs*args.rollout_steps,
                           pose_position_error_m=sum(position_errors)/len(position_errors),
                           pose_orientation_error_rad=sum(orientation_errors)/len(orientation_errors),
                           pose_fit_error_m=sum(fit_errors)/len(fit_errors) if fit_errors else None,
                           elapsed_seconds=time.monotonic()-started)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            temporary = args.output / 'progress.tmp'
            temporary.write_text(json.dumps(metrics, indent=2) + '\n')
            temporary.replace(args.output / 'progress.json')
            print('[POSE_LATENT_TRAIN] ' + json.dumps(metrics), flush=True)
            if completed == 1 or completed % args.save_interval == 0:
                save()
        checkpoint = save()
        (args.output / 'completion.json').write_text(json.dumps({
            'completed': completed >= args.max_iterations, 'updates': completed,
            'checkpoint': str(checkpoint), 'reason': 'budget' if completed >= args.max_iterations else 'interrupted',
        }, indent=2) + '\n')
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
        for hook in hooks:
            hook.remove()
        if env is not None:
            env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
