"""Collect new-student RGB trajectories for recovery and completion learning.

Mixed wait behaviors deliberately include failures. Simulation state labels all
six hypothetical goals at the same physical state, but model features contain
only measured RGB pose, proprioception, student memory and public request state.
This is training data with scripted delays, not real Astra evaluation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import types

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
for name in ('student_checkpoint', 'checkpoint', 'teacher_checkpoint', 'text_cache',
             'original_asset', 'pose_checkpoint', 'camera_calibration', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=2400)
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--split', choices=('train', 'validation'), required=True)
parser.add_argument('--continuous_protocol',action='store_true')
parser.add_argument('--maintenance_checkpoint',type=Path)
parser.add_argument('--latency_profile',type=Path)
parser.add_argument('--agent_schedule_profile',type=Path,help='Replay public request/action phases on fresh training states.')
parser.add_argument('--grounding_capture',action='store_true',help='Also save aligned RGB/geometry labels for semantic adaptation.')
parser.add_argument('--maintenance_wait_only',action='store_true',help='Apply the learned behavior only while requests are pending.')
add_runtime_args(parser)
args = parse_runtime_args(parser)
if args.agent_schedule_profile and (not args.continuous_protocol or args.latency_profile):
    raise ValueError('Agent schedules require the continuous protocol and supply their own latency.')
if args.output.exists():
    raise FileExistsError(args.output)

import torch
from viserdex.distillation.environment import SavedEnvironmentBinding, environment_contract
from viserdex.distillation.trainer import load_student_checkpoint
from viserdex.distillation.teacher import teacher_metadata
from goals import load_bank
from model import LatentStudent, PROP_KEEP

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source = load_student_checkpoint(args.student_checkpoint)
binding = SavedEnvironmentBinding(args.student_checkpoint.parent / 'env.yaml',
                                  source['metadata']['rollout_environment_sha256'])
teacher_info = teacher_metadata(args.teacher_checkpoint, 'RSLCube')
if teacher_info['sha256'] != source['metadata']['teacher']['sha256']:
    raise ValueError('Require the original teacher')
physical = {}

def verify(cfg):
    actual = environment_contract(cfg.to_dict())
    for section, names in (('scene', ('base', 'gsplat_camera', 'hand_depth_camera')),
                           ('observations', ('pose',))):
        for name in names:
            if name in binding.contract[section]:
                actual[section][name] = binding.contract[section][name]
            else:
                actual[section].pop(name, None)
    if args.continuous_protocol:
        actual['episode_length_s']=binding.contract['episode_length_s']
        for name in ('max_consecutive_success','object_out_of_reach'):
            actual['terminations'][name]=binding.contract['terminations'][name]
    if actual != binding.contract:
        raise ValueError('Physical environment changed while adding visual capture')

def configure(cfg):
    binding.configure(cfg, args)
    physical.update(binding.verify(cfg))
    from viserdex.renderer.hikrobot_occlusion import configure_original_capture
    configure_original_capture(cfg, args.camera_calibration)
    cfg.observations.pose.image_features.params['debug_vis'] = False
    if args.continuous_protocol:
        from newton_runtime import benchmark_drop
        cfg.episode_length_s=120.
        cfg.terminations.max_consecutive_success=None
        cfg.terminations.object_out_of_reach.func=benchmark_drop
        cfg.terminations.object_out_of_reach.params={'threshold':.24}
    verify(cfg)

app = SimulationRuntime(args, configure=configure)

import gymnasium as gym
from tensordict import TensorDict
from isaaclab.utils.math import matrix_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import viserdex.tasks
from viserdex.distillation.config import StudentCfg, PerceptionNoiseCfg
from viserdex.distillation.observations import StudentObservations
from viserdex.distillation.teacher import configure_clean_state_observations, audit_clean_state_observations, load_teacher, teacher_observations
from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func
from viserdex.renderer.hikrobot_occlusion import audit_original_capture
from perception import ReleasedPoseEstimator

def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def main():
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device or 'cuda:0'
    vectors, rotations, goal_manifest = load_bank(args.text_cache, args.original_asset, device)
    pack = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg_student = StudentCfg(**source['student_cfg'])
    if (pack['format'] != 'pose_latent_student_v1' or pack['original_student_cfg'] != asdict(cfg_student)
            or pack['metadata']['source_student_sha256'] != sha(args.student_checkpoint)):
        raise ValueError('Incompatible student checkpoint')
    for key in ('asset_sha256', 'cache_sha256'):
        if pack['metadata']['goal_bank'][key] != goal_manifest[key]:
            raise ValueError('Goal bank changed')
    student = LatentStudent(cfg_student, pack['latent_dim'], pack['goal_width']).to(device).eval()
    student.load_state_dict(pack['model'])
    student.requires_grad_(False)
    maintenance=None
    if args.maintenance_checkpoint:
        from control_modules import MaintenanceActor,features
        saved=torch.load(args.maintenance_checkpoint,map_location='cpu',weights_only=True)
        if saved['manifest']['source_student_sha256']!=sha(args.checkpoint):
            raise ValueError('Maintenance student mismatch')
        torch.testing.assert_close(saved['bank'],vectors.cpu(),atol=0,rtol=0)
        maintenance=MaintenanceActor(saved['feature_mean'],saved['feature_scale'],saved['bank']).to(device).eval()
        maintenance.load_state_dict(saved['model']);maintenance.requires_grad_(False)
    latency_values=None
    if args.latency_profile:
        latency_values=torch.tensor(json.loads(args.latency_profile.read_text())['latencies_s'],device=device)
        if len(latency_values)<10 or not torch.isfinite(latency_values).all() or not ((latency_values>0)&(latency_values<=120)).all():
            raise ValueError('Invalid empirical latency profile')
    schedule_profile=None
    if args.agent_schedule_profile:
        schedule_profile=json.loads(args.agent_schedule_profile.read_text())
        if schedule_profile['version']!='public_agent_control_schedule_v1' or schedule_profile['split']!=args.split:
            raise ValueError('Unexpected agent schedule contract')
        sequences=schedule_profile['schedules']
        if not sequences or any(not 1<=len(s['calls'])<=6 for s in sequences):
            raise ValueError('Expected nonempty schedules with at most six calls')
        schedule_duration=torch.ones((len(sequences),6),device=device,dtype=torch.long)
        schedule_goal=torch.full_like(schedule_duration,-1)
        schedule_known=torch.full_like(schedule_duration,-1)
        schedule_execution=torch.full_like(schedule_duration,30)
        schedule_length=torch.tensor([len(s['calls']) for s in sequences],device=device)
        for i,sequence in enumerate(sequences):
            for j,call in enumerate(sequence['calls']):
                if not 1<=call['duration_steps']<=3600 or not -1<=call['goal']<6 or not 10<=call['execution_steps']<=90:
                    raise ValueError('Invalid public request schedule')
                schedule_duration[i,j]=call['duration_steps']
                schedule_goal[i,j]=call['goal']
                schedule_known[i,j]=-1 if call['known'] is None else int(call['known'])
                schedule_execution[i,j]=call['execution_steps']
    def durations(count):
        if latency_values is None:return torch.randint(30,181,(count,),device=device)
        return (latency_values[torch.randint(len(latency_values),(count,),device=device)]*30).ceil().long()
    cfg = app.env_cfg()
    cfg.seed, cfg.log_dir = args.seed, str(args.output.resolve())
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    configure_clean_state_observations(cfg)
    verify(cfg)
    args.output.mkdir(parents=True)
    env = None
    grounding_store = None
    try:
        torch.manual_seed(args.seed)
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        raw = env.unwrapped
        command = raw.command_manager.get_term('object_pose')
        command.cfg.update_goal_on_success = False
        vision = _get_observation_term_func(raw.observation_manager, 'pose', 'image_features')
        vision.capture_enabled = True
        estimator = ReleasedPoseEstimator(args.pose_checkpoint, vision.pts_path, device)
        camera = audit_original_capture(vision, vision._hand_depth_camera, raw.scene['robot'], args.camera_calibration)
        from collections import Counter
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton import GeoType, ShapeFlags
        physics_model = NewtonManager.get_model()
        collision = (physics_model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        geometry = dict(world_count=physics_model.world_count, body_count=physics_model.body_count,
                        collision_shape_count=int(collision.sum()),
                        collision_types=dict(Counter(GeoType(int(t)).name for t in physics_model.shape_type.numpy()[collision])))
        n = raw.num_envs
        if geometry['world_count'] != n or geometry['collision_types'] != {'CONVEX_MESH': 116*n, 'BOX': n}:
            raise ValueError(f'Unexpected live collision geometry: {geometry}')
        print('LIVE_COLLISION_GEOMETRY', json.dumps(geometry), flush=True)
        goal = torch.ones(n, device=device, dtype=torch.long)
        original_resample = command._resample_command
        def resample(self, env_ids):
            original_resample(env_ids)
            goal[env_ids] = 1
            self.quat_command_w[env_ids] = rotations[1]
        command._resample_command = types.MethodType(resample, command)
        raw.reset(seed=args.seed)
        command.quat_command_w[:] = rotations[goal]
        command.time_left.fill_(1e6)
        def control_obs():
            for group in ('policy', 'privileged'):
                raw.obs_buf[group] = raw.observation_manager.compute_group(group)
            return TensorDict(raw.obs_buf, batch_size=[n])
        obs = control_obs()
        audit_clean_state_observations(raw, obs)
        teacher = load_teacher(teacher_info, obs, device)
        noise_cfg = PerceptionNoiseCfg(**source['noise_cfg'])
        if 'joint_delay_max_steps' not in source['noise_cfg']:
            noise_cfg.joint_delay_max_steps = 0
        adapter = StudentObservations(raw, cfg_student, noise_cfg)
        hidden = student.initial_state(n, device)
        episode = torch.zeros(n, dtype=torch.long, device=device)
        known = torch.zeros(n, dtype=torch.bool, device=device)
        pending = torch.ones(n, dtype=torch.bool, device=device)
        elapsed = torch.zeros(n, dtype=torch.long, device=device)
        duration = durations(n)
        if schedule_profile:
            schedule_id=torch.randint(len(sequences),(n,),device=device)
            schedule_cursor=torch.zeros(n,dtype=torch.long,device=device)
            duration=schedule_duration[schedule_id,schedule_cursor].clone()
        until_request = torch.zeros(n, dtype=torch.long, device=device)
        need_anchor = torch.ones(n, dtype=torch.bool, device=device)
        anchor_pose = torch.zeros(n, 7, device=device)
        anchor_rotation = torch.zeros(n, 4, device=device)
        anchor_action = torch.zeros(n, 21, device=device)
        behavior = torch.arange(n, device=device) % (5 if maintenance else 4)
        manager = raw.termination_manager
        original_compute = manager.compute
        terminal = {}
        def termination(self):
            value = original_compute()
            terminal['drop'] = self.get_term('object_out_of_reach').detach().clone()
            terminal['position'] = raw.scene['object'].data.root_pos_w.torch.detach().clone()
            return value
        manager.compute = types.MethodType(termination, manager)
        manifest = dict(version='pose_latent_control_collection_v1', split=args.split, seed=args.seed,
            checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
            teacher_sha256=sha(args.teacher_checkpoint), pose_estimator=estimator.metadata,
            camera=camera, live_geometry=geometry, physical_environment_binding=physical, goal_bank=goal_manifest,
            original_environment_seconds=binding.contract['episode_length_s'],runtime_environment_seconds=cfg.episode_length_s,
            continuous_protocol=args.continuous_protocol,step_dt=raw.step_dt,
            maintenance_sha256=sha(args.maintenance_checkpoint) if maintenance else None,
            latency_profile_sha256=sha(args.latency_profile) if args.latency_profile else None,
            latency_profile=json.loads(args.latency_profile.read_text()) if args.latency_profile else None,
            observations='RGB CNN pose, measured joints/action history, student belief, request timing and measured anchor only',
            labels='ground-truth orientation errors for all six goals; frozen teacher actions for each goal and request-anchor rotation',
            waits='empirical latency replay; no new upstream calls' if latency_values is not None else 'scripted 1..6 seconds; physical simulation keeps advancing; no Astra calls',
            behaviors=['continue', 'fixed_slow_0.15', 'latch_measured_joints', 'teacher_hold_anchor']+(['learned_maintenance'] if maintenance else []),
            completion_scope='physical orientation <=0.16 labels; semantic binding and autonomous stop still separate',
            drop_scope='frozen nominal-grasp displacement >=0.24m' if args.continuous_protocol else 'original environment object_out_of_reach diagnostic, not formal benchmark predicate',
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('collect_control.py','model.py','perception.py','goals.py')})
        manifest['agent_schedule_profile']=schedule_profile
        manifest['agent_schedule_profile_sha256']=sha(args.agent_schedule_profile) if schedule_profile else None
        manifest['maintenance_wait_only'] = args.maintenance_wait_only
        manifest['grounding_capture'] = args.grounding_capture
        save_json(args.output / 'manifest.json', manifest)
        grounding_datasets = {}
        grounding_manifest = None
        if args.grounding_capture:
            import h5py
            grounding_root = args.output / 'grounding'
            grounding_root.mkdir()
            grounding_store = h5py.File(grounding_root / 'samples.h5', 'w')
            hand = vision._hand_depth_camera
            grounding_manifest = dict(format_version=1,object='RSLCube',hand='Revo3',
                seed=args.seed,completed=False,split=args.split,
                camera_intrinsics=hand.data.intrinsic_matrices.torch[0].cpu().tolist(),
                camera_position_env=(hand.data.pos_w.torch[0]-raw.scene.env_origins[0]).cpu().tolist(),
                camera_quaternion_xyzw_ros=hand.data.quat_w_ros.torch[0].cpu().tolist(),
                object_keypoints=torch.cat((torch.zeros(1,3,device=device),vision.object_frame_pts[0])).cpu().tolist(),
                source_control_manifest_sha256=sha(args.output/'manifest.json'),
                capture='Measured-pose student trajectories with mixed wait behaviors and canonical skill changes.',
                label_scope='Object pose and hand depth are training labels only; RGB/instruction remain the only grounding inputs.')
            save_json(grounding_root/'manifest.json',grounding_manifest)
        frames, total, drops, resets = [], 0, 0, 0
        start = time.monotonic()
        def flush():
            nonlocal frames, total
            if not frames:
                return
            batch = {key:torch.cat([row[key] for row in frames]) for key in frames[0]}
            if grounding_store is not None:
                arrays = {key:batch[key].numpy() for key in ('env_id','episode_id','sim_step')}
                arrays.update(rgb=batch['rgb'].permute(0,2,3,1).numpy(),
                    object_pose=batch.pop('object_pose_for_labels').numpy(),
                    hand_depth=batch.pop('hand_depth_for_labels').numpy())
                for key,value in arrays.items():
                    if key not in grounding_datasets:
                        grounding_datasets[key]=grounding_store.create_dataset(key,shape=(0,*value.shape[1:]),
                            maxshape=(None,*value.shape[1:]),chunks=(1 if value.ndim>=3 else 256,*value.shape[1:]),
                            compression='lzf',dtype=value.dtype)
                    grounding_datasets[key].resize(total+len(value),axis=0)
                    grounding_datasets[key][total:total+len(value)]=value
                grounding_store.flush()
            torch.save(batch, args.output / f'samples_{total:07d}.pt')
            total += len(batch['goal_index'])
            frames = []
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    raise RuntimeError('Simulation interrupted')
                # Delivery is determined by scripted service timing, never by task success.
                delivered = pending & (elapsed >= duration)
                if delivered.any():
                    ids=delivered.nonzero().flatten()
                    if schedule_profile:
                        sid,cursor=schedule_id[ids],schedule_cursor[ids]
                        chosen=schedule_goal[sid,cursor]
                        binding_update=schedule_known[sid,cursor]
                        goal[ids[chosen>=0]]=chosen[chosen>=0]
                        known[ids[binding_update>=0]]=binding_update[binding_update>=0].bool()
                        until_request[ids]=schedule_execution[sid,cursor]
                        schedule_cursor[ids]+=1
                    else:
                        goal[delivered] = torch.randint(6, (int(delivered.sum()),), device=device)
                        known[delivered] = True
                        until_request[delivered] = torch.randint(90, 241, (int(delivered.sum()),), device=device)
                    pending[delivered] = False
                request = (~pending) & (until_request <= 0)
                if schedule_profile:
                    request &= schedule_cursor<schedule_length[schedule_id]
                pending[request] = True
                elapsed[request] = 0
                duration[request] = (schedule_duration[schedule_id[request],schedule_cursor[request]]
                    if schedule_profile else durations(int(request.sum())))
                need_anchor |= request
                command.quat_command_w[:] = rotations[goal]
                command.time_left.fill_(1e6)
                obs = control_obs()
                robot = raw.scene['robot']
                measure = estimator.predict_rectified(vision.masked_rgb.float()/255,
                    vision.camera.data.intrinsic_matrices.torch[0],
                    matrix_from_quat(vision.camera_quat_ros), vision.camera_pos,
                    matrix_from_quat(robot.data.root_quat_w.torch), robot.data.root_pos_w.torch)
                prop, ext, _ = adapter(obs, estimated_pose=(measure['position'], measure['quaternion']))
                if args.continuous_protocol:
                    prop[:,-1]=(1-raw.episode_length_buf.float()/600).clamp(0,1)
                proposal, reconstruction, next_hidden = student(prop[:, PROP_KEEP], ext[:, :7], vectors[goal], hidden)
                belief = student.core.belief(next_hidden[-1])
                current = prop[:, :21].clamp(-1, 1)
                anchor_pose[need_anchor] = ext[need_anchor, :7]
                anchor_rotation[need_anchor] = raw.scene['object'].data.root_quat_w.torch[need_anchor]
                anchor_action[need_anchor] = current[need_anchor]
                need_anchor[:] = False
                # Teacher-only anchor orientation never enters saved model features.
                command.quat_command_w[:] = anchor_rotation
                hold_obs = control_obs()
                hold_label = teacher(teacher_observations(hold_obs, prop, source['metadata']['teacher_quaternion_sign']))
                command.quat_command_w[:] = rotations[goal]
                executed = proposal.clone()
                selected = pending & (behavior == 1)
                executed[selected] = current[selected] + .15 * (proposal[selected].clamp(-1,1)-current[selected])
                selected = pending & (behavior == 2)
                executed[selected] = anchor_action[selected]
                selected = pending & (behavior == 3)
                executed[selected] = hold_label[selected]
                if maintenance is not None:
                    values=dict(proprio=prop[:,PROP_KEEP],measured_pose=ext[:,:7],belief=belief,proposal=proposal,
                        anchor_pose=anchor_pose,anchor_action=anchor_action,fit_error_m=measure['fit_error_m'],
                        known=known,pending=pending,elapsed_pending_s=torch.where(pending,elapsed.float()*raw.step_dt,0.))
                    maintained=maintenance(features(values),vectors[goal],proposal,current,pending)['action']
                    learned_selected=(behavior==4)&pending if args.maintenance_wait_only else behavior==4
                    executed[learned_selected]=maintained[learned_selected]
                save_frame = step % args.stride == 0
                if save_frame:
                    labels = []
                    for index in range(6):
                        command.quat_command_w[:] = rotations[index]
                        query_obs = control_obs()
                        labels.append(teacher(teacher_observations(query_obs, prop, source['metadata']['teacher_quaternion_sign'])))
                    command.quat_command_w[:] = rotations[goal]
                    actual = raw.scene['object'].data.root_quat_w.torch
                    normalized_actual = torch.nn.functional.normalize(actual.double(), dim=-1)
                    normalized_targets = torch.nn.functional.normalize(rotations.double(), dim=-1)
                    cosine = (normalized_actual[:, None]*normalized_targets[None]).sum(-1).abs().clamp(0,1)
                    errors = 2*torch.acos(cosine)
                    sample = dict(rgb=vision.masked_rgb.to(torch.uint8), proprio=prop[:, PROP_KEEP],
                        measured_pose=ext[:, :7], belief=belief, reconstruction=reconstruction,
                        proposal=proposal, anchor_pose=anchor_pose, anchor_action=anchor_action,
                        fit_error_m=measure['fit_error_m'], known=known, pending=pending,
                        elapsed_pending_s=torch.where(pending,elapsed.float()*raw.step_dt,0.), goal_index=goal,
                        behavior=behavior, env_id=torch.arange(n,device=device), episode_id=episode,
                        sim_step=torch.full((n,),step,device=device),episode_step=raw.episode_length_buf.clone(),angles_rad=errors.float(),
                        teacher_actions=torch.stack(labels,1), teacher_hold_action=hold_label,
                        executed_action=executed)
                    if schedule_profile:
                        sample.update(agent_schedule_id=schedule_id,agent_request_index=schedule_cursor)
                    if args.grounding_capture:
                        obj=raw.scene['object']
                        sample.update(object_pose_for_labels=torch.cat((obj.data.root_pos_w.torch-raw.scene.env_origins,
                            obj.data.root_quat_w.torch),dim=-1),hand_depth_for_labels=vision.hand_depth[:,0].half())
                    sample = {key:value.detach().cpu().clone() for key,value in sample.items()}
                obs, reward, done, _ = env.step(executed)
                done = done.bool()
                adapter.record_action(executed, done)
                hidden = next_hidden * (~done)[None,:,None]
                if save_frame:
                    sample['next_done'] = done.cpu().clone()
                    sample['next_drop'] = terminal['drop'].cpu().clone()
                    frames.append(sample)
                drops += int(terminal['drop'].sum())
                resets += int(done.sum())
                episode[done] += 1
                known[done] = False
                pending[done] = True
                elapsed[pending] += 1
                elapsed[done] = 0
                until_request[~pending] -= 1
                if schedule_profile:
                    schedule_id[done]=torch.randint(len(sequences),(int(done.sum()),),device=device)
                    schedule_cursor[done]=0
                    duration[done]=schedule_duration[schedule_id[done],schedule_cursor[done]]
                else:
                    duration[done] = durations(int(done.sum()))
                need_anchor |= done
                if (step+1) % 200 == 0:
                    flush()
                    progress=dict(step=step+1,target_steps=args.steps,samples=total,drops=drops,resets=resets,
                                  elapsed_seconds=time.monotonic()-start)
                    save_json(args.output/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
            flush()
        if grounding_manifest is not None:
            grounding_manifest.update(completed=True,samples=total,steps=args.steps)
            save_json(grounding_root/'manifest.json',grounding_manifest)
        save_json(args.output/'completion.json',dict(completed=True,samples=total,steps=args.steps,drops=drops,
                                                   resets=resets,elapsed_seconds=time.monotonic()-start))
    finally:
        if grounding_store is not None:
            grounding_store.close()
        if env is not None:
            env.close()

if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
