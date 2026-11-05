"""Shared measured-sensor runtime for continuous 120-second RSLCube agents.

Only benchmark termination/horizon and visual capture differ from the released
physical task. Target quaternion stays outside the direct-latent student input.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import types

import torch
from viserdex.utils.runtime import SimulationRuntime
from goals import load_bank
from model import LatentStudent, PROP_KEEP
from perception import ReleasedPoseEstimator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(parser):
    for name in ('student_checkpoint','checkpoint','text_cache','original_asset',
                 'pose_checkpoint','camera_calibration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--num_envs',type=int,choices=[1],default=1)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--record_video',action='store_true')
    parser.add_argument('--video_stride',type=int,default=2)


def benchmark_drop(env, threshold=.24):
    obj = env.scene['object']
    # Same definition as the old benchmark: distance from the nominal grasp
    # position, in metres. The reference is scene-local, then moved to world.
    reference = obj.data.default_root_state.torch[:, :3] + env.scene.env_origins
    return (obj.data.root_pos_w.torch-reference).norm(dim=-1) >= threshold


class NewtonRuntime:
    def __init__(self,args):
        from viserdex.distillation.environment import SavedEnvironmentBinding,environment_contract
        from viserdex.distillation.trainer import load_student_checkpoint
        from viserdex.distillation.config import StudentCfg,PerceptionNoiseCfg
        self.args=args
        self.video=None
        self.video_camera=None
        if args.record_video:
            if args.video_stride<1:raise ValueError('Positive video stride required')
            args.enable_cameras=True
        self.source=load_student_checkpoint(args.student_checkpoint)
        self.binding=SavedEnvironmentBinding(args.student_checkpoint.parent/'env.yaml',
                                             self.source['metadata']['rollout_environment_sha256'])
        self.environment_contract=environment_contract
        self.original=StudentCfg(**self.source['student_cfg'])
        self.noise_cfg=PerceptionNoiseCfg(**self.source['noise_cfg'])
        if 'joint_delay_max_steps' not in self.source['noise_cfg']:
            self.noise_cfg.joint_delay_max_steps=0
        self.device=args.device or 'cuda:0'
        self.vectors,self.rotations,self.goal_manifest=load_bank(args.text_cache,args.original_asset,self.device)
        pack=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if (pack['format']!='pose_latent_student_v1' or pack['original_student_cfg']!=asdict(self.original)
            or pack['metadata']['source_student_sha256']!=sha(args.student_checkpoint)):
            raise ValueError('Student provenance mismatch')
        for key in ('cache_sha256','asset_sha256'):
            if pack['metadata']['goal_bank'][key]!=self.goal_manifest[key]:
                raise ValueError('Student goal bank mismatch')
        self.student=LatentStudent(self.original,pack['latent_dim'],pack['goal_width']).to(self.device).eval()
        self.student.load_state_dict(pack['model'])
        self.student.requires_grad_(False)
        self.app=SimulationRuntime(args,configure=self.configure)

        import gymnasium as gym
        import viserdex.tasks
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from viserdex.distillation.teacher import configure_clean_state_observations
        from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func
        from viserdex.renderer.hikrobot_occlusion import audit_original_capture
        cfg=self.app.env_cfg()
        cfg.seed=args.seed
        cfg.log_dir=str(args.output.resolve())
        cfg.scene.clone_in_fabric=False
        cfg.scene.robot.spawn.activate_contact_sensors=False
        cfg.scene.object.spawn.activate_contact_sensors=False
        cfg.commands.object_pose.debug_vis=False
        configure_clean_state_observations(cfg)
        self.verify(cfg)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        self.env=RslRlVecEnvWrapper(gym.make(args.task,cfg=cfg))
        self.raw=self.env.unwrapped
        if abs(self.raw.step_dt-1/30)>1e-9:raise ValueError('30 Hz required')
        self.command=self.raw.command_manager.get_term('object_pose')
        self.command.cfg.update_goal_on_success=False
        self.vision=_get_observation_term_func(self.raw.observation_manager,'pose','image_features')
        self.vision.capture_enabled=True
        self.estimator=ReleasedPoseEstimator(args.pose_checkpoint,self.vision.pts_path,self.device)
        camera=audit_original_capture(self.vision,self.vision._hand_depth_camera,self.raw.scene['robot'],args.camera_calibration)
        video_camera_audit=None
        if args.record_video:
            from viserdex.renderer.hikrobot_calibration import audit_calibrated_camera
            self.video_camera=self.raw.scene['agent_video_camera']
            self.video_camera.update(0.,force_recompute=True)
            video_camera_audit=audit_calibrated_camera(self.video_camera,self.raw.scene['robot'],args.camera_calibration)
        from collections import Counter
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton import GeoType,ShapeFlags
        physical=NewtonManager.get_model()
        collision=(physical.shape_flags.numpy()&int(ShapeFlags.COLLIDE_SHAPES))!=0
        geometry=dict(world_count=physical.world_count,body_count=physical.body_count,
            collision_shape_count=int(collision.sum()),
            collision_types=dict(Counter(GeoType(int(t)).name for t in physical.shape_type.numpy()[collision])))
        if geometry['world_count']!=1 or geometry['collision_types']!={'CONVEX_MESH':116,'BOX':1}:
            raise ValueError(f'Unexpected collision geometry: {geometry}')
        self.metadata=dict(student_sha256=sha(args.checkpoint),original_student_sha256=sha(args.student_checkpoint),
            physical_binding=self.physical_binding,pose_estimator=self.estimator.metadata,camera=camera,
            live_geometry=geometry,goal_manifest=self.goal_manifest,
            explicit_task_overrides=dict(horizon_seconds=120,maximum_steps=3600,success_counter_termination=None,
                drop='distance from nominal object grasp position >=0.24m',goal_resampling='disabled',
                student_time_feature='clip(1 - elapsed_control_steps/600,0,1), preserving the released 20s clock scale'),
            goal_input='direct 768D latent; no target or relative-goal quaternion in student',
            capture='30Hz calibrated object GS with live black hand occlusion, actual frozen CNN',
            qualitative_recording=args.record_video,video_camera=video_camera_audit,
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('newton_runtime.py','model.py','goals.py','perception.py')})
        self.terminal={}
        original_compute=self.raw.termination_manager.compute
        runtime=self
        def compute(manager):
            done=original_compute()
            runtime.terminal=dict(quaternion=runtime.raw.scene['object'].data.root_quat_w.torch.detach().clone(),
                position=runtime.raw.scene['object'].data.root_pos_w.torch.detach().clone(),
                drop=manager.get_term('object_out_of_reach').detach().clone(),
                done=done.detach().clone(),
                timeout=manager.get_term('time_out').detach().clone())
            return done
        self.raw.termination_manager.compute=types.MethodType(compute,self.raw.termination_manager)
        self.cache=None

    def configure(self,cfg):
        self.binding.configure(cfg,self.args)
        self.physical_binding=self.binding.verify(cfg)
        from viserdex.renderer.hikrobot_occlusion import configure_original_capture
        configure_original_capture(cfg,self.args.camera_calibration)
        if self.args.record_video:
            from viserdex.renderer.hikrobot_calibration import make_calibrated_camera
            initial=cfg.scene.robot.init_state
            cfg.scene.agent_video_camera=make_calibrated_camera(self.args.camera_calibration,initial.pos,initial.rot)
        cfg.observations.pose.image_features.params['debug_vis']=False
        cfg.episode_length_s=120.
        cfg.terminations.max_consecutive_success=None
        cfg.terminations.object_out_of_reach.func=benchmark_drop
        cfg.terminations.object_out_of_reach.params={'threshold':.24}
        self.verify(cfg)

    def verify(self,cfg):
        actual=self.environment_contract(cfg.to_dict())
        for section,names in (('scene',('base','gsplat_camera','hand_depth_camera')),
                              ('observations',('pose',)),
                              ('terminations',('max_consecutive_success','object_out_of_reach'))):
            for name in names:
                if name in self.binding.contract[section]:
                    actual[section][name]=self.binding.contract[section][name]
                else:actual[section].pop(name,None)
        if self.args.record_video:actual['scene'].pop('agent_video_camera',None)
        actual['episode_length_s']=self.binding.contract['episode_length_s']
        if actual!=self.binding.contract:
            changed=[key for key in set(actual)|set(self.binding.contract)
                     if actual.get(key)!=self.binding.contract.get(key)]
            raise ValueError(f'Unapproved physical configuration change: {changed}')

    def reset(self,seed):
        from viserdex.distillation.observations import StudentObservations
        self.raw.reset(seed=seed)
        self.steps=0
        self.active_goal=1
        self.latent=self.vectors[1:2].clone()
        self.command.quat_command_w[:]=self.rotations[1]
        self.command.time_left.fill_(1e6)
        self.adapter=StudentObservations(self.raw,self.original,self.noise_cfg)
        self.hidden=self.student.initial_state(1,self.device)
        self.cache=None
        self.terminal={}
        self.anchor_pose=None
        self.anchor_action=None
        robot,obj=self.raw.scene['robot'],self.raw.scene['object']
        action=self.raw.action_manager.get_term('joint_pos')
        torch.testing.assert_close(action._prev_applied_actions,robot.data.joint_pos.torch[:,action._joint_ids],atol=0,rtol=0)
        values=dict(robot_root=robot.data.root_state_w.torch,object_root=obj.data.root_state_w.torch,
            joints=robot.data.joint_pos.torch,joint_velocity=robot.data.joint_vel.torch,
            raw_action=self.raw.action_manager.action,previous_action=self.raw.action_manager.prev_action,
            ema_previous_action=action._prev_applied_actions,
            grasp_reference=obj.data.default_root_state.torch[:,:3]+self.raw.scene.env_origins)
        record={k:v.detach().cpu().tolist() for k,v in values.items()}
        if bool(benchmark_drop(self.raw)[0]):raise ValueError('Reset already violates frozen drop rule')
        return dict(values=record,sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest())

    def set_skill(self,index,latent=None):
        if index not in range(6):raise ValueError('Unknown public skill')
        self.active_goal=index
        self.latent=self.vectors[index:index+1].clone() if latent is None else torch.as_tensor(latent,device=self.device,dtype=torch.float32).reshape(1,768)
        if not torch.isfinite(self.latent).all():raise ValueError('Nonfinite goal vector')
        self.command.quat_command_w[:]=self.rotations[index]
        self.command.time_left.fill_(1e6)

    @torch.inference_mode()
    def observe(self):
        if self.cache is None:
            from tensordict import TensorDict
            from isaaclab.utils.math import matrix_from_quat
            from viserdex.distillation.noise import quat_mul
            for group in ('policy','privileged'):
                self.raw.obs_buf[group]=self.raw.observation_manager.compute_group(group)
            obs=TensorDict(self.raw.obs_buf,batch_size=[1])
            robot=self.raw.scene['robot']
            measure=self.estimator.predict_rectified(self.vision.masked_rgb.float()/255,
                self.vision.camera.data.intrinsic_matrices.torch[0],
                matrix_from_quat(self.vision.camera_quat_ros),self.vision.camera_pos,
                matrix_from_quat(robot.data.root_quat_w.torch),robot.data.root_pos_w.torch)
            prop,ext,_=self.adapter(obs,estimated_pose=(measure['position'],measure['quaternion']))
            prop=prop[:,PROP_KEEP]
            prop[:,-1]=max(0.,1-self.steps/600)
            world_quaternion=quat_mul(robot.data.root_quat_w.torch,measure['quaternion'])
            q=torch.nn.functional.normalize(world_quaternion.double(),dim=-1)
            target=torch.nn.functional.normalize(self.rotations.double(),dim=-1)
            error=2*torch.acos((q[:,None]*target[None]).sum(-1).abs().clamp(0,1))
            self.cache=dict(proprio=prop,measured_pose=ext[:,:7],fit_error_m=measure['fit_error_m'],
                rgb=self.vision.masked_rgb[0].permute(1,2,0).to(torch.uint8).cpu().numpy(),
                estimated_errors=error[0],world_quaternion=world_quaternion)
        return self.cache

    def begin_request(self):
        sensor=self.observe()
        self.anchor_pose=sensor['measured_pose'].clone()
        self.anchor_action=sensor['proprio'][:,:21].clamp(-1,1).clone()

    @torch.inference_mode()
    def propose(self,pending,known,elapsed_pending_s):
        sensor=self.observe()
        if self.anchor_pose is None:self.begin_request()
        proposal,_,next_hidden=self.student(sensor['proprio'],sensor['measured_pose'],self.latent,self.hidden)
        data=dict(proprio=sensor['proprio'],measured_pose=sensor['measured_pose'],
            belief=self.student.core.belief(next_hidden[-1]),proposal=proposal,
            anchor_pose=self.anchor_pose,anchor_action=self.anchor_action,fit_error_m=sensor['fit_error_m'],
            pending=torch.tensor([pending],device=self.device),known=torch.tensor([known],device=self.device),
            elapsed_pending_s=torch.tensor([elapsed_pending_s],device=self.device))
        return data,next_hidden

    def step(self,action,hidden):
        if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite action')
        self.terminal={}
        _,_,done,_=self.env.step(action)
        self.adapter.record_action(action,done.bool())
        self.hidden=hidden*(~done.bool())[None,:,None]
        self.steps+=1
        self.cache=None
        return self.terminal

    def start_video(self,directory,title):
        if self.args.record_video:
            from video import EpisodeVideo
            self.video=EpisodeVideo(directory,title,self.args.video_stride)

    def record_video(self,sensor,**context):
        if self.video is None or self.steps%self.args.video_stride:return
        self.video_camera.update(0.,force_recompute=True)
        native=self.video_camera.data.output['rgb'].torch[0,...,:3].to(torch.uint8).cpu().numpy()
        self.video.append(self.steps,native,sensor['rgb'],context)

    def finish_video(self,reason):
        if self.video is not None:
            self.video.close(reason,self.steps)
            self.video=None

    def close(self):
        self.finish_video('interrupted')
        self.env.close()
        self.app.close()
