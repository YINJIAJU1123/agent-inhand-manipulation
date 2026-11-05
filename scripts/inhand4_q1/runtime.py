"""120-second dice runtime with noisy simulator pose and direct latent control.

Only the scorer reads true object orientation. Observations expose noisy sensor
history; images are rendered without labels. Formal agents use one environment.
"""
import hashlib,json,types
from pathlib import Path
import torch
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(1,str(ROOT/"scripts/dice_latent_v1"))
sys.path.insert(1,str(ROOT/"scripts/pose_latent_studies_v2/state_pair"))
from goals import load_bank
from model import physical_history
from feature_model import FeatureLatentActor,FORMAT

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def add_arguments(p):
    for name in ('source_checkpoint','checkpoint','env_config','bank','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--relocation',type=Path,required=True)
    p.add_argument('--assets_manifest',type=Path)
    p.add_argument('--layout_condition',choices=['identity','permuted'],default='identity')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--num_envs',type=int,default=1)

def benchmark_drop(env,threshold=.24):
    from viserdex.supermimic.simulation import tensor
    obj=env.scene['object'];ref=tensor(obj.data.default_root_state)[:,:3]+env.scene.env_origins
    return (tensor(obj.data.root_pos_w)-ref).norm(dim=-1)>=threshold

class DiceRuntime:
    def __init__(self,args):
        from asset_binding import install
        self.relocation=install(args.relocation)
        from viserdex.distillation.environment import SavedEnvironmentBinding,environment_contract
        from viserdex.distillation.teacher import configure_clean_state_observations
        from viserdex.utils.runtime import SimulationRuntime
        from viserdex.supermimic.simulation import TASK_ID,TransitionRecorderCfg
        from viserdex.supermimic.core import Config
        from benchmark_adapter import BenchmarkAdapter,BenchmarkHoldCommand
        from capture import configure_capture,Capture,EYE,LOOKAT,SIZE
        self.args=args;self.environment_contract=environment_contract
        self.payload=torch.load(args.source_checkpoint,map_location='cpu',weights_only=False)
        self.binding=SavedEnvironmentBinding(args.env_config)
        if self.payload['metadata']['environment_contract']!=self.binding.contract_sha:raise ValueError('Source environment mismatch')
        pack=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if pack['format']!=FORMAT or pack['metadata']['source_checkpoint_sha256']!=sha(args.source_checkpoint):raise ValueError('Latent checkpoint provenance mismatch')
        self.vectors,self.rotations,self.goal_manifest=load_bank(args.bank,'cpu')
        if pack['metadata']['goal_bank']['cache_sha256']!=self.goal_manifest['cache_sha256']:raise ValueError('Latent bank mismatch')
        self.student=FeatureLatentActor(Config(**pack['config']));self.student.load_state_dict(pack['model'],strict=True)
        self.student.eval().requires_grad_(False)
        def configure(cfg):
            self.binding.configure(cfg,args);configure_clean_state_observations(cfg)
            cfg.recorders=TransitionRecorderCfg();cfg.scene.clone_in_fabric=False
            cfg.scene.robot.spawn.activate_contact_sensors=False;cfg.scene.object.spawn.activate_contact_sensors=False
            cfg.commands.object_pose.debug_vis=False;cfg.seed=args.seed
            configure_capture(cfg)
            self.physical_binding=self.binding.verify(cfg)
            cfg.episode_length_s=120.
            cfg.terminations.max_consecutive_success=None
            cfg.terminations.object_out_of_reach.func=benchmark_drop
            cfg.terminations.object_out_of_reach.params={'threshold':.24}
            cfg.commands.object_pose.class_type=BenchmarkHoldCommand
            cfg.commands.object_pose.update_goal_on_success=False
            cfg.commands.object_pose.resampling_time_range=(1e6,1e6)
            self.verify(cfg)
            self.visual_override=None
            if args.assets_manifest:
                from pxr import Usd
                from build_marked_assets import records
                manifest=json.loads(args.assets_manifest.read_text())
                source=Path(cfg.scene.object.spawn.usd_path)
                entry=manifest['conditions'][args.layout_condition];asset=Path(entry['asset'])
                if source.resolve()!=Path(manifest['source']).resolve() or sha(source)!=manifest['source_sha256']:raise ValueError('Wrong source asset')
                if sha(asset)!=entry['sha256']:raise ValueError('Changed visual layer')
                for tex in entry['textures']:
                    if sha(tex['path'])!=tex['sha256']:raise ValueError('Changed glyph texture')
                if records(Usd.Stage.Open(str(source)),True)!=records(Usd.Stage.Open(str(asset)),True):raise ValueError('Nonvisual physics changed')
                cfg.scene.object.spawn.usd_path=str(asset)
                self.visual_override=dict(manifest_sha256=sha(args.assets_manifest),condition=args.layout_condition,
                    asset_sha256=sha(asset),scope='surface glyphs only, unchanged nonvisual properties')
        self.app=SimulationRuntime(args,configure=configure)
        import gymnasium as gym
        torch.manual_seed(args.seed);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        self.raw=gym.make(TASK_ID,cfg=self.app.cfg).unwrapped;self.device=self.raw.device
        self.student.to(self.device);self.vectors=self.vectors.to(self.device);self.rotations=self.rotations.to(self.device)
        self.command=self.raw.command_manager.get_term('object_pose');self.capture=Capture(self.raw)
        self.Adapter=BenchmarkAdapter
        if abs(self.raw.step_dt-.05)>1e-9 or self.raw.max_episode_length!=2400:raise ValueError('Unexpected benchmark clock')
        self.metadata=dict(format='inhand4_benchmark_runtime_v1',relocation=self.relocation,task_id=TASK_ID,visual_override=self.visual_override,source_checkpoint_sha256=sha(args.source_checkpoint),student_sha256=sha(args.checkpoint),
            physical_binding=self.physical_binding,goal_manifest=self.goal_manifest,pose_estimator_used=False,
            pose_source='simulator through released noise, bias, failure, jitter and delay model',noise_config=self.payload['noise_config'],
            explicit_overrides=dict(horizon_s=120,steps=2400,drop_distance_m=.24,success_termination=False,automatic_goal_resampling=False,
                student_clock='clip(1 - elapsed_steps/400,0,1)',command_subclass='honors disabled goal update; no physics override'),
            camera=dict(eye=EYE,lookat=LOOKAT,size=SIZE),inputs=['physical_history_10x116','continuous_latent_history_10x768'],
            source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('goals.py'),*(ROOT/'scripts/dice_latent_v1'/n for n in ['benchmark_adapter.py','feature_model.py','capture.py'])]})
        self.cache=None
    def verify(self,cfg):
        actual=self.environment_contract(cfg.to_dict())
        actual['episode_length_s']=self.binding.contract['episode_length_s']
        for name in ['max_consecutive_success','object_out_of_reach']:
            actual['terminations'][name]=self.binding.contract['terminations'][name]
        for name in ['class_type','update_goal_on_success','resampling_time_range']:
            actual['commands']['object_pose'][name]=self.binding.contract['commands']['object_pose'][name]
        if actual!=self.binding.contract:
            changes=[k for k in set(actual)|set(self.binding.contract) if actual.get(k)!=self.binding.contract.get(k)]
            raise ValueError('Unexpected physical changes: '+str(changes))
    def reset(self,seed):
        self.trace_trial_seed=int(seed)
        from viserdex.distillation.config import PerceptionNoiseCfg
        from viserdex.supermimic.simulation import tensor
        from viserdex.supermimic.core import quat_unique
        # The old recorder must not inspect a half-reset new episode.
        self.raw.supermimic_adapter=None
        self.raw.reset(seed=seed);self.steps=0;self.active_goal=1
        self.command.quat_command_w[:]=self.rotations[1];self.command.time_left.fill_(1e6)
        self.command.hold_ticks.zero_();self.command.just_completed.zero_()
        self.adapter=self.Adapter(self.raw,PerceptionNoiseCfg(**self.payload['noise_config']),10)
        self.latent=self.vectors[1:2].expand(self.raw.num_envs,-1).clone()
        self.latent_history=self.latent[:,None].expand(-1,10,-1).clone()
        self.cache=None;self.anchor_pose=None;self.anchor_action=None
        robot,obj=self.raw.scene['robot'],self.raw.scene['object'];action=self.raw.action_manager.get_term('joint_pos')
        values=dict(robot_root=tensor(robot.data.root_state_w),object_root=tensor(obj.data.root_state_w),joints=tensor(robot.data.joint_pos),
            joint_velocity=tensor(robot.data.joint_vel),ema=tensor(action._prev_applied_actions))
        record={k:v.detach().cpu().tolist() for k,v in values.items()}
        if bool(benchmark_drop(self.raw).any()):raise ValueError('Reset violates frozen drop rule')
        return dict(values=record,sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest())
    def set_skill(self,index,latent=None):
        from viserdex.supermimic.core import GOAL_ROT,HOLD,quat_unique,quat_mul
        if index not in range(6):raise ValueError('Bad skill')
        self.active_goal=index
        self.latent=self.vectors[index:index+1].expand(self.raw.num_envs,-1).clone() if latent is None else torch.as_tensor(latent,dtype=torch.float32,device=self.device).reshape(self.raw.num_envs,768)
        if not torch.isfinite(self.latent).all():raise ValueError('Nonfinite latent')
        self.command.quat_command_w[:]=self.rotations[index];self.command.time_left.fill_(1e6)
        self.command.hold_ticks.zero_();self.command.just_completed.zero_()
        self.adapter.state[:,GOAL_ROT]=quat_unique(quat_mul(self.adapter.base_inverse,self.command.quat_command_w))
        self.adapter.state[:,HOLD]=0
        # Current latent applies to the current measurement; older nine frames remain historical.
        self.latent_history[:,-1]=self.latent;self.cache=None
        expected=None if latent is None else torch.as_tensor(latent,dtype=torch.float32,device=self.device).reshape(self.raw.num_envs,768)
        if expected is not None:torch.testing.assert_close(self.latent,expected,atol=0,rtol=0)
        record=dict(trial_seed=self.trace_trial_seed,step=self.steps,selected_skill=index,
            continuous_latent_supplied=latent is not None,
            applied_latent_sha256=hashlib.sha256(self.latent.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            input_latent_sha256=None if expected is None else hashlib.sha256(expected.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        with (self.args.output/'action_interface.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    @torch.no_grad()
    def observe(self,include_rgb=False):
        from viserdex.supermimic.core import quat_mul,quat_unique
        if self.cache is None:
            physical=physical_history(self.adapter.history)
            pose=physical[:,-1,109:116]
            output=self.student(physical,self.latent_history,return_features=True)
            measured=pose[:,3:7]/pose[:,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6)
            goals=quat_unique(quat_mul(self.adapter.base_inverse[:,None].expand(-1,6,-1),self.rotations[None].expand(self.raw.num_envs,-1,-1)))
            errors=2*torch.acos((measured[:,None]*goals).sum(-1).abs().clamp(0,1))
            joint_normalized=physical[:,-1,:21]
            limits=self.adapter.limits
            measured_joints=(joint_normalized+1)*.5*(limits[...,1]-limits[...,0])+limits[...,0]
            current=((measured_joints-measured_joints.new_tensor(self.student.config.action_home))/.5).clamp(-1,1)
            if self.anchor_pose is None:self.anchor_pose=pose.clone();self.anchor_action=current.clone()
            self.cache=dict(physical=physical,proprio=physical[:,-1,:109],measured_pose=pose,belief=output['features'],proposal=output['action'],
                current=current,estimated_errors=errors,fit_error_m=torch.zeros(self.raw.num_envs,device=self.device))
        if include_rgb and 'rgb' not in self.cache:self.cache['rgb']=self.capture.read()['rgb']
        return self.cache
    def values(self,pending,known,elapsed_pending):
        s=self.observe();n=self.raw.num_envs
        return dict(s,anchor_pose=self.anchor_pose,anchor_action=self.anchor_action,
            pending=torch.full((n,),bool(pending),device=self.device),known=torch.full((n,),bool(known),device=self.device),
            elapsed_pending_s=torch.full((n,),float(elapsed_pending),device=self.device))
    @torch.no_grad()
    def step(self,action):
        import numpy as np
        from isaaclab_newton.physics import NewtonManager
        from viserdex.supermimic.core import POS,ROT,quat_mul,quat_apply,quat_conjugate
        from viserdex.supermimic.simulation import tensor
        old_goal=self.command.quat_command_w.clone()
        record,_,_,_=self.adapter.step(action)
        if int(np.bitwise_or.reduce(NewtonManager._solver.mjw_data.overflow.numpy().ravel())):raise RuntimeError('Physics solver capacity overflow')
        done=record['done'];p=record['next_physical']
        base_quat=quat_conjugate(self.adapter.base_inverse)
        world_q=quat_mul(base_quat,p[:,ROT]);world_p=quat_apply(base_quat,p[:,POS])+tensor(self.raw.scene['robot'].data.root_pos_w)
        ref=tensor(self.raw.scene['object'].data.default_root_state)[:,:3]+self.raw.scene.env_origins
        dropped=(world_p-ref).norm(dim=-1)>=.24
        if (~done).any():torch.testing.assert_close(self.command.quat_command_w[~done],old_goal[~done],atol=0,rtol=0)
        self.steps+=1
        expected_clock=(1-self.raw.episode_length_buf.float()/400).clamp(0,1)
        if (~done).any():torch.testing.assert_close(self.adapter.history[~done,-1,112],expected_clock[~done],atol=1e-7,rtol=0)
        self.latent_history=torch.cat((self.latent_history[:,1:],self.latent[:,None]),1)
        self.latent_history=torch.where(done[:,None,None],self.latent[:,None],self.latent_history)
        self.cache=None
        return dict(quaternion=world_q,position=world_p,drop=dropped,done=done,record=record)
    def close(self):
        self.raw.close();self.app.close()
