"""Collect real source-policy RGB trajectories, with labels in separate fields."""
import pyglet
pyglet.options['headless']=True
import argparse,json,hashlib,time
from pathlib import Path
import torch
from viserdex.utils.runtime import SimulationRuntime,add_runtime_args,parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID,TransitionRecorderCfg
from capture import Capture,configure_capture,EYE,LOOKAT,SIZE
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','env_config','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--num_envs',type=int,default=8)
p.add_argument('--steps',type=int,default=1280);p.add_argument('--stride',type=int,default=5)
p.add_argument('--split',choices=('train','validation','smoke'),required=True)
add_runtime_args(p,TASK_ID);args=parse_runtime_args(p)
if args.output.exists():raise FileExistsError(args.output)
payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
binding=SavedEnvironmentBinding(args.env_config)
if payload['metadata']['environment_contract']!=binding.contract_sha:raise ValueError('Physical binding mismatch')
def configure(cfg):
    binding.configure(cfg,args);configure_clean_state_observations(cfg);cfg.recorders=TransitionRecorderCfg()
    cfg.scene.clone_in_fabric=False;cfg.scene.robot.spawn.activate_contact_sensors=False;cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False;cfg.seed=args.seed
    configure_capture(cfg);binding.verify(cfg)
app=SimulationRuntime(args,configure=configure)
import gymnasium as gym
import h5py,numpy as np
from PIL import Image,ImageDraw
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import Config
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.simulation import Adapter,tensor
from isaaclab_newton.physics import NewtonManager
from goals import QUATERNIONS_XYZW,NORMALS,install_training_goals

def main():
    args.output.mkdir(parents=True);torch.manual_seed(args.seed);torch.set_num_threads(8)
    source=Actor(Config(**payload['config']));source.load_state_dict(payload['actor'],strict=True)
    env=gym.make(TASK_ID,cfg=app.cfg).unwrapped;source.to(env.device).eval().requires_grad_(False)
    quats=torch.tensor(QUATERNIONS_XYZW,device=env.device)
    goals=install_training_goals(env.command_manager.get_term('object_pose'),quats)
    env.reset();adapter=Adapter(env,PerceptionNoiseCfg(**payload['noise_config']),10);capture=Capture(env)
    episodes=np.zeros(args.num_envs,dtype=np.int64);count=0;previews=[];started=time.monotonic()
    manifest=dict(format='dice_newton_rgb_v1',completed=False,split=args.split,seed=args.seed,
        num_envs=args.num_envs,stride=args.stride,requested_steps=args.steps,frame_size=SIZE,
        source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),environment_contract=binding.contract_sha,
        camera_eye_env=EYE,camera_lookat_env=LOOKAT,face_normals_body=NORMALS,pose_estimator_used=False,
        model_inputs='RGB and language only; states/depth in separate supervision fields',image_contains_goal_overlay=False)
    def save(): (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        with h5py.File(args.output/'samples.h5','w') as out,torch.no_grad():
            datasets={}
            def append(name,value):
                v=np.asarray(value)
                if name not in datasets:
                    kw=dict(compression='gzip',compression_opts=1) if v.ndim>1 else {}
                    datasets[name]=out.create_dataset(name,shape=(0,*v.shape),maxshape=(None,*v.shape),dtype=v.dtype,chunks=(1,*v.shape),**kw)
                ds=datasets[name];ds.resize(count+1,axis=0);ds[count]=v
            for step in range(args.steps):
                record,_,_,_=adapter.step(source(adapter.history));episodes+=record['done'].cpu().numpy()
                if int(np.bitwise_or.reduce(NewtonManager._solver.mjw_data.overflow.numpy().ravel())):raise RuntimeError('Capacity overflow')
                if (step+1)%args.stride:continue
                for env_id in range(args.num_envs):
                    frame=capture.read(env_id,depth=True)
                    pose=torch.cat((tensor(adapter.obj.data.root_pos_w)[env_id],tensor(adapter.obj.data.root_quat_w)[env_id])).cpu().numpy()
                    values=frame|dict(object_pose=pose,env_id=np.int32(env_id),episode_id=np.int32(episodes[env_id]),sim_step=np.int32(step+1),active_training_goal=np.int32(int(goals[env_id])))
                    for name,value in values.items():append(name,value)
                    if len(previews)<24 and (step+1)%(args.stride*8)==0:previews.append(frame['rgb'])
                    count+=1
                if (step+1)%100==0:print(json.dumps(dict(step=step+1,frames=count,elapsed=time.monotonic()-started)),flush=True)
            manifest.update(completed=True,frames=count,elapsed=time.monotonic()-started,max_errors=adapter.max_errors)
        if previews:
            sheet=Image.new('RGB',(4*256,((len(previews)+3)//4)*256),'white')
            for i,im in enumerate(previews):sheet.paste(Image.fromarray(im).resize((256,256)),(i%4*256,i//4*256))
            sheet.save(args.output/'preview.jpg',quality=90)
    finally:
        save();env.close()
    print(json.dumps(manifest),flush=True)
try:main()
finally:app.close()
