"""Native renderer and physical-invariance diagnostic, not a task evaluation."""
import pyglet
pyglet.options['headless']=True
import argparse,hashlib,json,sys,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(1,str(ROOT/'scripts/dice_latent_v1'))
sys.path.insert(1,str(ROOT/'scripts/pose_latent_studies_v2/state_pair'))
from relocated_environment import install
pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--relocation',type=Path,required=True)
early,_=pre.parse_known_args();install(early.relocation)
from viserdex.utils.runtime import SimulationRuntime,add_runtime_args,parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID,tensor
from capture import Capture,configure_capture
from goals import QUATERNIONS_XYZW
from build_marked_assets import records as digest
from pxr import Usd

p=argparse.ArgumentParser(description=__doc__,parents=[pre])
p.add_argument('--env_config',type=Path,required=True)
p.add_argument('--assets_manifest',type=Path,required=True)
p.add_argument('--condition',choices=['native','identity','permuted'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=1)
p.add_argument('--seed',type=int,default=186511)
p.add_argument('--deterministic_diagnostic',action='store_true')
add_runtime_args(p,TASK_ID);a=parse_runtime_args(p)
if a.num_envs!=1:raise ValueError('Single-world diagnostic required')
a.output.mkdir(parents=True,exist_ok=False)
m=json.loads(a.assets_manifest.read_text());binding=SavedEnvironmentBinding(a.env_config)
source=Path(m['source'])
if str(source)!=binding.assets['object']['path']:raise ValueError('Wrong original asset')
asset=source if a.condition=='native' else Path(m['conditions'][a.condition]['asset'])
runtime_native_digest=digest(Usd.Stage.Open(str(source)),True)
if hashlib.sha256(source.read_bytes()).hexdigest()!=m['source_sha256']:raise ValueError('Original USD changed')
if digest(Usd.Stage.Open(str(asset)),True)!=runtime_native_digest:raise ValueError('Physical USD changed')
if a.condition!='native' and hashlib.sha256(asset.read_bytes()).hexdigest()!=m['conditions'][a.condition]['sha256']:
    raise ValueError('Visual layer changed')

def configure(cfg):
    binding.configure(cfg,a);configure_clean_state_observations(cfg)
    cfg.scene.clone_in_fabric=False
    cfg.scene.robot.spawn.activate_contact_sensors=False
    cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False;cfg.seed=a.seed
    configure_capture(cfg)
    binding.verify(cfg)
    if a.deterministic_diagnostic:
        if cfg.sim.physics.deterministic_mode!='not_guaranteed' or cfg.sim.physics.solver_cfg.disable_sensors is not False:
            raise ValueError('Unexpected original solver configuration')
        cfg.sim.physics.deterministic_mode='run_to_run'
        cfg.sim.physics.solver_cfg.disable_sensors=True
    # The only saved-config exception is this pre-audited visual layer.
    cfg.scene.object.spawn.usd_path=str(asset)

app=SimulationRuntime(a,configure=configure)
import gymnasium as gym
from isaaclab_newton.physics import NewtonManager

def state(env):
    return np.concatenate([tensor(env.scene['object'].data.root_state_w).cpu().numpy().ravel(),
        tensor(env.scene['robot'].data.joint_pos).cpu().numpy().ravel(),
        tensor(env.scene['robot'].data.joint_vel).cpu().numpy().ravel()])

try:
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(8)
    if a.deterministic_diagnostic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
    if a.deterministic_diagnostic:
        import warp as wp
        assert NewtonManager._solver._deterministic==wp.DeterministicMode.RUN_TO_RUN
        assert env.cfg.sim.physics.solver_cfg.disable_sensors
    try:
        env.reset(seed=a.seed);capture=Capture(env);initial=state(env)
        backend=NewtonManager.get_model()
        fields=['body_mass','body_inertia','body_com','shape_body','shape_flags','shape_material_mu','shape_material_restitution']
        np.savez(a.output/'backend_parameters.npz',**{name:getattr(backend,name).numpy() for name in fields})
        states=[initial];actions=[]
        for step in range(32):
            action=torch.zeros((1,21),device=env.device)
            env.step(action);states.append(state(env));actions.append(action.cpu().numpy())
            if int(np.bitwise_or.reduce(NewtonManager._solver.mjw_data.overflow.numpy().ravel())):
                raise RuntimeError('Solver overflow')
        np.savez(a.output/'physics_trace.npz',states=np.array(states),actions=np.array(actions))
        # Direct pose placement is only used for a six-face renderer audit.
        env.reset(seed=a.seed);obj=env.scene['object'];origin=env.scene.env_origins[0]
        viz=capture.visualizer
        eye=origin+torch.tensor([.02,-.011,.92],device=env.device)
        look=origin+torch.tensor([.02,-.01,.68],device=env.device)
        env.sim.set_camera_view(tuple(eye.cpu().tolist()),tuple(look.cpu().tolist()))
        sheet=Image.new('RGB',(3*512,2*540),'white')
        images=[]
        for i,q in enumerate(QUATERNIONS_XYZW):
            pose=torch.tensor([[.02,-.01,.68,*q]],device=env.device);pose[:,:3]+=origin
            obj.write_root_pose_to_sim(pose);obj.write_root_velocity_to_sim(torch.zeros(1,6,device=env.device))
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.step_dt)
            for _ in range(3):env.sim.render()
            rgb=viz.render_rgb_array()[...,:3].copy();im=Image.fromarray(rgb)
            im.save(a.output/f'skill_{i+1}.png');images.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            x=i%3*512;y=i//3*540;sheet.paste(im,(x,y+28))
            ImageDraw.Draw(sheet).text((x+12,y+7),f'{a.condition} / physical skill {i+1}',fill='black')
        sheet.save(a.output/'faces.jpg',quality=92)
        record={'status':'completed_diagnostic','condition':a.condition,'task_trials':0,
            'seed':a.seed,'object_key':os.environ['VISERDEX_OBJECT'],'steps':32,'deterministic_diagnostic':a.deterministic_diagnostic,
            'solver_settings':{'deterministic_mode':env.cfg.sim.physics.deterministic_mode,'disable_sensors':env.cfg.sim.physics.solver_cfg.disable_sensors},'initial_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),
            'outside_visual_sha256':runtime_native_digest,'static_outside_visual_sha256':m['nonvisual_sha256'],'image_hashes':images,
            'asset':str(asset),'mapping_for_audit_only':list(range(1,7)) if a.condition=='native' else m['conditions'][a.condition]['letters_by_physical_skill']}
        (a.output/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:env.close()
finally:app.close()
