"""Render actual Newton visual assets at known orientations, with no goal overlays.

Diagnostic only: authored body poses are assigned without integrating physics.
These images are not formal evaluation trials or model-performance evidence.
"""
import pyglet
pyglet.options["headless"] = True
import argparse,json
from pathlib import Path
from viserdex.utils.runtime import SimulationRuntime,add_runtime_args,parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--env_config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=1);p.add_argument('--seed',type=int,default=186105)
add_runtime_args(p,TASK_ID);args=parse_runtime_args(p)
binding=SavedEnvironmentBinding(args.env_config)
def configure(cfg):
    binding.configure(cfg,args);configure_clean_state_observations(cfg)
    cfg.scene.clone_in_fabric=False;cfg.scene.robot.spawn.activate_contact_sensors=False;cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False;cfg.seed=args.seed
    v=NewtonGLVisualizerCfg(headless=True,window_width=512,window_height=512,
        eye=(.02,-.01,.92),lookat=(.02,-.01,.68),streaming_view=False,enable_picking=False,background_color=(.08,.11,.16))
    cfg.sim.visualizer_cfgs=[v];cfg.sim.default_visualizer_cfg=v
    binding.verify(cfg)
app=SimulationRuntime(args,configure=configure)
import gymnasium as gym
import torch
from PIL import Image,ImageDraw
from goals import QUATERNIONS_XYZW,NORMALS

def main():
    args.output.mkdir(parents=True,exist_ok=False)
    env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
    try:
        env.reset();obj=env.scene['object'];origin=env.scene.env_origins[0]
        viz=next(v for v in env.sim.visualizers if v.cfg.visualizer_type=='newton_gl')
        # Small off-axis camera avoids a singular look-at while showing the top.
        eye=origin+torch.tensor([.02,-.011,.92],device=env.device)
        look=origin+torch.tensor([.02,-.01,.68],device=env.device)
        env.sim.set_camera_view(tuple(eye.cpu().tolist()),tuple(look.cpu().tolist()))
        sheet=Image.new('RGB',(3*512,2*540),'white')
        for i,q in enumerate(QUATERNIONS_XYZW):
            pose=torch.tensor([[.02,-.01,.68,*q]],device=env.device);pose[:,:3]+=origin
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1,6,device=env.device))
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.step_dt)
            for _ in range(3):env.sim.render()
            im=Image.fromarray(viz.render_rgb_array()[...,:3]);im.save(args.output/f'expected_face_{i+1}.png')
            x=i%3*512;y=i//3*540;sheet.paste(im,(x,y+28));ImageDraw.Draw(sheet).text((x+200,y+6),f'Expected {i+1}',fill='black')
        sheet.save(args.output/'live_faces.jpg',quality=90)
        (args.output/'audit.json').write_text(json.dumps(dict(scope='Visual asset frame audit only, no physics rollout',normals=NORMALS,quaternions_xyzw=QUATERNIONS_XYZW,images_have_target_overlay=False),indent=2)+'\n')
    finally:env.close()
try:main()
finally:app.close()
