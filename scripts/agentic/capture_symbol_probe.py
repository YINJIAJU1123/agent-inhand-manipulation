import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/BrainCo_DexHand'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--resolution',type=int,default=512)
AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.enable_cameras=True
app=AppLauncher(a).app
import torch, gymnasium as gym
from PIL import Image
import BrainCo_DexHand
from isaaclab_tasks.utils import parse_env_cfg
from symbol_scene import SymbolScene
try:
    cfg=parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',device='cuda:0',num_envs=6)
    cfg.tiled_camera.width=a.resolution; cfg.tiled_camera.height=a.resolution
    cfg.seed=731; cfg.max_consecutive_success=1; cfg.goal_yaw=0.0
    env=gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',cfg=cfg)
    raw=env.unwrapped; env.reset()
    symbols=SymbolScene(raw)
    symbols.set_layout(torch.stack([torch.arange(6).roll(i) for i in range(6)]))
    symbols.update(); images=raw.capture_camera()['rgb'][...,:3].cpu().numpy()
    dest=Path(a.output); dest.mkdir(parents=True,exist_ok=True)
    for i,im in enumerate(images): Image.fromarray(im).save(dest/f'probe_{i}.png')
    print(json.dumps({'output':str(dest),'layout':symbols.layout.tolist()}),flush=True)
    env.close()
finally: app.close()
