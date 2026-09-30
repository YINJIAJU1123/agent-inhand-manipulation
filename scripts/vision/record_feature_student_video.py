"""Record a qualitative RGB video from the closed-loop visual student."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--task', default='BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0')
parser.add_argument('--output', required=True)
parser.add_argument('--target-face', type=int, default=-1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--video-length', type=int, default=360)
parser.add_argument('--vision-stride', type=int, default=2)
parser.add_argument('--semantic-grid-size', type=int, default=16)
parser.add_argument('--semantic-depth-grid-size', type=int, default=16)
parser.add_argument('--action-scale', type=float, default=1.0)
parser.add_argument('--freeze-manifest', default='configs/visual_student_freeze.json')
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.target_face < -1 or args.target_face > 5:
    raise ValueError('target face must be -1 or in [0,5]')
args.enable_cameras = True
REPO_ROOT = Path(__file__).resolve().parents[2]
DEXHAND_SOURCE = REPO_ROOT / 'source' / 'BrainCo_DexHand'
sys.path.insert(0, str(DEXHAND_SOURCE))
sys.path.insert(0, str(REPO_ROOT))
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import BrainCo_DexHand  # noqa: F401,E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_feature_student import VisualLanguageStudent, VisualStudentBatch  # noqa: E402
from BrainCo_DexHand.algo.agentic.language_goal import FACE_NAMES  # noqa: E402


def semantic_features(rgb, depth, grid_size, depth_grid_size):
    rgb = rgb.to(torch.float32)
    if rgb.max() > 1.5: rgb = rgb / 255.0
    rgb = rgb[..., :3]
    n, h, w, _ = rgb.shape
    pixels = rgb.reshape(n, h*w, 3)
    colors = torch.tensor([[0.85,0.05,0.05],[0.05,0.75,0.15],[0.05,0.25,0.90],
                           [0.95,0.75,0.05],[0.80,0.05,0.75],[0.05,0.80,0.85]], device=rgb.device, dtype=rgb.dtype)
    weights = torch.exp(-(pixels[:,None]-colors[None,:,None]).square().mean(-1)/0.025)
    weights = weights * (pixels.mean(-1)[:,None] > 0.06)
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=rgb.device,dtype=rgb.dtype), torch.linspace(-1,1,w,device=rgb.device,dtype=rgb.dtype), indexing='ij')
    xx, yy = xx.reshape(1,1,-1), yy.reshape(1,1,-1)
    mass = weights.mean(-1); denom = weights.sum(-1).clamp_min(1e-6)
    cx = (weights*xx).sum(-1)/denom; cy = (weights*yy).sum(-1)/denom
    sx = torch.sqrt((weights*(xx-cx[...,None]).square()).sum(-1)/denom)
    sy = torch.sqrt((weights*(yy-cy[...,None]).square()).sum(-1)/denom)
    marker = torch.stack((mass,cx,cy,sx+sy),-1).reshape(n,-1)
    center = rgb[:,h//4:3*h//4,w//4:3*w//4]
    stats = torch.cat((rgb.mean((1,2)),rgb.std((1,2)),center.mean((1,2)),center.std((1,2))),-1)
    pieces=[stats]
    if grid_size:
        grid=torch.nn.functional.interpolate(rgb.permute(0,3,1,2),size=(grid_size,grid_size),mode='bilinear',align_corners=False)
        pieces.append(grid.reshape(n,-1))
    if depth_grid_size:
        d=torch.nan_to_num(depth.to(rgb.dtype),nan=2.0,posinf=2.0,neginf=0.0).clamp(0,2)/2
        if d.ndim==3:d=d.unsqueeze(-1)
        d=torch.nn.functional.interpolate(d.permute(0,3,1,2),size=(depth_grid_size,depth_grid_size),mode='bilinear',align_corners=False)
        pieces.append(d.reshape(n,-1))
    pieces.append(marker)
    return torch.cat(pieces,-1)


def main():
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg=parse_env_cfg(args.task, device=str(device), num_envs=1)
    cfg.max_consecutive_success=1; cfg.record_eval_metrics=True; cfg.seed=args.seed; cfg.goal_yaw=0.0
    if args.target_face >= 0: cfg.fixed_target_face=args.target_face
    env=gym.make(args.task,cfg=cfg); raw=env.unwrapped
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    policy=VisualLanguageStudent(ckpt['rgb_dim'],ckpt['language_dim'],ckpt['proprio_dim'],ckpt['action_dim'],memory_mode=ckpt.get('memory_mode','plain')).to(device)
    policy.load_state_dict(ckpt['model']); policy.eval()
    history=int(ckpt.get('history',8)); rgb_hist=torch.zeros((1,history,ckpt['rgb_dim']),device=device); prop_hist=torch.zeros((1,history,ckpt['proprio_dim']),device=device)
    initialized=False; outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    obs,_=env.reset(); frames=[]; trace=[]; face_names=['red','green','blue','yellow','magenta','cyan']; done=False
    with torch.no_grad():
        for step in range(args.video_length):
            camera=raw.capture_camera(); rgb=camera['rgb'][..., :3].to(torch.uint8)
            image=semantic_features(rgb,camera.get('depth'),args.semantic_grid_size,args.semantic_depth_grid_size)
            face=raw.target_face.long(); lang=torch.eye(6,device=device)[face]
            prop=raw.compute_student_proprio().float()
            if step % max(args.vision_stride,1)==0:
                rgb_hist=torch.cat((rgb_hist[:,1:],image[:,None]),1); prop_hist=torch.cat((prop_hist[:,1:],prop[:,None]),1)
                if not initialized:
                    rgb_hist[:]=image[:,None]; prop_hist[:]=prop[:,None]; initialized=True
            policy_out=policy(VisualStudentBatch(rgb_hist,lang,prop_hist)); action=policy_out['action']
            frame=rgb[0].detach().cpu().numpy(); Image.fromarray(frame).save(outdir/f'frame_{step:06d}.png')
            _,_,terminated,truncated,info=env.step(action)
            metrics=info['semantic_metrics']; done=bool((terminated|truncated)[0].item())
            trace.append({'step':step,'target_face':int(face[0]),'target_name':face_names[int(face[0])],'action':action[0].cpu().tolist(),'orientation_error_rad':float(metrics['orientation_error'][0]),'goal_reached':bool(metrics['goal_reached'][0]),'dropped':bool(metrics['dropped'][0]),'done':done})
            if done: break
    metadata={'checkpoint':args.checkpoint,'task':args.task,'target_face_override':None if args.target_face<0 else args.target_face,'seed':args.seed,'target_name':face_names[int(trace[0]['target_face'])] if trace else None,'frames':len(trace),'fps':30,'trace':trace}
    (outdir/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'output':str(outdir),'frames':len(trace),'done':done,'target':metadata['target_name']},indent=2))
    env.close()

try: main()
finally: app.close()
