"""Public sensor preprocessing and frozen assets for completion experiments.

No pose, layout or evaluator result is accepted by these preprocessing functions.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
STUDENT = ROOT / "outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt"
ADAPTER = ROOT / "outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt"
CONTROLLER = ROOT / "outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt"
PYTHON = "/home/yin/Grasping/Grasppose/env_isaaclab/bin/python"
COLORS = ("red", "green", "blue", "yellow", "magenta", "cyan")
SYMBOLS = ("star", "circle", "triangle", "cross", "diamond", "bars")
ARMS = ("baseline", "verify", "observe", "joint")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def color_patches(rgb, size=48):
    """RGB-only bounding boxes, including absent/occluded colors.

    Identical preprocessing for training and inference. No projected GT regions.
    """
    device = rgb.device
    image = rgb.float() / 255.
    n, h, w, _ = image.shape
    prototypes = torch.tensor(
        [[.85,.05,.05],[.05,.75,.15],[.05,.25,.90],
         [.95,.75,.05],[.80,.05,.75],[.05,.80,.85]], device=device)
    distance = (image[:,None] - prototypes[None,:,None,None]).square().mean(-1)
    masks = (distance < .035) & ((image.max(-1).values-image.min(-1).values)[:,None] > .3)
    patches = torch.zeros(n,6,3,size,size,device=device)
    geometry = torch.zeros(n,6,5,device=device)
    for i in range(n):
        for c in range(6):
            yy, xx = masks[i,c].nonzero(as_tuple=True)
            if len(xx) < 8:
                continue
            # Color regions may include a large face background. This remains
            # an observation heuristic, not a symbol detector or supervision.
            x0, x1 = max(0,int(xx.min())-5), min(w,int(xx.max())+6)
            y0, y1 = max(0,int(yy.min())-5), min(h,int(yy.max())+6)
            patch = image[i,y0:y1,x0:x1].permute(2,0,1)[None]
            patches[i,c] = F.interpolate(patch,(size,size),mode="bilinear",align_corners=False)[0]
            geometry[i,c] = torch.tensor(
                [len(xx)/(h*w), (x0+x1)/(2*w), (y0+y1)/(2*h),
                 (x1-x0)/w, (y1-y0)/h], device=device)
    return patches, geometry


def sensor_frame(raw, symbols):
    from student_features import _semantic_rgb_features
    symbols.update()
    camera = raw.capture_camera(refresh=True)
    rgb = camera["rgb"][...,:3].to(torch.uint8)
    srgb = F.interpolate(rgb.permute(0,3,1,2).float(),(256,256),
                         mode="bilinear",align_corners=False).permute(0,2,3,1)
    depth = camera["depth"]
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    depth = F.interpolate(depth.permute(0,3,1,2),(256,256),
                          mode="bilinear",align_corners=False).permute(0,2,3,1)
    features = _semantic_rgb_features(srgb,16,depth,16)
    compact = _semantic_rgb_features(srgb,4,depth,4)
    return camera, rgb, features, compact, raw.compute_student_proprio().float()


def history_append(history, frame, length=4):
    return (frame[:,None].repeat(1,length,1) if history is None else
            torch.cat((history[:,1:],frame[:,None]),1))


def environment(seed, max_steps, device, num_envs=6):
    import gymnasium as gym
    import BrainCo_DexHand
    from isaaclab_tasks.utils import parse_env_cfg
    from symbol_scene import SymbolScene
    cfg = parse_env_cfg("BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0",
                        device=device, num_envs=num_envs)
    cfg.tiled_camera.width = cfg.tiled_camera.height = 512
    cfg.seed = seed
    cfg.max_consecutive_success = 0
    cfg.freeze_goal_for_episode = True
    cfg.record_eval_metrics = True
    cfg.goal_yaw = 0.
    cfg.episode_length_s = (max_steps+30)/30.
    env = gym.make("BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0",cfg=cfg)
    raw = env.unwrapped
    raw.cfg.sim.render_interval = 1000000
    symbols = SymbolScene(raw)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    raw.goal_markers.set_visibility(False)
    raw._compute_intermediate_values()
    raw._update_face_markers()
    return env, raw, symbols


def load_student(device):
    from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent
    pack = torch.load(STUDENT,map_location=device,weights_only=True)
    contract = json.loads((ROOT/"configs/visual_student_freeze.json").read_text())
    assert pack["freeze_id"] == contract["freeze_id"] and pack["action_scale"] == 1.
    student = VisualLanguageStudent(pack["rgb_dim"],pack["language_dim"],
                                   pack["proprio_dim"],pack["action_dim"],
                                   memory_mode=pack["memory_mode"]).to(device).eval()
    student.load_state_dict(pack["model"])
    student.requires_grad_(False)
    text = torch.load(ADAPTER,map_location=device,weights_only=True)["canonical_embeddings"].float()
    return student, text, pack
