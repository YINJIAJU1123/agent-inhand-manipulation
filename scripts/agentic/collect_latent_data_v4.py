"""Collect NEW training scenes using frozen skills and depth-tested supervision."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/BrainCo_DexHand'))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--seeds', type=int, nargs='+', required=True)
p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
p.add_argument('--max-steps', type=int, default=300)
p.add_argument('--capture-stride', type=int, default=30)
p.add_argument('--checkpoint', default=str(ROOT / 'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'))
p.add_argument('--adapter', default=str(ROOT / 'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if len(a.seeds) != 1:
    raise ValueError('Use one seed per fresh simulator process; use run_latent_data_v4.py for multiple seeds')
a.enable_cameras = True
app = AppLauncher(a).app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
import BrainCo_DexHand
from isaaclab_tasks.utils import parse_env_cfg
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch
from latent_data_v4 import split_layouts, visibility_labels
from student_features import _semantic_rgb_features
from symbol_scene import SymbolScene, SYMBOLS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


try:
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'frames.jsonl').exists():
        raise RuntimeError('Refusing to overwrite a dataset')
    ckpt = torch.load(a.checkpoint, map_location='cuda:0', weights_only=False)
    canonical = torch.load(a.adapter, map_location='cuda:0', weights_only=False)['canonical_embeddings'].float()
    policy = VisualLanguageStudent(ckpt['rgb_dim'], ckpt['language_dim'], ckpt['proprio_dim'],
                                   ckpt['action_dim'], memory_mode=ckpt['memory_mode']).cuda().eval()
    policy.load_state_dict(ckpt['model'])
    cfg = parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0', device='cuda:0', num_envs=6)
    cfg.tiled_camera.width = cfg.tiled_camera.height = 512
    cfg.tiled_camera.data_types = ['rgb', 'depth']
    cfg.seed = a.seeds[0]
    cfg.max_consecutive_success = 0
    cfg.freeze_goal_for_episode = True
    cfg.record_eval_metrics = True
    cfg.goal_yaw = 0.
    cfg.episode_length_s = (a.max_steps + 30) / 30.
    env = gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0', cfg=cfg)
    raw = env.unwrapped
    symbols = SymbolScene(raw)
    rows = 0
    with (dest / 'frames.jsonl').open('w') as log:
        for seed in a.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.reset(seed=seed)
            raw._compute_intermediate_values()
            raw._update_face_markers()
            raw.goal_markers.set_visibility(False)
            layout = split_layouts(seed, a.split)
            symbols.set_layout(layout)
            rng = np.random.default_rng(seed + 7)
            schedule = rng.integers(0, 6, size=(a.max_steps // 60 + 1, 6))
            active = np.ones(6, dtype=bool)
            ih = torch.zeros(6, ckpt['history'], ckpt['rgb_dim'], device='cuda:0')
            ph = torch.zeros(6, ckpt['history'], ckpt['proprio_dim'], device='cuda:0')
            with torch.inference_mode():
                for step in range(a.max_steps):
                    if step % 2 == 0:
                        symbols.update()
                        camera = raw.capture_camera()
                        rgb = camera['rgb'][..., :3].to(torch.uint8)
                        srgb = torch.nn.functional.interpolate(rgb.permute(0, 3, 1, 2).float(), size=(256, 256), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        depth = camera['depth']
                        if depth.ndim == 3:
                            depth = depth.unsqueeze(-1)
                        depth = torch.nn.functional.interpolate(depth.permute(0, 3, 1, 2), size=(256, 256), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        feat = _semantic_rgb_features(srgb, 16, depth, 16)
                        prop = raw.compute_student_proprio().float()
                        ih = torch.cat([ih[:, 1:], feat[:, None]], 1)
                        ph = torch.cat([ph[:, 1:], prop[:, None]], 1)
                        if step == 0:
                            ih[:] = feat[:, None]
                            ph[:] = prop[:, None]
                    if step % a.capture_stride == 0:
                        labels = visibility_labels(raw, camera)
                        # Counterfactual render at the SAME physical state: remove only
                        # symbol geometry, then measure visible black/white symbol pixels.
                        # Neither this second image nor privileged boxes enter model inputs.
                        symbols.markers.set_visibility(False)
                        without = raw.capture_camera()['rgb'][..., :3].to(torch.uint8)
                        symbols.markers.set_visibility(True)
                        raw.capture_camera()  # flush renderer; no physics step
                        difference = (rgb.float() - without.float()).abs().mean(-1) > 20
                        achromatic = (rgb.max(-1).values - rgb.min(-1).values) < 45
                        ink = (rgb.max(-1).values < 80) | (rgb.min(-1).values > 190)
                        changed = difference & achromatic & ink
                        counts = [[0] * 6 for _ in range(6)]
                        labels['label_usable'] = [[False] * 6 for _ in range(6)]
                        fractions = [.315, .78, .3825, .4375, .4, .3825]
                        for case in range(6):
                            for color, points in enumerate(labels['projected_points'][case]):
                                xx, yy = zip(*points)
                                x0, x1 = max(0, int(min(xx)) - 2), min(512, int(max(xx)) + 3)
                                y0, y1 = max(0, int(min(yy)) - 2), min(512, int(max(yy)) + 3)
                                symbol = layout[case][color]
                                pixels = int(changed[case, y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
                                counts[case][symbol] = pixels
                                expected = labels['projected_area'][case][color] * fractions[symbol]
                                labels['readable'][case][color] = bool(labels['facing'][case][color] >= .5 and
                                    pixels >= max(35, .65 * expected))
                                labels['label_usable'][case][color] = bool(labels['readable'][case][color] or pixels <= 3 or labels['facing'][case][color] <= 0)
                        labels['symbol_pixels'] = counts
                        if step == 0:
                            for case in range(6):
                                Image.fromarray(without[case].cpu().numpy()).save(dest / f'probe_without_symbols_seed{seed}_case{case}.png')
                        for case in range(6):
                            if not active[case]:
                                continue
                            name = f'seed{seed}_case{case}_step{step:04d}'
                            Image.fromarray(rgb[case].cpu().numpy()).save(dest / (name + '.png'))
                            np.savez_compressed(dest / (name + '.npz'), rgb_history=ih[case].cpu().numpy(), proprio_history=ph[case].cpu().numpy())
                            row = dict(seed=seed, case=case, step=step, split=a.split, image=name + '.png', history=name + '.npz', layout=layout[case], symbols=list(SYMBOLS), **{k: v[case] for k, v in labels.items()})
                            log.write(json.dumps(row) + '\n')
                            rows += 1
                        log.flush()
                    ids = torch.tensor(schedule[step // 60], device='cuda:0')
                    action = policy(VisualStudentBatch(ih, canonical[ids], ph))['action']
                    action[torch.as_tensor(~active, device='cuda:0')] = 0
                    _, _, terminated, truncated, info = env.step(action)
                    active &= ~(terminated | truncated | info['semantic_metrics']['dropped']).cpu().numpy()
                    if not active.any():
                        break
            print(json.dumps({'seed': seed, 'frames_total': rows}), flush=True)
    manifest = dict(status='completed', split=a.split, seeds=a.seeds, frames=rows,
                    checkpoint_sha256=sha(a.checkpoint), adapter_sha256=sha(a.adapter),
                    frames_sha256=sha(dest / 'frames.jsonl'), max_steps=a.max_steps,
                    supervision='Same-state symbol-visible versus symbol-hidden render difference in projected symbol regions, with facing and pixel-count gates. Privileged labels only; counterfactual image/geometry never actor inputs.',
                    model_inputs=['RGB full frame', 'RGB-derived crop', 'instruction'],
                    source_sha256={x: sha(Path(__file__).parent / x) for x in ['collect_latent_data_v4.py', 'latent_data_v4.py']})
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env.close()
finally:
    app.close()
