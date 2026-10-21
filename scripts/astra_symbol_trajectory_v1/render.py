"""Replay recorded actions and supervise new symbols at verified physical states.

Rendering variants are training augmentation. They do not run a policy or imply
real-time closed-loop performance. All privileged values remain labels/audits.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/BrainCo_DexHand')] + [str(ROOT/'scripts'/name) for name in
    ('agentic', 'semantic_completion_v1', 'astra_completion_v3', 'async_stability_v1')]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--views', type=int, default=6)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.views < 1:
    p.error('--views must be positive')
a.enable_cameras = True
app = AppLauncher(dict(vars(a), multi_gpu=False)).app
import numpy as np
import torch
from PIL import Image, ImageDraw
from common import environment, save, sha, COLORS, SYMBOLS
from placement import minimum_vertical_offsets
from latent_data_v4 import visibility_labels
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_data import layout_bank, snapshot, compare, projected_boxes
import importlib.util
label_spec = importlib.util.spec_from_file_location('trajectory_symbol_labels', ROOT/'scripts/astra_symbol_regions_v1/labels.py')
label_module = importlib.util.module_from_spec(label_spec)
label_spec.loader.exec_module(label_module)
readable = label_module.readable

out = a.output
out.mkdir(parents=True, exist_ok=False)
env = None
try:
    protocol = json.loads((a.source/'protocol.json').read_text())
    report = json.loads((a.source/'report.json').read_text())
    for name in ['actions', 'states', 'sampled_rgb']:
        if sha(a.source/(name+'.pt')) != report[name+'_sha256']:
            raise ValueError('Trajectory artifact checksum mismatch: ' + name)
    actions = torch.load(a.source/'actions.pt', weights_only=True, map_location=a.device)
    states = torch.load(a.source/'states.pt', weights_only=True, map_location='cpu')
    originals = torch.load(a.source/'sampled_rgb.pt', weights_only=True, map_location='cpu')
    env, raw, symbols = environment(protocol['seed'], protocol['max_steps']+300, a.device)
    original_layout = protocol['original_layout']
    symbols.set_layout(original_layout)
    offsets, placement = minimum_vertical_offsets(raw)
    root = raw.object.data.root_state_w.clone()
    root[:, 2] += torch.tensor(offsets, device=a.device)
    raw.object.write_root_state_to_sim(root)
    raw.scene.write_data_to_sim(); raw.sim.forward(); raw.scene.update(0.)
    raw._compute_intermediate_values(); raw._update_face_markers()
    compare(snapshot(raw), torch.load(a.source/'initial.pt', weights_only=True, map_location='cpu'))
    bank = layout_bank(protocol['split'])
    bank_set = {tuple(row) for row in bank.tolist()}
    assert all(tuple(row) in bank_set for row in original_layout)
    rng = np.random.default_rng(protocol['seed']+9017)
    images, boxes, labels, rows, audits, previews = [], [], [], [], [], {}
    phase_counts, readable_counts = Counter(), Counter()
    source_render_difference = []
    start = time.time()

    def frame():
        # Allow a rendered layout change to settle before using its labels.
        raw.capture_camera(refresh=True)
        return raw.capture_camera(refresh=True)

    with torch.inference_mode():
        for step, action in enumerate(actions):
            if str(step) in states:
                sample = states[str(step)]
                frozen = snapshot(raw)
                errors = compare(frozen, sample['physics'])
                symbols.set_layout(original_layout); symbols.update()
                original = frame()['rgb'][..., :3].to(torch.uint8)
                source_render_difference.append(dict(step=step,
                    rgb_mean_absolute_difference=float((original.cpu().float()-originals[str(step)].float()).abs().mean())))
                symbols.markers.set_visibility(False)
                without = frame()['rgb'][..., :3].to(torch.uint8)
                symbols.markers.set_visibility(True)
                coverage = np.zeros((6, 6, 6), dtype=np.int32)
                for view in range(a.views):
                    layout = []
                    for i in range(6):
                        score = coverage[i, np.arange(6)[None], bank].sum(-1)
                        choices = np.flatnonzero(score == score.min())
                        choice = np.asarray(original_layout[i]) if view == 0 else bank[rng.choice(choices)]
                        layout.append(choice.tolist())
                        coverage[i, np.arange(6), choice] += 1
                    symbols.set_layout(layout); symbols.update()
                    camera = frame()
                    rgb = camera['rgb'][..., :3].to(torch.uint8)
                    valid, usable = readable(raw, camera, rgb, without, layout)
                    projection = visibility_labels(raw, camera)
                    target_boxes = projected_boxes(projection['projected_points'])
                    compare(snapshot(raw), frozen, tolerance=0.)
                    for i in range(6):
                        if not sample['active'][i]:
                            continue
                        lab = torch.where(valid[i], torch.tensor(layout[i]), torch.where(usable[i], 6, -100))
                        images.append(rgb[i].permute(2, 0, 1).cpu().clone())
                        boxes.append(target_boxes[i]); labels.append(lab)
                        rows.append(dict(seed=protocol['seed'], split=protocol['split'], step=step,
                            phase=sample['phase'], pending=sample['pending'], case=i, view=view,
                            source_mode=protocol['mode']))
                        phase_counts[sample['phase']] += 1
                        readable_counts[sample['phase']] += int(valid[i].sum())
                        # Show original symbols across phases and times, rather
                        # than selecting only clear or successful examples.
                        key = (sample['phase'], step//300)
                        if view == 0 and key not in previews:
                            im = Image.fromarray(rgb[i].cpu().numpy())
                            draw = ImageDraw.Draw(im)
                            draw.text((6, 6), f'{sample["phase"]} step {step} case {i}', fill='white', stroke_width=1, stroke_fill='black')
                            for c in range(6):
                                if valid[i, c]:
                                    x, y, w, h = (target_boxes[i, c]*512).tolist()
                                    draw.rectangle((x-w/2, y-h/2, x+w/2, y+h/2), outline='white', width=2)
                                    draw.text((max(0,x-w/2), max(22,y-h/2-14)), COLORS[c]+' '+SYMBOLS[int(lab[c])], fill='white', stroke_width=1, stroke_fill='black')
                            previews[key] = im
                symbols.set_layout(original_layout); symbols.update(); frame()
                compare(snapshot(raw), frozen, tolerance=0.)
                audits.append(dict(step=step, max_error=max(errors.values()), active=int(sample['active'].sum())))
                save(out/'progress.json', dict(status='rendering', step=step, views=len(rows), phase_counts=dict(phase_counts), wall_s=time.time()-start))
            env.step(action)
    data = dict(rgb=torch.stack(images), boxes=torch.stack(boxes), labels=torch.stack(labels), rows=rows)
    assert data['rgb'].dtype == torch.uint8 and data['rgb'].shape[1:] == (3, 512, 512)
    assert torch.isfinite(data['boxes']).all() and ((data['boxes'] >= 0) & (data['boxes'] <= 1)).all()
    torch.save(data, out/'data.pt')
    save(out/'replay_audit.json', dict(physical_states=audits, source_render_difference=source_render_difference))
    count = len(previews)
    canvas = Image.new('RGB', (1024, ((count+3)//4)*256))
    for i, im in enumerate(previews.values()):
        canvas.paste(im.resize((256, 256)), ((i%4)*256, (i//4)*256))
    canvas.save(out/'label_preview.jpg', quality=90)
    known = (data['labels'] >= 0) & (data['labels'] < 6)
    result = dict(status='completed', seed=protocol['seed'], split=protocol['split'], views=len(rows),
        phase_counts=dict(phase_counts), readable_by_phase=dict(readable_counts), readable=int(known.sum()),
        label_counts=torch.bincount(data['labels'][data['labels'] >= 0], minlength=7).tolist(),
        ambiguous=int((data['labels'] < 0).sum()), per_color_readable=known.sum(0).tolist(),
        sampled_states=len(audits), replay_max_error=max(row['max_error'] for row in audits),
        data_sha256=sha(out/'data.pt'), source_actions_sha256=report['actions_sha256'],
        source_states_sha256=report['states_sha256'], source_sha256=sha(__file__),
        label_source_sha256=sha(ROOT/'scripts/astra_symbol_regions_v1/labels.py'),
        preview_sha256=sha(out/'label_preview.jpg'), cloud_calls=0,
        scope='New supplied-color trajectory seeds. RGB-only training input. Readability is a renderer proxy, not human annotation. Exact physics replay checks do not imply identical fresh RGB. No Astra or task-success evidence.')
    save(out/'manifest.json', result)
    save(out/'progress.json', dict(status='completed', views=len(rows), wall_s=time.time()-start))
    print('TRAJECTORY_DATA_RESULT '+json.dumps(result), flush=True)
except BaseException:
    save(out/'failure.json', dict(error=traceback.format_exc()))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
