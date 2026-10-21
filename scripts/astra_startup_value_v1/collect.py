"""Matched startup actions followed by an independently assigned color skill.

The startup policy never sees the future command or simulator geometry. Frozen
RGB-D features and candidate commands are saved before the scoring target is
generated. The resulting labels are local control outcomes, not Astra success.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/BrainCo_DexHand')] + [str(ROOT / 'scripts' / x) for x in (
    'agentic', 'semantic_completion_v1', 'astra_runtime_v2', 'async_stability_v1',
    'astra_retention_online_v1', 'astra_control_prefix_v1')]
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['record', 'latch', 'pose'] + [f'skill{i}' for i in range(6)], required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--wait-steps', type=int, choices=[180, 360], required=True)
p.add_argument('--source', type=Path)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if (a.mode == 'record') != (a.source is None):
    p.error('A branch requires a recorded initial snapshot; recording has no source.')
a.enable_cameras = True
app = AppLauncher(dict(vars(a), multi_gpu=False)).app

import numpy as np
import torch
from common import (load_student, environment, sensor_frame, history_append,
                    STUDENT, ADAPTER, CONTROLLER, save, sha)
from evidence_runtime import CachedDenseEvidence
from latency_controller_v1 import normalized_target
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from control_protocol import physical, assert_equal, pose_errors
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch

out = a.output
out.mkdir(parents=True, exist_ok=False)
env = None
try:
    n = 6
    student, text, pack = load_student(a.device)
    model = RecoveryController(616).to(a.device).eval()
    model.load_state_dict(torch.load(CONTROLLER, map_location=a.device, weights_only=True)['model'])
    model.requires_grad_(False)
    dense = ROOT / 'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
    evidence = CachedDenseEvidence(dense, a.device, n)
    max_steps = a.wait_steps + 1020
    env, raw, symbols = environment(a.seed, max_steps + 300, a.device)
    layout = split_layouts(a.seed, 'test')
    symbols.set_layout(layout)
    offsets, placement = minimum_vertical_offsets(raw)
    state = raw.object.data.root_state_w.clone()
    state[:, 2] += torch.tensor(offsets, device=a.device)
    raw.object.write_root_state_to_sim(state)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(0.)
    raw._compute_intermediate_values()
    raw._update_face_markers()
    save(out / 'placement.json', placement)
    cold = normalized_target(raw).clone()
    runtime = ControllerRuntime(model, cold, False)
    source_files = [Path(__file__), STUDENT, ADAPTER, CONTROLLER, dense] + [ROOT / 'scripts' / name for name in (
        'semantic_completion_v1/common.py', 'astra_runtime_v2/runtime.py', 'astra_learning_v1/runtime.py',
        'astra_retention_online_v1/evidence_runtime.py', 'astra_retention_online_v1/retention_runtime.py',
        'astra_completion_v3/completion_runtime.py', 'astra_control_prefix_v1/control_protocol.py',
        'agentic/recovery_controller_v3.py', 'agentic/recovery_controller_v2.py',
        'async_stability_v1/placement.py', 'async_stability_v1/geometry_v2.py')]
    frozen = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    save(out / 'protocol.json', dict(seed=a.seed, mode=a.mode, cases=n, wait_steps=a.wait_steps,
        max_steps=max_steps, source_sha256=frozen, cloud_calls=0,
        controls='Shared initial physical and measured sensor snapshot, frozen student/V3 and action scale.',
        startup='Fixed initial latch or one public orientation skill through original pending-request V3.',
        future_command='Independent seeded random color permutation, used only after the wait. Never a startup input.',
        scoring='Six consecutive full-quaternion errors <=0.16 rad after delivery, retaining startup failures.',
        scope='Development action-outcome labels; no language grounding, Astra calls or autonomous stop claims.'))

    if a.mode == 'record':
        before = physical(raw)
        camera, rgb, visual, compact, proprio = sensor_frame(raw, symbols)
        ih = history_append(None, visual, pack['history'])
        ph = history_append(None, proprio, pack['history'])
        with torch.inference_mode():
            evidence.update(rgb, ih, ph, camera['depth'])
            proposals = torch.stack([
                student(VisualStudentBatch(ih, text[c].expand(n, -1), ph))['action']
                for c in range(6)], dim=1)
            bank = evidence.retention_features
            base = cold[:, None].expand_as(proposals)
            skill_features = torch.cat((bank, proposals - base, base, bank.new_zeros(n, 6, 1)), dim=-1)
            hold_features = torch.cat((bank.mean(1), torch.zeros_like(cold), cold, bank.new_ones(n, 1)), dim=-1)
            features = torch.cat((skill_features, hold_features[:, None]), dim=1)
        if not torch.isfinite(features).all():
            raise RuntimeError('Nonfinite decision features')
        # Persist policy inputs before creating a future goal or scoring labels.
        torch.save(dict(features=features.cpu(), rgb=rgb.cpu(), ih=ih.cpu(), ph=ph.cpu(),
                        compact=compact.cpu(), proprio=proprio.cpu()), out / 'decision_inputs.pt')
        assert_equal(physical(raw), before, tolerance=0.)
        future_colors = np.random.default_rng(a.seed + 710000).permutation(6).tolist()
        bootstrap = dict(seed=a.seed, wait_steps=a.wait_steps, physical=before, cold=cold.cpu(),
            probability=evidence.last.cpu(), future_colors=future_colors, layout=layout,
            decision_inputs_sha256=sha(out / 'decision_inputs.pt'),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(bootstrap, out / 'bootstrap.pt')
        save(out / 'report.json', dict(status='completed', mode=a.mode, cases=n, seed=a.seed,
            bootstrap_sha256=sha(out / 'bootstrap.pt'), decision_inputs_sha256=sha(out / 'decision_inputs.pt'),
            feature_shape=list(features.shape), cloud_calls=0))
    else:
        source = json.loads((a.source / 'protocol.json').read_text())
        report = json.loads((a.source / 'report.json').read_text())
        if source['source_sha256'] != frozen or source['wait_steps'] != a.wait_steps:
            raise RuntimeError('Frozen source or wait protocol changed')
        if report['status'] != 'completed' or sha(a.source / 'bootstrap.pt') != report['bootstrap_sha256']:
            raise RuntimeError('Invalid initial snapshot')
        b = torch.load(a.source / 'bootstrap.pt', map_location='cpu', weights_only=True)
        if b['seed'] != a.seed or b['layout'] != layout or sha(a.source / 'decision_inputs.pt') != b['decision_inputs_sha256']:
            raise RuntimeError('Wrong or changed sensor snapshot')
        sensor = torch.load(a.source / 'decision_inputs.pt', map_location='cpu', weights_only=True)
        mismatch = assert_equal(physical(raw), b['physical'], tolerance=0.)
        assert_equal({'cold': cold.cpu()}, {'cold': b['cold']}, tolerance=0.)
        torch.set_rng_state(b['torch_rng'])
        torch.cuda.set_rng_state(b['cuda_rng'])
        ih, ph, compact, proprio = [sensor[k].to(a.device) for k in ('ih', 'ph', 'compact', 'proprio')]
        # Only measured probabilities or the assigned intervention choose the startup action.
        if a.mode.startswith('skill'):
            startup_ids = torch.full((n,), int(a.mode[-1]), device=a.device, dtype=torch.long)
        else:
            startup_ids = b['probability'].to(a.device).argmax(-1)
        save(out / 'initial_audit.json', dict(physical_errors=mismatch,
            decision_inputs_sha256=b['decision_inputs_sha256'], shared_sensor_state=True,
            startup_colors=startup_ids.cpu().tolist()))
        # A future command exists in the experiment, but is delivered only below.
        future_ids = torch.tensor(b['future_colors'], device=a.device)
        active = torch.ones(n, dtype=torch.bool, device=a.device)
        rows = [dict(case=i, drop=False, drop_step=None, completed=False, completion_step=None) for i in range(n)]
        dwell = torch.zeros(n, dtype=torch.long, device=a.device)
        traces, actions = [], []
        started = time.time()
        with torch.inference_mode():
            for step in range(max_steps):
                if step % 2 == 0:
                    if step:
                        camera, rgb, visual, compact, proprio = sensor_frame(raw, symbols)
                        ih = history_append(ih, visual, pack['history'])
                        ph = history_append(ph, proprio, pack['history'])
                    runtime.update_visual(compact)
                goals = text[startup_ids if step < a.wait_steps else future_ids]
                current = normalized_target(raw)
                proposal = student(VisualStudentBatch(ih, goals, ph))['action']
                pending = active & (step < a.wait_steps)
                age = torch.full((n,), step / 30 if step <= a.wait_steps else 0., device=a.device)
                action, diag = runtime.action(step, goals, proprio, proposal, current, pending,
                    torch.ones_like(active), [step == a.wait_steps] * n, age)
                if a.mode == 'latch' and step < a.wait_steps:
                    action = cold.clone()
                action[~active] = current[~active]
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite action')
                runtime.commit(action)
                actions.append(action.cpu().clone())
                _, _, term, trunc, info = env.step(action)
                newly = active & info['semantic_metrics']['dropped'].bool()
                for i in newly.nonzero().flatten().tolist():
                    rows[i].update(drop=True, drop_step=step + 1)
                # Ground truth is used only for outcome labels and evaluation termination.
                error = pose_errors(raw.object_rot, future_ids)
                reached = active & ~newly & (error <= .16) & (step >= a.wait_steps)
                dwell = torch.where(reached, dwell + 1, torch.zeros_like(dwell))
                completed = active & (dwell >= 6)
                for i in completed.nonzero().flatten().tolist():
                    rows[i].update(completed=True, completion_step=step + 1)
                active &= ~(newly | term | trunc | completed)
                if step < 60 or step % 10 == 0:
                    traces.append(dict(step=step, active=active.cpu().tolist(),
                        new_drops=newly.cpu().tolist(), score_error=error.cpu().tolist(),
                        score_distance=info['semantic_metrics']['object_distance'].cpu().tolist()))
                if step % 60 == 0:
                    save(out / 'progress.json', dict(status='running', step=step, active=int(active.sum()),
                        completed=sum(r['completed'] for r in rows), drops=sum(r['drop'] for r in rows),
                        wall_s=time.time() - started))
                if not active.any():
                    break
        torch.save(torch.stack(actions), out / 'actions.pt')
        save(out / 'trace.json', traces)
        result = dict(status='completed', seed=a.seed, mode=a.mode, cases=n, wait_steps=a.wait_steps,
            episodes=rows, future_colors=b['future_colors'], completed=sum(r['completed'] for r in rows),
            drops=sum(r['drop'] for r in rows),
            startup_drops=sum(r['drop'] and r['drop_step'] <= a.wait_steps for r in rows),
            post_handoff_drops=sum(r['drop'] and r['drop_step'] > a.wait_steps for r in rows),
            decision_inputs_sha256=b['decision_inputs_sha256'], cloud_calls=0,
            note='Matched local action consequences with independent future color. No Astra task-success estimate.')
        save(out / 'report.json', result)
        print('STARTUP_VALUE_RESULT ' + json.dumps(result), flush=True)
except BaseException:
    save(out / 'failure.json', dict(error=traceback.format_exc()))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
