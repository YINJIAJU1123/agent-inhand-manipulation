"""Record full observation and execution trajectories for RGB supervision.

Uses the frozen supplied-color probe. No privileged state selects actions.
Pre-action physics and actions enable a separate checked rendering replay.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/BrainCo_DexHand')] + [str(ROOT / 'scripts' / name) for name in
    ('agentic', 'semantic_completion_v1', 'astra_completion_v3', 'astra_runtime_v2', 'async_stability_v1', 'astra_observation_control_v1')]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['none', 'execute', 'all'], required=True)
parser.add_argument('--max-steps', type=int, default=1800)
parser.add_argument('--stride', type=int, default=60)
parser.add_argument('--split', choices=['train', 'calibration', 'audit'], required=True)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args()
if a.stride < 2 or a.stride % 2:
    parser.error('--stride must be positive and even')
if a.max_steps < 900:
    parser.error('--max-steps must be at least 900')
a.enable_cameras = True
app = AppLauncher(dict(vars(a), multi_gpu=False)).app
import numpy as np
import torch
from common import load_student, environment, sensor_frame, history_append, STUDENT, ADAPTER, CONTROLLER, save, sha
from completion_runtime import DenseEvidenceRuntime
from hold import ObservationHold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_data import layout_bank, snapshot
from latency_controller_v1 import normalized_target
from placement import minimum_vertical_offsets
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch


def errors_for_colors(quaternion, colors):
    s = 2 ** -.5
    poses = torch.tensor([[s, 0, -s, 0], [s, 0, s, 0], [s, s, 0, 0], [s, -s, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], device=quaternion.device)
    return 2 * torch.acos((quaternion * poses[colors]).sum(-1).abs().clamp(0, 1))


out = a.output
out.mkdir(parents=True, exist_ok=False)
env = None
try:
    n = 6
    student, text, checkpoint = load_student(a.device)
    controller = RecoveryController(616).to(a.device).eval()
    controller.load_state_dict(torch.load(CONTROLLER, map_location=a.device, weights_only=True)['model'])
    controller.requires_grad_(False)
    env, raw, symbols = environment(a.seed, a.max_steps + 300, a.device, num_envs=n)
    rng = np.random.default_rng(a.seed + 115)
    bank = layout_bank(a.split)
    original_layout = bank[rng.choice(len(bank), n, replace=False)].tolist()
    symbols.set_layout(original_layout)
    raw.goal_markers.set_visibility(False)
    offsets, placement = minimum_vertical_offsets(raw)
    save(out / 'placement.json', placement)
    state = raw.object.data.root_state_w.clone()
    state[:, 2] += torch.tensor(offsets, device=a.device)
    raw.object.write_root_state_to_sim(state)
    raw.scene.write_data_to_sim(); raw.sim.forward(); raw.scene.update(0.)
    raw._compute_intermediate_values(); raw._update_face_markers()
    physical = lambda: {key: value.detach().cpu().clone() for key, value in dict(joint_pos=raw.hand.data.joint_pos,
        joint_vel=raw.hand.data.joint_vel, object_root=raw.object.data.root_state_w, prev_targets=raw.prev_targets).items()}
    torch.save(physical(), out / 'initial.pt')
    evidence_path = ROOT / 'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
    evidence = DenseEvidenceRuntime(evidence_path, a.device, n)
    retention = ObservationHold(n, a.mode, threshold=evidence.threshold)
    cold = normalized_target(raw).clone()
    runtime = ControllerRuntime(controller, cold, False)
    active = torch.ones(n, dtype=torch.bool, device=a.device)
    ih = ph = None
    rows = [dict(case=i, drop=False, drop_step=None, drop_state=None, pending_drop=False,
                 inspect_steps=0, inspect_ready_steps=0, execute_steps=0, execute_ready_steps=0, final_ready=False) for i in range(n)]
    log = []
    actions, states, sampled_rgb = [], {}, {}
    started = time.time()
    # Two observation colors then one execution color. Six-second simulated
    # requests begin 60 control steps after the previous reply, as below.
    # Each request lasts 180 steps: 0->180, 240->420, 480->660, ...
    requests = list(range(0, a.max_steps, 240))
    deliveries = [step + 180 for step in requests]
    save(out / 'protocol.json', dict(seed=a.seed, mode=a.mode, requests=requests, deliveries=deliveries,
        max_steps=a.max_steps, nominal_cases=n, cloud_calls=0,
        student_sha256=sha(STUDENT), adapter_sha256=sha(ADAPTER), controller_sha256=sha(CONTROLLER), evidence_sha256=sha(evidence_path),
        source_sha256={'rollout.py':sha(__file__), 'trajectory_data.py':sha(Path(__file__).with_name('trajectory_data.py')), 'frozen_probe':sha(ROOT/'scripts/astra_observation_control_v1/evaluate.py'), 'frozen_hold':sha(ROOT/'scripts/astra_observation_control_v1/hold.py')},
        split=a.split, stride=a.stride, original_layout=original_layout,
        policy='First delivered color = case index. Second observation = (case+2)%6. Third and later execution = case index.',
        scope='Supplied-color deterministic-delay integration only. No language, learned semantic binding, autonomous stop or real Astra outcome.',
        label_boundary='Pose errors and simulator drops are used only for scoring and episode termination. Fixed request timing and measured pose probabilities control retention.'))
    with torch.inference_mode():
        for step in range(a.max_steps):
            if step % 2 == 0:
                camera, rgb, features, compact, proprio = sensor_frame(raw, symbols)
                ih = history_append(ih, features, checkpoint['history'])
                ph = history_append(ph, proprio, checkpoint['history'])
                runtime.update_visual(compact)
            if step % 10 == 0:
                evidence.update(rgb, ih, ph, camera['depth'])
            delivered_count = sum(t <= step for t in deliveries)
            request_id = sum(t <= step for t in requests)
            pending_now = request_id > delivered_count
            delivered = [step in deliveries] * n
            commanded_state = 'initial' if delivered_count == 0 else 'inspect' if delivered_count <= 2 else 'execute'
            colors = [None] * n if delivered_count == 0 else [(i + 2) % 6 if delivered_count == 2 else i for i in range(n)]
            ids = torch.tensor([c if c is not None else 0 for c in colors], device=a.device)
            goals = text[ids]
            proposal = student(VisualStudentBatch(ih, goals, ph))['action']
            current = normalized_target(raw)
            known = torch.full((n,), delivered_count > 0, dtype=torch.bool, device=a.device)
            pending = active & pending_now
            age_s = (step - requests[request_id - 1]) / 30 if pending_now else 6. if step in deliveries else 0.
            age = torch.full((n,), age_s, device=a.device)
            action, _ = runtime.action(step, goals, proprio, proposal, current, pending, known, delivered, age)
            action = retention.apply(step=step, states=[commanded_state] * n, colors=colors, pending=pending,
                delivered=delivered, request_ids=[request_id] * n, active=active, ready_probability=evidence.last,
                feature_bank=None, fresh=step % 10 == 0, student_action=proposal, current=current, base_action=action)
            action[~known] = cold[~known]
            action[~active] = current[~active]
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite control command')
            runtime.commit(action)
            if step % a.stride == 0:
                states[str(step)] = dict(physics=snapshot(raw), active=active.cpu().clone(), phase=commanded_state, pending=pending_now, colors=colors)
                sampled_rgb[str(step)] = rgb.cpu().clone()
            actions.append(action.detach().cpu().clone())
            _, _, term, trunc, info = env.step(action)
            error = errors_for_colors(raw.object_rot, ids)
            for i in range(n):
                if not active[i]:
                    continue
                drop = bool(info['semantic_metrics']['dropped'][i])
                if commanded_state in ('inspect', 'execute'):
                    rows[i][commanded_state + '_steps'] += 1
                    rows[i][commanded_state + '_ready_steps'] += int(not drop and float(error[i]) <= .16)
                rows[i]['final_ready'] = commanded_state == 'execute' and not drop and float(error[i]) <= .16
                if drop:
                    rows[i].update(drop=True, drop_step=step + 1, drop_state=commanded_state, pending_drop=bool(pending[i]), final_ready=False)
                if drop or bool(term[i] or trunc[i]):
                    active[i] = False
            if step % 10 == 0:
                log.append(dict(step=step, state=commanded_state, colors=colors, pending=pending_now, request_id=request_id,
                                active=active.cpu().tolist(), probabilities=evidence.last.cpu().tolist(), error_for_scoring=error.cpu().tolist()))
            if step % 100 == 0:
                save(out / 'progress.json', dict(status='probing', step=step, active=int(active.sum()), retention_events=len(retention.events), elapsed_s=time.time() - started))
            if not active.any():
                break
    for row in rows:
        for phase in ('inspect', 'execute'):
            row[phase + '_ready_fraction_conditional_on_survival'] = row[phase + '_ready_steps'] / max(1, row[phase + '_steps'])
    result = dict(status='completed', seed=a.seed, mode=a.mode, cases=n, drops=sum(row['drop'] for row in rows),
        final_ready=sum(row['final_ready'] for row in rows),
        inspect_ready_fraction_fixed_horizon=sum(row['inspect_ready_steps'] for row in rows) / (n * 480),
        execute_ready_fraction_fixed_horizon=sum(row['execute_ready_steps'] for row in rows) / (n * (a.max_steps - 660)),
        events=len(retention.events), episodes=rows, cloud_calls=0,
        note='Fixed nominal denominators retain deaths as failures. Supplied-color integration, not semantic success. Compare all modes from identical initial physics.')
    torch.save(torch.stack(actions), out/'actions.pt')
    torch.save(states, out/'states.pt')
    torch.save(sampled_rgb, out/'sampled_rgb.pt')
    result.update(steps=len(actions), sampled_steps=len(states), split=a.split,
                  actions_sha256=sha(out/'actions.pt'), states_sha256=sha(out/'states.pt'), sampled_rgb_sha256=sha(out/'sampled_rgb.pt'))
    save(out / 'report.json', result)
    save(out / 'trace.json', log)
    save(out / 'retention_events.json', retention.events)
    save(out / 'progress.json', dict(status='completed', elapsed_s=time.time() - started))
    print('OBSERVATION_CONTROL_RESULT ' + json.dumps(result), flush=True)
except BaseException:
    save(out / 'failure.json', dict(error=traceback.format_exc()))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
