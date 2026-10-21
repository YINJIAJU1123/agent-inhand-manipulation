"""Continuous control from one shared action prefix and measured sensor state.

A public, fixed request schedule replaces the remote agent. Its observation and
execution colors never depend on hidden geometry or outcomes. The first pending
inspection state with any measured pose probability >= threshold is the common
branch point for the whole six-case batch. All cases, including prefix drops,
remain in the denominator. No Astra requests or semantic success claims.
"""
import argparse, copy, json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/BrainCo_DexHand')] + [str(ROOT/'scripts'/s) for s in
    ('agentic','semantic_completion_v1','astra_runtime_v2','async_stability_v1','astra_retention_online_v1')]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['record','none','hold','learned'], required=True)
p.add_argument('--seed', type=int)
p.add_argument('--source', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--max-steps', type=int, default=1800)
p.add_argument('--retention-head', type=Path, default=ROOT/'outputs/astra_retention_v1/learning_20261022/train/head.pt')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.mode == 'record' and (a.seed is None or a.source is not None):
    p.error('Recording needs a seed and no source')
if a.mode != 'record' and a.source is None:
    p.error('A shared recorded source is required')
if a.max_steps != 1800:
    p.error('This frozen diagnostic uses 1800 steps')
a.enable_cameras = True
app = AppLauncher(dict(vars(a), multi_gpu=False)).app
import numpy as np
import torch
from common import load_student, environment, sensor_frame, history_append, STUDENT, ADAPTER, CONTROLLER, save, sha
from evidence_runtime import CachedDenseEvidence
from retention_runtime import RetentionRuntime, load_value_head
from latency_controller_v1 import normalized_target
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_protocol import command_state, physical, assert_equal, pose_errors

out = a.output
out.mkdir(parents=True, exist_ok=False)
env = None
try:
    n = 6
    recording = a.mode == 'record'
    source_meta = None
    if not recording:
        source_meta = json.loads((a.source/'protocol.json').read_text())
        source_report = json.loads((a.source/'report.json').read_text())
        assert source_report['status'] == 'completed' and source_report['branch_step'] is not None
        for name in ['actions.pt', 'bootstrap.pt', 'initial.pt']:
            if sha(a.source/name) != source_report['artifact_sha256'][name]:
                raise RuntimeError('Source checksum differs: '+name)
        a.seed = source_meta['seed']
        bootstrap = torch.load(a.source/'bootstrap.pt', map_location='cpu', weights_only=True)
        source_actions = torch.load(a.source/'actions.pt', map_location=a.device, weights_only=True)
        branch_step = bootstrap['step']
    else:
        bootstrap = None
        branch_step = None
    student, text, pack = load_student(a.device)
    controller = RecoveryController(616).to(a.device).eval()
    controller.load_state_dict(torch.load(CONTROLLER, map_location=a.device, weights_only=True)['model'])
    controller.requires_grad_(False)
    evidence_path = ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
    evidence = CachedDenseEvidence(evidence_path, a.device, n)
    head = load_value_head(a.retention_head, evidence_path, a.device, require_gate=True) if a.mode == 'learned' else None
    retention = RetentionRuntime(n, 'none' if recording else a.mode, head, threshold=evidence.threshold)
    env, raw, symbols = environment(a.seed, a.max_steps+300, a.device)
    layout = split_layouts(a.seed, 'test') if recording else source_meta['layout']
    symbols.set_layout(layout)
    offsets, placement = minimum_vertical_offsets(raw)
    root = raw.object.data.root_state_w.clone()
    root[:,2] += torch.tensor(offsets, device=a.device)
    raw.object.write_root_state_to_sim(root)
    raw.scene.write_data_to_sim(); raw.sim.forward(); raw.scene.update(0.)
    raw._compute_intermediate_values(); raw._update_face_markers()
    save(out/'placement.json', placement)
    cold = normalized_target(raw).clone()
    runtime = ControllerRuntime(controller, cold, False)
    if recording:
        initial = dict(physical=physical(raw), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    else:
        initial = torch.load(a.source/'initial.pt', map_location='cpu', weights_only=True)
        assert_equal(physical(raw), initial['physical'])
        torch.set_rng_state(initial['torch_rng']); torch.cuda.set_rng_state(initial['cuda_rng'])
    torch.save(initial, out/'initial.pt')
    sources = {str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),Path(__file__).with_name('control_protocol.py'),
        ROOT/'scripts/astra_retention_online_v1/retention_runtime.py',ROOT/'scripts/astra_retention_online_v1/evidence_runtime.py',
        ROOT/'scripts/astra_learning_v1/runtime.py',ROOT/'scripts/astra_runtime_v2/runtime.py',ROOT/'scripts/semantic_completion_v1/common.py',
        ROOT/'scripts/agentic/recovery_controller_v3.py', ROOT/'scripts/agentic/recovery_controller_v2.py',
        ROOT/'scripts/astra_retention_v1/window.py', ROOT/'scripts/astra_retention_v1/value_model.py',
        ROOT/'scripts/astra_completion_v3/completion_runtime.py']}
    if source_meta is not None and sources != source_meta['source_sha256']:
        raise RuntimeError('Recorder and intervention source differ')
    assets = dict(student_sha256=sha(STUDENT), adapter_sha256=sha(ADAPTER), controller_sha256=sha(CONTROLLER), evidence_sha256=sha(evidence_path))
    if source_meta is not None and any(source_meta[k] != v for k,v in assets.items()):
        raise RuntimeError('Recorder and intervention asset differ')
    save(out/'protocol.json', dict(seed=a.seed, mode=a.mode, layout=layout, cases=n, max_steps=a.max_steps,
        source_sha256=sources, student_sha256=sha(STUDENT), adapter_sha256=sha(ADAPTER), controller_sha256=sha(CONTROLLER),
        evidence_sha256=sha(evidence_path), retention_head_sha256=sha(a.retention_head) if head is not None else None,
        source=str(a.source) if a.source else None, branch_rule='First fresh pending inspect frame with any active commanded-color pose probability >= frozen threshold. Entire batch branches together.',
        input='Public commands, measured RGB-D and proprioception only. No scoring geometry influences triggering or control.',
        scope='Continuous fixed-color control diagnostic. No language, autonomous semantic stop, historical replies, or new Astra calls.',
        cloud_calls=0, candidate_status='Exploratory learned candidate; actual runtime-cohort gate remains failed. No automatic deployment.'))
    active = torch.ones(n, dtype=torch.bool, device=a.device)
    rows = [dict(case=i, drop=False, drop_step=None, drop_phase=None, pending_drop=False,
        inspect_steps=0, inspect_ready_steps=0, execute_steps=0, execute_ready_steps=0, final_ready=False) for i in range(n)]
    ih = ph = None
    actions, trace = [], []
    started = time.time()
    with torch.inference_mode():
        for step in range(a.max_steps):
            command = command_state(step, n)
            colors = command['colors']
            ids = torch.tensor([c if c is not None else 0 for c in colors], device=a.device)
            goals = text[ids]
            if not recording and step < branch_step:
                # Match render cadence without letting fresh images perturb the
                # common recorded action prefix. Physics keeps advancing.
                if step%2 == 0:
                    symbols.update(); raw.capture_camera(refresh=True)
                action = source_actions[step].clone()
                env.step(action)
                actions.append(action.cpu().clone())
                if step%100 == 0:
                    save(out/'progress.json',dict(status='replaying',step=step,branch_step=branch_step,wall_s=time.time()-started))
                continue
            if not recording and step == branch_step:
                errors = assert_equal(physical(raw), bootstrap['physical'])
                active = bootstrap['active'].to(a.device)
                rows = copy.deepcopy(bootstrap['rows'])
                ih, ph = bootstrap['ih'].to(a.device), bootstrap['ph'].to(a.device)
                proprio = bootstrap['proprio'].to(a.device)
                for key, value in bootstrap['controller'].items():
                    setattr(runtime, key, value.to(a.device) if torch.is_tensor(value) else copy.deepcopy(value))
                evidence.last = bootstrap['ready_probability'].to(a.device)
                evidence.retention_features = bootstrap['feature_bank'].to(a.device)
                proposal, current, base_action = [bootstrap[key].to(a.device) for key in ['proposal','current','base_action']]
                torch.set_rng_state(bootstrap['torch_rng']); torch.cuda.set_rng_state(bootstrap['cuda_rng'])
                torch.save(dict(physical=physical(raw), ih=ih.cpu(), ph=ph.cpu(), proposal=proposal.cpu(), base_action=base_action.cpu(),
                    ready_probability=evidence.last.cpu(), feature_bank=evidence.retention_features.cpu(), proprio=proprio.cpu(), current=current.cpu()),out/'branch_start.pt')
                save(out/'prefix_audit.json',dict(step=step,maximum=max(errors.values()),errors=errors,
                    actions_sha256=sha(a.source/'actions.pt'),bootstrap_sha256=sha(a.source/'bootstrap.pt'),shared_first_sensor_state=True))
            else:
                if step%2 == 0:
                    camera,rgb,features,compact,proprio = sensor_frame(raw,symbols)
                    ih=history_append(ih,features,pack['history']);ph=history_append(ph,proprio,pack['history'])
                    runtime.update_visual(compact)
                if step%10 == 0:
                    evidence.update(rgb,ih,ph,camera['depth'])
                proposal=student(VisualStudentBatch(ih,goals,ph))['action']
                current=normalized_target(raw)
                known=torch.full((n,),colors[0] is not None,dtype=torch.bool,device=a.device)
                pending=active & command['pending']
                age=torch.full((n,),command['age'],device=a.device)
                base_action,_=runtime.action(step,goals,proprio,proposal,current,pending,known,[command['delivered']]*n,age)
            eligible=active & (evidence.last[torch.arange(n,device=a.device),ids]>=evidence.threshold)
            if recording and branch_step is None and step%10==0 and command['pending'] and command['phase']=='inspect' and eligible.any():
                branch_step=step
                bootstrap=dict(step=step,physical=physical(raw),active=active.cpu().clone(),rows=copy.deepcopy(rows),
                    ih=ih.cpu().clone(),ph=ph.cpu().clone(),proprio=proprio.cpu().clone(),
                    controller={k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in runtime.__dict__.items() if k!='model'},
                    ready_probability=evidence.last.cpu().clone(),feature_bank=evidence.retention_features.cpu().clone(),
                    proposal=proposal.cpu().clone(),current=current.cpu().clone(),base_action=base_action.cpu().clone(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),eligible=eligible.cpu())
                torch.save(bootstrap,out/'bootstrap.pt')
            pending=active & command['pending']
            action=retention.apply(step=step,states=['execute' if command['phase']=='inspect' else command['phase']]*n,
                colors=colors,pending=pending,delivered=[command['delivered']]*n,request_ids=[command['request_id']]*n,
                active=active,ready_probability=evidence.last,feature_bank=evidence.retention_features,fresh=step%10==0,
                student_action=proposal,current=current,base_action=base_action)
            if command['phase']=='initial':action=cold.clone()
            action[~active]=current[~active]
            if not torch.isfinite(action).all():raise RuntimeError('Nonfinite action')
            runtime.commit(action)
            actions.append(action.cpu().clone())
            _,_,term,trunc,info=env.step(action)
            errors=pose_errors(raw.object_rot,ids)
            for i in range(n):
                if not active[i]:continue
                drop=bool(info['semantic_metrics']['dropped'][i])
                phase=command['phase']
                if phase in ('inspect','execute'):
                    rows[i][phase+'_steps']+=1
                    rows[i][phase+'_ready_steps']+=int(not drop and float(errors[i])<=.16)
                rows[i]['final_ready']=phase=='execute' and not drop and float(errors[i])<=.16
                if drop:rows[i].update(drop=True,drop_step=step+1,drop_phase=phase,pending_drop=bool(pending[i]),final_ready=False)
                if drop or bool(term[i] or trunc[i]):active[i]=False
            if step%10==0:
                trace.append(dict(step=step,phase=command['phase'],pending=command['pending'],active=active.cpu().tolist(),
                    probabilities=evidence.last.cpu().tolist(),score_errors=errors.cpu().tolist()))
            if step%100==0:
                save(out/'progress.json',dict(status='recording' if recording else 'branching',step=step,
                    branch_step=branch_step,active=int(active.sum()),events=len(retention.events),wall_s=time.time()-started))
            if not active.any():break
    torch.save(torch.stack(actions),out/'actions.pt')
    save(out/'trace.json',trace);save(out/'retention_events.json',retention.events)
    artifacts=['actions.pt','initial.pt']+(['bootstrap.pt'] if recording and bootstrap is not None else [])
    result=dict(status='completed',mode=a.mode,seed=a.seed,cases=n,branch_step=branch_step,
        drops=sum(r['drop'] for r in rows),prefix_drops=int((~bootstrap['active']).sum()) if bootstrap is not None else None,
        final_ready=sum(r['final_ready'] for r in rows),inspect_ready_fraction=sum(r['inspect_ready_steps'] for r in rows)/(n*480),
        execute_ready_fraction=sum(r['execute_ready_steps'] for r in rows)/(n*1140),episodes=rows,
        artifact_sha256={name:sha(out/name) for name in artifacts},source_sha256=sha(__file__),cloud_calls=0,
        note='Fixed nominal phase denominators include prefix failures and dropped remainder. Scoring only relative to supplied colors. Source record is an additional no-treatment repeat; not real Astra or semantic success.')
    save(out/'report.json',result);save(out/'progress.json',dict(status='completed',branch_step=branch_step,wall_s=time.time()-started))
    print('CONTROL_PREFIX_RESULT '+json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
