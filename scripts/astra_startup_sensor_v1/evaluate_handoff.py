"""Check startup stabilization through the first public-color skill handoff.

Both arms share the prior measured initial snapshot. At six seconds the same
predetermined color command is delivered. Score six consecutive <=0.16rad
steps, retaining every startup failure. No semantic goal or external agent.
"""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/x) for x in
    ('agentic','semantic_completion_v1','astra_runtime_v2','async_stability_v1','astra_retention_online_v1','astra_control_prefix_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['latch','visual_v3'],required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
if (a.mode=='record') != (a.source is None):p.error('Only recording has no source')
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch
from common import load_student,environment,sensor_frame,history_append,STUDENT,ADAPTER,CONTROLLER,save,sha
from evidence_runtime import CachedDenseEvidence
from latency_controller_v1 import normalized_target
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from control_protocol import physical,assert_equal,pose_errors
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
out=a.output;out.mkdir(parents=True,exist_ok=False);env=None
try:
    n=6;student,text,pack=load_student(a.device)
    model=RecoveryController(616).to(a.device).eval();model.load_state_dict(torch.load(CONTROLLER,map_location=a.device,weights_only=True)['model']);model.requires_grad_(False)
    evidence_path=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
    evidence=CachedDenseEvidence(evidence_path,a.device,n)
    env,raw,symbols=environment(a.seed,1500,a.device);layout=split_layouts(a.seed,'test');symbols.set_layout(layout)
    offsets,placement=minimum_vertical_offsets(raw);state=raw.object.data.root_state_w.clone();state[:,2]+=torch.tensor(offsets,device=a.device)
    raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
    save(out/'placement.json',placement)
    cold=normalized_target(raw).clone();runtime=ControllerRuntime(model,cold,False)
    frozen={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/astra_startup_sensor_v1/evaluate.py',ROOT/'scripts/astra_control_prefix_v1/control_protocol.py',
        ROOT/'scripts/semantic_completion_v1/common.py',ROOT/'scripts/astra_runtime_v2/runtime.py',ROOT/'scripts/astra_learning_v1/runtime.py',
        ROOT/'scripts/astra_retention_online_v1/evidence_runtime.py',ROOT/'scripts/astra_completion_v3/completion_runtime.py',
        ROOT/'scripts/agentic/recovery_controller_v3.py',ROOT/'scripts/agentic/recovery_controller_v2.py',
        ROOT/'scripts/async_stability_v1/placement.py',ROOT/'scripts/async_stability_v1/geometry_v2.py',STUDENT,ADAPTER,CONTROLLER,evidence_path]}
    save(out/'protocol.json',dict(seed=a.seed,mode=a.mode,cases=n,max_steps=1200,handoff_step=180,source_sha256=frozen,handoff_source_sha256=sha(__file__),
        selected_goal='Visual startup hypothesis only before step 180. Then every case i receives public color i in both arms, independent of hidden semantic layout.',
        controls='Same frozen student/V3/evidence/checkpoints, collision-clearance reset and initial sensor snapshot. Only startup action differs.',
        scope='Six-second startup, then first public color skill. Scored completion requires six consecutive steps <=0.16rad, no drop. No Astra, language grounding or agent-declared completion.',cloud_calls=0))
    if a.mode=='record':
        before=physical(raw)
        camera,rgb,features,compact,proprio=sensor_frame(raw,symbols)
        ih=history_append(None,features,pack['history']);ph=history_append(None,proprio,pack['history'])
        evidence.update(rgb,ih,ph,camera['depth'])
        assert_equal(physical(raw),before,tolerance=0.)
        bootstrap=dict(seed=a.seed,physical=physical(raw),ih=ih.cpu(),ph=ph.cpu(),compact=compact.cpu(),proprio=proprio.cpu(),
            probability=evidence.last.cpu(),cold=cold.cpu(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),layout=layout)
        torch.save(bootstrap,out/'bootstrap.pt')
        save(out/'report.json',dict(status='completed',mode='record',seed=a.seed,cases=n,bootstrap_sha256=sha(out/'bootstrap.pt'),cloud_calls=0))
    else:
        source=json.loads((a.source/'protocol.json').read_text());report=json.loads((a.source/'report.json').read_text())
        if source['source_sha256']!=frozen or report['status']!='completed' or sha(a.source/'bootstrap.pt')!=report['bootstrap_sha256']:
            raise RuntimeError('Frozen source, assets, or bootstrap changed')
        b=torch.load(a.source/'bootstrap.pt',map_location='cpu',weights_only=True)
        if b['seed']!=a.seed or b['layout']!=layout:raise RuntimeError('Wrong source initialization')
        mismatch=assert_equal(physical(raw),b['physical'],tolerance=0.)
        assert_equal({'cold':cold.cpu()},{'cold':b['cold']},tolerance=0.)
        torch.set_rng_state(b['torch_rng']);torch.cuda.set_rng_state(b['cuda_rng'])
        ih,ph,compact,proprio=[b[k].to(a.device) for k in ('ih','ph','compact','proprio')]
        ids=b['probability'].to(a.device).argmax(-1);startup_ids=ids.clone();public_ids=torch.arange(n,device=a.device);goals=text[ids]
        save(out/'initial_audit.json',dict(physical_errors=mismatch,shared_sensor_state=True,
            source_sha256=sha(a.source/'bootstrap.pt'),chosen_orientation_hypotheses=ids.cpu().tolist(),
            selected_model_scores=b['probability'].gather(1,ids.cpu()[:,None]).flatten().tolist()))
        active=torch.ones(n,dtype=torch.bool,device=a.device);rows=[dict(case=i,drop=False,drop_step=None,completed=False,completion_step=None) for i in range(n)]
        dwell=torch.zeros(n,dtype=torch.long,device=a.device)
        traces=[];actions=[];started=time.time()
        with torch.inference_mode():
            for step in range(1200):
                if step%2==0:
                    if step:
                        camera,rgb,features,compact,proprio=sensor_frame(raw,symbols)
                        ih=history_append(ih,features,pack['history']);ph=history_append(ph,proprio,pack['history'])
                    runtime.update_visual(compact)
                ids=startup_ids if step<180 else public_ids;goals=text[ids]
                current=normalized_target(raw);proposal=student(VisualStudentBatch(ih,goals,ph))['action']
                pending=active & (step<180)
                age=torch.full((n,),step/30 if step<=180 else 0.,device=a.device)
                action,diag=runtime.action(step,goals,proprio,proposal,current,pending,torch.ones_like(active),[step==180]*n,age)
                if a.mode=='latch' and step<180:action=cold.clone()
                action[~active]=current[~active]
                if not torch.isfinite(action).all():raise RuntimeError('Nonfinite startup action')
                runtime.commit(action);actions.append(action.cpu().clone())
                pre=dict(step=step,active=active.cpu().tolist(),object_velocity=raw.object.data.root_state_w[:,7:].cpu().tolist(),
                    object_distance=torch.linalg.vector_norm(raw.object_pos-raw.in_hand_pos,dim=-1).cpu().tolist(),
                    maximum_command_delta=(action-current).abs().amax(-1).cpu().tolist(),rate=diag['rate'])
                _,_,term,trunc,info=env.step(action);metrics=info['semantic_metrics']
                newly=active & metrics['dropped'].bool()
                for i in newly.nonzero().flatten().tolist():rows[i].update(drop=True,drop_step=step+1)
                error=pose_errors(raw.object_rot,public_ids)
                reached=active & ~newly & (error<=.16) & (step>=180)
                dwell=torch.where(reached,dwell+1,torch.zeros_like(dwell))
                completed=active & (dwell>=6)
                for i in completed.nonzero().flatten().tolist():rows[i].update(completed=True,completion_step=step+1)
                active &= ~(newly|term|trunc|completed)
                if step<60 or step%10==0:
                    pre.update(new_drops=newly.cpu().tolist(),post_distance=metrics['object_distance'].cpu().tolist(),score_orientation_error=error.cpu().tolist(),completed=completed.cpu().tolist());traces.append(pre)
                if step%60==0:save(out/'progress.json',dict(status='running',step=step,active=int(active.sum()),wall_s=time.time()-started))
                if not active.any():break
        torch.save(torch.stack(actions),out/'actions.pt');save(out/'trace.json',traces)
        result=dict(status='completed',mode=a.mode,seed=a.seed,cases=n,episodes=rows,drops=sum(r['drop'] for r in rows),completed=sum(r['completed'] for r in rows),startup_drops=sum(r['drop'] and r['drop_step']<=180 for r in rows),post_handoff_drops=sum(r['drop'] and r['drop_step']>180 for r in rows),
            drops_by_seconds={str(t):sum(r['drop'] and r['drop_step']<=30*t for r in rows) for t in (1,3,6,12)},cloud_calls=0,
            note='All six original cases retained. Development first-skill completion with a public supplied color, not autonomous semantic success. No default change.')
        save(out/'report.json',result);print('STARTUP_RESULT '+json.dumps(result),flush=True)
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
