"""Isolate pre-reply fixed skill control; semantic decisions and scoring stay with Astra."""
import argparse,copy,json,os,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic'),str(ROOT/'scripts/astra_runtime_v2'),str(ROOT/'scripts/semantic_completion_v1'),str(ROOT/'scripts/astra_completion_v3'),str(ROOT/'scripts/async_stability_v1'),str(ROOT/'scripts/astra_retention_online_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--initialization',choices=['legacy','collision_clearance'],default='collision_clearance')
p.add_argument('--startup-mode',choices=['latch','fixed_green'],required=True)
p.add_argument('--memory-mode',choices=['none','memory'],required=True)
p.add_argument('--retention-scope',choices=['none'],default='none')
p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--arm',choices=['baseline','verify'],required=True)
p.add_argument('--evidence-model',required=True);p.add_argument('--template-set',default='canonical',choices=['canonical','paraphrase'])
p.add_argument('--max-steps',type=int,default=3600);p.add_argument('--budget',type=int,default=6)
p.add_argument('--record-video',action='store_true')
p.add_argument('--offline-smoke',action='store_true')
p.add_argument('--offline-retention-probe',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
if a.offline_retention_probe and not a.offline_smoke:
    p.error('--offline-retention-probe requires --offline-smoke')
if not a.offline_smoke:
    from astra_bridge_v6 import bridge_health
    bridge_health()
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from PIL import Image,ImageDraw
from common import *
from agent import FeedbackClient,reject_stop
from evidence_runtime import CachedDenseEvidence
sys.path.insert(0,str(ROOT/'scripts/astra_semantic_memory_live_v1'))
from online_memory import OnlineMemory,CHECKPOINT_SHA256
import importlib.util
_hold_spec=importlib.util.spec_from_file_location('observation_retention_hold',ROOT/'scripts/astra_observation_control_v1/hold.py')
_hold_module=importlib.util.module_from_spec(_hold_spec);_hold_spec.loader.exec_module(_hold_module)
ObservationHold=_hold_module.ObservationHold
from astra_agent_v4 import AstraAgent
from astra_bridge_v6 import BridgeClient
from async_agent_v7 import AsyncAgent
from latency_controller_v1 import normalized_target
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from semantic_protocol_v3 import make_cases
from latent_data_v4 import split_layouts
from scoring import Score,service_anomaly
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance

out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
env=None;pool=ThreadPoolExecutor(max_workers=6);videos=[];wrappers=[]
try:
    student,text,ck=load_student(a.device)
    controller=RecoveryController(616).to(a.device).eval()
    controller.load_state_dict(torch.load(CONTROLLER,map_location=a.device,weights_only=True)['model'])
    controller.requires_grad_(False)
    env,raw,symbols=environment(a.seed,a.max_steps,a.device)
    cases=make_cases(a.seed,0,a.template_set);layout=split_layouts(a.seed,'test');targets=[]
    for i,c in enumerate(cases):
        target=layout[i].index(SYMBOLS.index(c['target_symbol']));targets.append(target)
        raw.cfg.fixed_target_face=target;raw._reset_target_pose(torch.tensor([i],device=a.device))
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
    raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
    save(out/'initial.json',dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout))
    if a.initialization=='collision_clearance':
        from placement import minimum_vertical_offsets
        offsets,placement_audit=minimum_vertical_offsets(raw);save(out/'placement.json',placement_audit)
        state=raw.object.data.root_state_w.clone();state[:,2]+=torch.tensor(offsets,device=a.device)
        raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
    save(out/'initial_applied.json',dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist()))
    verification=a.arm=='verify';observation=False
    evidence=CachedDenseEvidence(a.evidence_model,a.device,6)
    retention=ObservationHold(6,a.retention_scope,threshold=evidence.threshold)
    semantic_memory=OnlineMemory(a.device,6)
    memory_feedback=[];startup_trace=[]
    class OfflineClient:
        def __init__(self,case):self.calls=0;self.case=case
        def ask(self,instruction,rgb,feedback,*history):
            supplied=feedback.get('learned_semantic_memory')
            if (supplied is not None)!=(a.memory_mode=='memory'):
                raise RuntimeError('Wrong memory exposure in scripted transport')
            if supplied is not None and set(supplied['colors'])!=set(COLORS):
                raise RuntimeError('Malformed semantic evidence payload')
            self.calls+=1
            if a.offline_retention_probe:
                # Fixed public color probes, independent of the hidden layout,
                # requested target, and outcome. Never a semantic-agent score.
                time.sleep(6.)
                tool='inspect_face' if self.calls==1 else 'reorient_target' if self.calls==2 else 'stop' if self.calls==a.budget else 'verify_target'
                args=dict(color=COLORS[self.case],goal_symbol='star',evidence='offline color probe only')
                if tool=='inspect_face':args.pop('goal_symbol');args['steps']=90
                if tool=='reorient_target':args['steps']=90
                return dict(tool=tool,arguments=args,response_id=f'r{self.calls}',call_id=f'c{self.calls}',usage={},elapsed_s=6.)
            # No scorer information; exercises inspect and premature stop.
            tool='inspect_face' if self.calls==1 else 'stop'
            args={'color':'red','steps':10,'evidence':'offline plumbing only'} if self.calls==1 else {'color':'red','goal_symbol':'star','evidence':'offline plumbing only'}
            return dict(tool=tool,arguments=args,response_id=f'r{self.calls}',call_id=f'c{self.calls}',usage={},elapsed_s=0.)
    clients=[FeedbackClient(OfflineClient(i) if a.offline_smoke else BridgeClient()) for i in range(6)]
    agents=[AstraAgent(clients[i],c['instruction'],budget=a.budget) for i,c in enumerate(cases)]
    wrappers=[AsyncAgent(x,pool,'astra') for x in agents]
    runtime=ControllerRuntime(controller,normalized_target(raw),False)
    cold=normalized_target(raw).clone();scores=[Score() for _ in range(6)]
    active=[True]*6;pending_stop=[None]*6;events=[];wait_drop=[False]*6;cold_drop=[False]*6
    ih=ph=None;start=time.time();action_prefix=[];last_dropped=[False]*6;response_states={}
    torch.save(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'initial_rng.pt')
    save(out/'manifest.json',dict(version='astra_startup_live_v1',startup_mode=a.startup_mode,startup_skill=1,memory_mode=a.memory_mode,semantic_reader_sha256=CHECKPOINT_SHA256,semantic_threshold=.85,retention_scope=a.retention_scope,retention_horizon=180,initialization=a.initialization,seed=a.seed,arm=a.arm,template=a.template_set,
        student_sha256=sha(STUDENT),controller_sha256=sha(CONTROLLER),evidence_sha256=sha(a.evidence_model),
        max_steps=a.max_steps,budget=a.budget,offline_smoke=a.offline_smoke,offline_retention_probe=a.offline_retention_probe,
        offline_scope='Fixed color, six-second simulated requests. Integration only, not Astra task performance.' if a.offline_retention_probe else None,
        success='Agent-declared completion at correct semantic target, full orientation error <= .16rad, no drop. Reaching alone never terminates.',
        verifier='Learned fresh-sensor gate on Astra stop proposals; can accept a pending proposal later while executing its skill.',
        observer='Astra selects observations and semantic bindings. Retention may hold the commanded observation color during a pending request; it does not select another color.',
        candidate_status='Exploratory live integration of a frozen RGB evidence reader. It passed fresh identity/coverage validation, but has not established task benefit or corrected known historical Astra errors. Dense pose verifier unchanged.',
        offline_gate_passed=bool(evidence.pack['offline_gate_passed']),threshold=evidence.threshold,
        timing='Simulation continues during real inference; delivery aligned to ceil(wall latency * 30). Local inference shares GPU.',
        labels='No geometry, layout, true goal color, scorer flags or error enter agent/model.',
        baseline_support='All arms receive remaining calls/time and actual executed skill; same frozen V3 controller.',
        source_sha256={str(x.relative_to(ROOT)):sha(x) for folder in ['async_stability_v1','astra_completion_v3','semantic_completion_v3','semantic_completion_v2','semantic_completion_v1','astra_retention_v1','astra_retention_online_v1','astra_retention_live_v2','astra_startup_live_v1','astra_semantic_memory_live_v1','astra_semantic_memory_v1','astra_symbol_verified_v1','astra_symbol_invariant_v1','astra_symbol_regions_v1','astra_observation_control_v1','astra_runtime_v2'] for x in (ROOT/'scripts'/folder).glob('*.py')}))
    if a.record_video:
        from rollout_video import RolloutVideo
        videos=[RolloutVideo(out/f'case{i}.mp4') for i in range(6)]
    with torch.inference_mode():
        for step in range(a.max_steps):
            if step%2==0:
                camera,rgb,feat,compact,prop=sensor_frame(raw,symbols)
                ih=history_append(ih,feat,ck['history']);ph=history_append(ph,prop,ck['history'])
                runtime.update_visual(compact)
            # All arms run this same frontend to equalize local inference load.
            if step%10==0:
                evidence.update(rgb,ih,ph,camera['depth'])
                semantic_memory.update(rgb,step)
            delivered=[False]*6
            errors=rotation_distance(raw.object_rot,raw.goal_rot)
            for i,w in enumerate(wrappers):
                if not active[i]:continue
                delivered[i]=w.poll(step) if step%2==0 else False  # Fresh measured image at every consumed reply.
                if w.closed:
                    active[i]=False;scores[i].finish(step,'infrastructure_error');continue
                decision=w.snapshot['decision']
                if delivered[i]:
                    if decision['state']!='stop':pending_stop[i]=None
                    if decision['state']=='inspect':
                        original=decision['color'];chosen=original;values=None
                        evidence.visits[i,chosen]+=1
                        events.append(dict(case=i,step=step,event='inspect',original=original,executed=chosen,values=values))
                        decision=w.snapshot['decision']
                    if decision['state']=='stop':
                        if step%10!=0:evidence.update(rgb,ih,ph,camera['depth'])
                        pending_stop[i]=(decision['color'],w.snapshot['goal_symbol'])
                        allowed=not verification or evidence.can_stop(i,*pending_stop[i],cases[i]['instruction'])
                        proposal_correct=(w.snapshot['goal']==targets[i] and w.snapshot['goal_symbol']==cases[i]['target_symbol'] and float(errors[i])<=.16 and not last_dropped[i])
                        events.append(dict(case=i,step=step,event='astra_stop_proposed',accepted=allowed,scorer_proposal_correct=proposal_correct))
                        if verification and not allowed:
                            reject_stop(w,step);decision=w.snapshot['decision']
                stop=decision['state']=='stop'
                if verification and pending_stop[i] is not None and (step%10==0 or delivered[i]):
                    stop=evidence.can_stop(i,*pending_stop[i],cases[i]['instruction'])
                grounded=(w.snapshot['goal']==targets[i] and w.snapshot['goal_symbol']==cases[i]['target_symbol'] and decision['state'] in ('execute','verify','stop'))
                if scores[i].update(step,grounded,float(errors[i]),last_dropped[i],stop=stop):
                    active[i]=False;w.close(step);continue
                if decision['state']=='abstain':
                    active[i]=False;scores[i].finish(step,'abstain');w.close(step);continue
                if step%10==0 and w.future is None:
                    clients[i].context=dict(remaining_calls=a.budget-w.submitted,remaining_sim_seconds=(a.max_steps-step)/30,
                         actual_executed_skill=COLORS[decision['color']] if decision['color'] is not None else ('green' if a.startup_mode=='fixed_green' else None),
                         actual_executed_state=decision['state'] if decision['color'] is not None else ('startup' if a.startup_mode=='fixed_green' else decision['state']),local_stop_proposal_pending=pending_stop[i] is not None,
                         **evidence.summary(i,verification,observation))
                    if a.memory_mode=='memory':
                        clients[i].context['learned_semantic_memory']=semantic_memory.summary(i,step)
                    before=w.submitted
                    w.submit(rgb[i].cpu().numpy().copy(),prop[i].cpu().numpy().copy(),step)
                    if w.submitted>before:
                        memory_feedback.append(dict(case=i,step=step,sequence=w.submitted,
                            exposed=a.memory_mode=='memory',context=copy.deepcopy(clients[i].context.get('learned_semantic_memory'))))
            if any(delivered):
                physical={k:v.detach().cpu().clone() for k,v in dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,cur_targets=raw.cur_targets,actions=raw.actions,prev_actions=raw.prev_actions,ih=ih,ph=ph).items()}
                controller_state={k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in runtime.__dict__.items() if k!='model'}
                response_states[step]=dict(physical=physical,controller=controller_state,active=list(active),delivered=list(delivered),snapshots=[copy.deepcopy(w.snapshot) for w in wrappers],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            known=torch.tensor([w.snapshot['decision']['color'] is not None for w in wrappers],device=a.device)
            ids=torch.tensor([w.snapshot['decision']['color'] if w.snapshot['decision']['color'] is not None else 1 for w in wrappers],device=a.device)
            goals=text[ids];proposal=student(VisualStudentBatch(ih,goals,ph))['action']
            current=normalized_target(raw)
            pending=torch.tensor([active[i] and w.future is not None for i,w in enumerate(wrappers)],device=a.device)
            known=torch.tensor([w.snapshot['decision']['color'] is not None for w in wrappers],device=a.device)
            age=torch.tensor([(step-w.request_step)/30 if w.future is not None or delivered[i] else 0. for i,w in enumerate(wrappers)],device=a.device)
            control_known=known | (torch.tensor(active,device=a.device) & (a.startup_mode=='fixed_green'))
            actions,diag=runtime.action(step,goals,prop,proposal,current,pending,control_known,delivered,age)
            actions=retention.apply(step=step,
                states=[w.snapshot['decision']['state'] for w in wrappers],
                colors=[w.snapshot['decision']['color'] for w in wrappers],
                pending=pending,delivered=delivered,request_ids=[w.submitted for w in wrappers],
                active=active,ready_probability=evidence.last,feature_bank=evidence.retention_features,
                fresh=step%10==0,student_action=proposal,current=current,base_action=actions)
            if a.startup_mode=='latch':actions[~known]=cold[~known]
            if step%10==0:
                startup_trace.append(dict(step=step,active=list(active),semantic_skill_delivered=known.cpu().tolist(),
                    local_startup_active=(~known & torch.tensor(active,device=a.device)).cpu().tolist(),
                    startup_mode=a.startup_mode,control_known=control_known.cpu().tolist(),
                    action_delta_from_latch=(actions-cold).abs().amax(-1).cpu().tolist()))
            for i in range(6):
                if not active[i]:actions[i]=current[i]
            runtime.commit(actions);action_prefix.append(actions.cpu())
            if videos and step%2==0:
                for i,v in enumerate(videos):
                    if active[i]:
                        image=Image.fromarray(rgb[i].cpu().numpy());draw=ImageDraw.Draw(image)
                        draw.rectangle((0,0,512,40),fill='black');draw.text((5,4),f'{a.arm} sim={step/30:.1f}s calls={wrappers[i].submitted} pending={bool(pending[i])}',fill='white')
                        draw.text((5,21),'Autonomous stop | physics continues',fill='white');v.append(np.asarray(image))
            _,_,term,trunc,info=env.step(actions)
            metrics=info['semantic_metrics']
            for i,w in enumerate(wrappers):
                if not active[i]:continue
                w.advanced();dropped=bool(metrics['dropped'][i]);last_dropped[i]=dropped
                grounded=(w.snapshot['goal']==targets[i] and w.snapshot['goal_symbol']==cases[i]['target_symbol'] and w.snapshot['decision']['state'] in ('execute','verify'))
                terminal=scores[i].update(step+1,grounded,float(metrics['orientation_error'][i]),dropped)
                if dropped:
                    wait_drop[i]=bool(pending[i]);cold_drop[i]=not bool(known[i])
                if terminal or bool(term[i] or trunc[i]):
                    scores[i].finish(step+1);active[i]=False;w.close(step+1)
            if step%100==0:
                torch.save(dict(actions=torch.stack(action_prefix),responses=response_states),out/'replay_latest.pt.tmp');(out/'replay_latest.pt.tmp').replace(out/'replay_latest.pt')
                save(out/'progress.json',dict(step=step,active=sum(active),reached=sum(s.reached for s in scores),success=sum(s.success for s in scores),calls=sum(w.submitted for w in wrappers),wall_s=time.time()-start))
                print(json.dumps(json.loads((out/'progress.json').read_text())),flush=True)
            if not any(active):break
    for w in wrappers:w.close(step+1)
    for v in videos:v.close()
    videos=[]
    pool.shutdown(wait=True,cancel_futures=True)
    for w in wrappers:w.poll(step+1)
    rows=[]
    for i,w in enumerate(wrappers):
        rows.append(dict(case=i,instruction=cases[i]['instruction'],target_symbol=cases[i]['target_symbol'],target_color=targets[i],
             **scores[i].finish(step+1),wait_drop=wait_drop[i],cold_drop=cold_drop[i],calls=w.submitted,service_anomaly=service_anomaly(w.events),
             async_events=w.events,trace=agents[i].trace,feedback_trace=agents[i].feedback_trace,final_delivered=w.snapshot,
             controller_events=[e for e in events if e['case']==i]))
    torch.save(torch.stack(action_prefix),out/'actions.pt')
    torch.save(response_states,out/'response_states.pt')
    save(out/'retention_events.json',retention.events)
    save(out/'memory_feedback.json',memory_feedback)
    save(out/'startup_trace.json',startup_trace)
    torch.save(semantic_memory.records,out/'semantic_memory_predictions.pt')
    save(out/'report.json',dict(metadata=json.loads((out/'manifest.json').read_text()),episodes=rows,wall_s=time.time()-start))
    print(json.dumps(dict(completed=True,success=sum(s.success for s in scores),reached=sum(s.reached for s in scores))),flush=True)
except BaseException:
    save(out/'runtime_failure.json',dict(error=traceback.format_exc(),step=locals().get('step')))
    traceback.print_exc();raise
finally:
    pool.shutdown(wait=True,cancel_futures=True)
    for v in videos:v.close()
    if env is not None:env.close()
    app.close()
