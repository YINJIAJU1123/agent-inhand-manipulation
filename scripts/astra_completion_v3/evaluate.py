"""Exploratory paired dense-verifier evaluation. Real moving physics; no promotion."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic'),str(ROOT/'scripts/astra_runtime_v2'),str(ROOT/'scripts/semantic_completion_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--arm',choices=['baseline','verify'],required=True)
p.add_argument('--evidence-model',required=True);p.add_argument('--template-set',default='canonical',choices=['canonical','paraphrase'])
p.add_argument('--max-steps',type=int,default=3600);p.add_argument('--budget',type=int,default=6)
p.add_argument('--record-video',action='store_true')
p.add_argument('--offline-smoke',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
if not a.offline_smoke:
    from astra_bridge_v6 import bridge_health
    bridge_health()
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from PIL import Image,ImageDraw
from common import *
from agent import FeedbackClient,reject_stop
from completion_runtime import DenseEvidenceRuntime
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
    verification=a.arm=='verify';observation=False
    evidence=DenseEvidenceRuntime(a.evidence_model,a.device,6)
    class OfflineClient:
        def __init__(self):self.calls=0
        def ask(self,instruction,rgb,feedback,*history):
            self.calls+=1
            # No scorer information; exercises inspect and premature stop.
            tool='inspect_face' if self.calls==1 else 'stop'
            args={'color':'red','steps':10,'evidence':'offline plumbing only'} if self.calls==1 else {'color':'red','goal_symbol':'star','evidence':'offline plumbing only'}
            return dict(tool=tool,arguments=args,response_id=f'r{self.calls}',call_id=f'c{self.calls}',usage={},elapsed_s=0.)
    clients=[FeedbackClient(OfflineClient() if a.offline_smoke else BridgeClient()) for _ in range(6)]
    agents=[AstraAgent(clients[i],c['instruction'],budget=a.budget) for i,c in enumerate(cases)]
    wrappers=[AsyncAgent(x,pool,'astra') for x in agents]
    runtime=ControllerRuntime(controller,normalized_target(raw),False)
    cold=normalized_target(raw).clone();scores=[Score() for _ in range(6)]
    active=[True]*6;pending_stop=[None]*6;events=[];wait_drop=[False]*6;cold_drop=[False]*6
    ih=ph=None;start=time.time();action_prefix=[];last_dropped=[False]*6
    save(out/'manifest.json',dict(version='dense_completion_exploratory_v3',seed=a.seed,arm=a.arm,template=a.template_set,
        student_sha256=sha(STUDENT),controller_sha256=sha(CONTROLLER),evidence_sha256=sha(a.evidence_model),
        max_steps=a.max_steps,budget=a.budget,offline_smoke=a.offline_smoke,
        success='Agent-declared completion at correct semantic target, full orientation error <= .16rad, no drop. Reaching alone never terminates.',
        verifier='Learned fresh-sensor gate on Astra stop proposals; can accept a pending proposal later while executing its skill.',
        observer='Disabled. Astra selects all observations and semantic bindings.',
        candidate_status='Exploratory evaluation of a below-gate candidate, not baseline promotion.',
        offline_gate_passed=bool(evidence.pack['offline_gate_passed']),threshold=evidence.threshold,
        timing='Simulation continues during real inference; delivery aligned to ceil(wall latency * 30). Local inference shares GPU.',
        labels='No geometry, layout, true goal color, scorer flags or error enter agent/model.',
        baseline_support='All arms receive remaining calls/time and actual executed skill; same frozen V3 controller.',
        source_sha256={str(x.relative_to(ROOT)):sha(x) for folder in ['astra_completion_v3','semantic_completion_v3','semantic_completion_v2','semantic_completion_v1'] for x in (ROOT/'scripts'/folder).glob('*.py')}))
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
            if step%10==0:evidence.update(rgb,ih,ph,camera['depth'])
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
                         actual_executed_skill=COLORS[decision['color']] if decision['color'] is not None else None,
                         actual_executed_state=decision['state'],local_stop_proposal_pending=pending_stop[i] is not None,
                         **evidence.summary(i,verification,observation))
                    w.submit(rgb[i].cpu().numpy().copy(),prop[i].cpu().numpy().copy(),step)
            ids=torch.tensor([w.snapshot['decision']['color'] or 0 for w in wrappers],device=a.device)
            goals=text[ids];proposal=student(VisualStudentBatch(ih,goals,ph))['action']
            current=normalized_target(raw)
            pending=torch.tensor([active[i] and w.future is not None for i,w in enumerate(wrappers)],device=a.device)
            known=torch.tensor([w.snapshot['decision']['color'] is not None for w in wrappers],device=a.device)
            age=torch.tensor([(step-w.request_step)/30 if w.future is not None or delivered[i] else 0. for i,w in enumerate(wrappers)],device=a.device)
            actions,diag=runtime.action(step,goals,prop,proposal,current,pending,known,delivered,age)
            actions[~known]=cold[~known]
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
