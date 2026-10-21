"""Sensor-triggered, same-prefix continuation/slow/hold counterfactuals.

No external model is called. The selected color is the recorded Astra hypothesis.
Ground-truth errors score the intervention and never select an action or trigger.
"""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/f) for f in ('agentic','semantic_completion_v1','astra_completion_v3','async_stability_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--rate',type=float,choices=[0.,.25,1.],required=True);p.add_argument('--horizon',type=int,default=180);p.add_argument('--case',type=int)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch
from common import *
from completion_runtime import DenseEvidenceRuntime
from semantic_protocol_v3 import make_cases
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from sensors import object_regions
from latency_controller_v1 import normalized_target
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch

def all_errors(q):
 s=2**-.5;targets=torch.tensor([[s,0,-s,0],[s,0,s,0],[s,s,0,0],[s,-s,0,0],[1,0,0,0],[0,1,0,0]],device=q.device)
 return 2*torch.acos((q[:,None]*targets[None]).sum(-1).abs().clamp(0,1))

env=None
try:
 src=Path(a.source).resolve();report=json.loads((src/'report.json').read_text());meta=report['metadata'];seed=meta['seed']
 plan={r['case']:r for r in json.loads(Path(a.plan).read_text()) if Path(r['source']).resolve()==src}
 if a.case is not None:plan={i:r for i,r in plan.items()if i==a.case}
 actions=torch.load(src/'actions.pt',map_location='cpu',weights_only=True);recorded=torch.load(src/'response_states.pt',map_location='cpu',weights_only=False)
 n=len(report['episodes']);student,text,pack=load_student(a.device);env,raw,symbols=environment(seed,meta['max_steps'],a.device,num_envs=n)
 layout=json.loads((src/'initial.json').read_text())['layout']
 for i,c in enumerate(report['episodes']):
  raw.cfg.fixed_target_face=c['target_color'];raw._reset_target_pose(torch.tensor([i],device=a.device))
 raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
 initial=dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout)
 assert initial==json.loads((src/'initial.json').read_text())
 if meta['initialization']=='collision_clearance':
  offsets,_=minimum_vertical_offsets(raw);state=raw.object.data.root_state_w.clone();state[:,2]+=torch.tensor(offsets,device=a.device)
  raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
 evidence=DenseEvidenceRuntime(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt',a.device,n)
 rng=torch.load(src/'initial_rng.pt',weights_only=True);torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
 ih=ph=None;started=time.time();branch_states={};branch_features={};results={};trace=[];audits=[]
 ids=torch.tensor([plan[i]['goal'] if i in plan else 0 for i in range(n)],device=a.device)
 total=max((r['step']+a.horizon for r in plan.values()),default=0)
 with torch.inference_mode():
  for step in range(total):
   if step%2==0:
    camera,rgb,feat,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,feat,pack['history']);ph=history_append(ph,prop,pack['history'])
   if step in recorded:
    actual=dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,cur_targets=raw.cur_targets,actions=raw.actions,prev_actions=raw.prev_actions)
    mask=[i for i in plan if step<=plan[i]['step']]
    if mask:
     delta={k:float((v.cpu()[mask]-recorded[step]['physical'][k][mask]).abs().max())for k,v in actual.items()};audits.append(dict(step=step,cases=mask,errors=delta))
     save(out/'prefix_audit.json',dict(audits=audits,maximum=max(max(x['errors'].values())for x in audits)))
     if max(delta.values())>1e-6:raise RuntimeError('Pre-branch physical replay mismatch at '+str(step)+': '+str(delta))
   new=[i for i in plan if plan[i]['step']==step]
   if new:
    regions=object_regions(rgb,camera['depth']);x=evidence.encoder.candidates(ih,ph)
    for i in new:
     branch_states[i]={k:v[i].cpu().clone()for k,v in dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,cur_targets=raw.cur_targets,actions=raw.actions,prev_actions=raw.prev_actions).items()}
     branch_features[i]=dict(x=x[i].cpu(),**{k:v[i].cpu()for k,v in regions.items()})
     goal=plan[i]['goal'];e=float(all_errors(raw.object_rot)[i,goal]);results[i]=dict(case=i,trigger_step=step,goal=goal,rate=a.rate,initial_error=e,initial_ready=e<=.16,binding_correct=goal==report['episodes'][i]['target_color'],drop=False,ready_steps=0,steps=0,final_error=e)
   proposal=student(VisualStudentBatch(ih,text[ids],ph))['action'];current=normalized_target(raw)
   command=actions[step].to(a.device).clone() if step<len(actions) else current.clone()
   active=[]
   for i,r in plan.items():
    if r['step']<=step:
     if step<r['step']+a.horizon and not results[i]['drop']:
      command[i]=current[i]+a.rate*(proposal[i]-current[i]);active.append(i)
     else:command[i]=current[i]
   _,_,term,trunc,info=env.step(command);e=all_errors(raw.object_rot)
   for i in active:
    result=results[i];err=float(e[i,result['goal']]);drop=bool(info['semantic_metrics']['dropped'][i]);result['steps']+=1;result['ready_steps']+=int(err<=.16 and not drop);result['drop']|=drop;result['final_error']=err
    trace.append(dict(case=i,step=step+1,error=err,drop=drop))
   if step%100==0:save(out/'progress.json',dict(status='branching',step=step,total_steps=total,rate=a.rate,wall_s=time.time()-started));print(step,flush=True)
 for r in results.values():
  r['ready_fraction']=r['ready_steps']/a.horizon;r['final_ready']=r['final_error']<=.16 and not r['drop'];r['quality']=r['ready_fraction']-float(r['drop'])
 torch.save(dict(states=branch_states,features=branch_features),out/'branches.pt')
 save(out/'report.json',dict(status='completed',source=str(src),rate=a.rate,horizon=a.horizon,results=list(results.values()),trace=trace,audits=audits,source_sha256=sha(__file__),plan_sha256=sha(a.plan),cloud_calls=0,note='Recorded Astra hypothesis, six-second local branch. Not an online Astra outcome; labels are goal-relative and do not establish semantic correctness.'))
 save(out/'progress.json',dict(status='completed',rate=a.rate,branches=len(results),wall_s=time.time()-started))
except BaseException:
 save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if env is not None:env.close()
 app.close()
