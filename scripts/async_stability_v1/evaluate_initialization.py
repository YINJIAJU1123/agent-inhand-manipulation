"""Local diagnosis of cold stabilization and reply handoff using recorded calls.

No cloud calls. Replayed decisions are fixed interventions, not fresh Astra
responses to changed images. Optional exact action-prefix replay preserves the
contact solver history before a counterfactual branch.
"""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic'),str(ROOT/'scripts/astra_runtime_v2'),str(ROOT/'scripts/semantic_completion_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True);p.add_argument('--source-report',required=True)
p.add_argument('--initial-lift',type=float,required=True)
p.add_argument('--mode',choices=['cold','replay'],default='cold')
p.add_argument('--cold-policy',choices=['latch','mean','median','nearest','red'],default='latch')
p.add_argument('--handoff',choices=['v3','direct','ramp','cap'],default='v3')
p.add_argument('--max-steps',type=int,default=360)
p.add_argument('--prefix-cache');p.add_argument('--branch-step',type=int,default=0)
p.add_argument('--record-video',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from common import load_student,environment,sensor_frame,history_append,STUDENT,ADAPTER,CONTROLLER,save,sha,COLORS
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
from latency_controller_v1 import normalized_target
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from semantic_protocol_v3 import make_cases
from latent_data_v4 import split_layouts
from semantic_agent_v3 import SYMBOLS
from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
env=None;writers=[]
try:
 source=json.loads(Path(a.source_report).read_text());meta=source['metadata'];seed=meta['seed'];template=meta['template_set']
 student,text,ck=load_student(a.device)
 model=RecoveryController(616).to(a.device).eval();model.load_state_dict(torch.load(CONTROLLER,map_location=a.device,weights_only=True)['model']);model.requires_grad_(False)
 env,raw,symbols=environment(seed,max(a.max_steps,3600),a.device)
 n=6;cases=make_cases(seed,0,template);layout=split_layouts(seed,'test');targets=[]
 for i,c in enumerate(cases):
  target=layout[i].index(SYMBOLS.index(c['target_symbol']));targets.append(target)
  raw.cfg.fixed_target_face=target;raw._reset_target_pose(torch.tensor([i],device=a.device))
 raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
 initial=dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout)
 save(out/'initial.json',initial)
 old=json.loads(Path(a.source_report).with_name('initial.json').read_text())
 initial_error=max(float((torch.tensor(initial[k])-torch.tensor(old[k])).abs().max()) for k in ('joint_pos','joint_vel','object_root','prev_targets'))
 if initial_error>1e-7:raise RuntimeError(f'Historical initial state mismatch {initial_error}')
 if a.initial_lift:
  state=raw.object.data.root_state_w.clone();state[:,2]+=a.initial_lift
  raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
  save(out/'initialization_intervention.json',dict(z_lift_m=a.initial_lift,changed_initial_state=True,not_policy_improvement=True,all_cases_retained=True))
 events=[]
 for row in source['episodes']:
  trace=row['trace'];plan=[]
  for event in row['async_events']:
   seq=event['sequence']
   item=dict(request=event['request_step'],delivery=event.get('delivery_step',a.max_steps+1),status=event['status'],latency=event.get('wall_latency_s'),decision=None)
   if event['status']=='delivered':
    call=trace[seq-1];name=call['tool'];args=call['arguments'];color=COLORS.index(args['color']) if 'color' in args else None
    item['decision']=dict(color=color,state={'inspect_face':'inspect','reorient_target':'execute','verify_target':'verify','stop':'stop','abstain':'abstain'}[name],symbol=args.get('goal_symbol'))
   plan.append(item)
  events.append(plan)
 cold=normalized_target(raw).clone();runtime=ControllerRuntime(model,cold,False)
 snapshots=[dict(color=None,state='initial',symbol=None) for _ in range(n)];goals_known=[None]*n;symbols_known=[None]*n
 active=torch.ones(n,dtype=torch.bool,device=a.device);dropped=~active;reached=~active;success=~active
 finish=torch.full((n,),-1,device=a.device,dtype=torch.long);ih=ph=None;records=[];actions_saved=[];branch_state=None
 prefix=torch.load(a.prefix_cache,map_location='cpu',weights_only=True) if a.prefix_cache else None
 if prefix is not None and (prefix['seed']!=seed or prefix['template']!=template):raise RuntimeError('Wrong prefix cache')
 if a.record_video:
  from rollout_video import RolloutVideo
  writers=[RolloutVideo(out/f'case{i}.mp4') for i in range(n)]
 manifest=dict(args=vars(a),seed=seed,template=template,initial_error=initial_error,
  source_report_sha256=sha(a.source_report),student_sha256=sha(STUDENT),adapter_sha256=sha(ADAPTER),controller_sha256=sha(CONTROLLER),
  cloud_calls=0,interpretation='Fixed recorded reply schedule; not a fresh Astra success result. Cold controls never receive the task target. All old cases and initial states retained.',
  source_sha256={str(x):sha(x) for x in Path(__file__).parent.glob('*.py')})
 save(out/'manifest.json',manifest);start=time.monotonic();last_delivery=torch.full((n,),-100000,device=a.device,dtype=torch.long)
 with torch.inference_mode():
  for step in range(a.max_steps):
   if step%2==0:
    camera,rgb,feat,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,feat,ck['history']);ph=history_append(ph,prop,ck['history']);runtime.update_visual(compact)
   if step==a.branch_step:
    branch_state={k:v.detach().cpu().clone() for k,v in dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,ih=ih,ph=ph).items()}
    if prefix is not None:
     errs={k:float((v-prefix['branch_state'][k]).abs().max()) for k,v in branch_state.items() if k not in ('ih','ph')}
     save(out/'prefix_audit.json',dict(errors=errs,max_abs_error=max(errs.values())))
     if max(errs.values())!=0.:raise RuntimeError('Contact-prefix replay mismatch')
     ih=prefix['branch_state']['ih'].to(a.device);ph=prefix['branch_state']['ph'].to(a.device)
   delivered=[False]*n;pending=torch.zeros_like(active);age=torch.zeros(n,device=a.device)
   if a.mode=='replay':
    for i,plan in enumerate(events):
     if not active[i]:continue
     for event in plan:
      if event['request']<=step<event['delivery']:
       pending[i]=True;age[i]=(step-event['request'])/30
      if event['delivery']==step and event['decision'] is not None:
       delivered[i]=True;age[i]=(step-event['request'])/30;last_delivery[i]=step;snapshots[i]=event['decision'].copy()
       if snapshots[i]['state'] not in ('inspect','abstain'):
        goals_known[i]=snapshots[i]['color'];symbols_known[i]=snapshots[i]['symbol']
   known=torch.tensor([s['color'] is not None for s in snapshots],device=a.device)
   ids=torch.tensor([s['color'] or 0 for s in snapshots],device=a.device);goals=text[ids]
   proposal=student(VisualStudentBatch(ih,goals,ph))['action'];current=normalized_target(raw)
   actions,diag=runtime.action(step,goals,raw.compute_student_proprio().float(),proposal,current,pending,known,delivered,age)
   if a.handoff=='direct':actions=torch.where((known & ~pending)[:,None],proposal,actions)
   elif a.handoff=='ramp':
    fraction=((step-last_delivery).float()/15).clamp(0,1);rate=.15+.85*fraction
    actions=torch.where((known & ~pending)[:,None],current+rate[:,None]*(proposal-current),actions)
   elif a.handoff=='cap':
    clamped=current+(actions-current).clamp(-.12,.12)
    actions=torch.where((known & ~pending)[:,None],clamped,actions)
   cold_action=cold
   if a.cold_policy!='latch':
    if a.cold_policy=='red':cold_action=student(VisualStudentBatch(ih,text[torch.zeros(n,dtype=torch.long,device=a.device)],ph))['action']
    else:
     candidates=student(VisualStudentBatch(ih[:,None].expand(-1,6,-1,-1).reshape(n*6,ck['history'],-1),text.repeat(n,1),ph[:,None].expand(-1,6,-1,-1).reshape(n*6,ck['history'],-1)))['action'].reshape(n,6,-1)
     if a.cold_policy=='mean':cold_action=candidates.mean(1)
     elif a.cold_policy=='median':cold_action=candidates.median(1).values
     else:
      chosen=(candidates-current[:,None]).square().mean(-1).argmin(1);cold_action=candidates[torch.arange(n,device=a.device),chosen]
   actions=torch.where(known[:,None],actions,cold_action)
   actions=torch.where(active[:,None],actions,current)
   if prefix is not None and step<a.branch_step:actions=prefix['actions'][step].to(a.device)
   if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action')
   actions_saved.append(actions.detach().cpu().clone());runtime.commit(actions)
   prior=active.clone();distance=torch.linalg.vector_norm(raw.object_pos-raw.goal_pos,dim=-1)
   trace=dict(step=step,active=active.cpu().tolist(),known=known.cpu().tolist(),pending=pending.cpu().tolist(),delivered=delivered,colors=[s['color'] for s in snapshots],states=[s['state'] for s in snapshots],
    object_distance=distance.cpu().tolist(),object_velocity=raw.object.data.root_state_w[:,7:].cpu().tolist(),max_command_delta=(actions-current).abs().amax(-1).cpu().tolist(),rms_command_delta=(actions-current).square().mean(-1).sqrt().cpu().tolist(),controller_rate=diag['rate'])
   if writers and step%2==0:
    for i,w in enumerate(writers):
     if active[i]:w.append(rgb[i].cpu().numpy())
   if a.mode=='replay':
    errors=rotation_distance(raw.object_rot,raw.goal_rot)
    for i,s in enumerate(snapshots):
     if active[i] and s['state'] in ('stop','abstain'):
      success[i]=(s['state']=='stop' and goals_known[i]==targets[i] and symbols_known[i]==cases[i]['target_symbol'] and float(errors[i])<=.16)
      active[i]=False;finish[i]=step
   _,_,term,trunc,info=env.step(actions);m=info['semantic_metrics'];drop=m['dropped'].bool() & active
   grounded=torch.tensor([goals_known[i]==targets[i] and symbols_known[i]==cases[i]['target_symbol'] and snapshots[i]['state'] in ('execute','verify') for i in range(n)],device=a.device)
   ok=m['goal_reached'].bool() & grounded & active & ~drop
   reached|=ok;success|=ok;dropped|=drop
   done=active & (drop|term|trunc|ok);finish[done]=step+1;active&=~done
   trace.update(new_drop=drop.cpu().tolist(),goal_reached=ok.cpu().tolist(),post_distance=m.get('object_distance',distance).cpu().tolist());records.append(trace)
   if step%100==0:
    save(out/'progress.json',dict(step=step,active=int(active.sum()),drop=int(dropped.sum()),success=int(success.sum()),wall_s=time.monotonic()-start));print(json.dumps(json.loads((out/'progress.json').read_text())),flush=True)
   if not active.any():break
 for w in writers:w.close()
 writers=[]
 save(out/'trace.json',records)
 torch.save(dict(actions=torch.stack(actions_saved),branch_state=branch_state,seed=seed,template=template),out/'prefix.pt')
 rows=[dict(case=i,success=bool(success[i]),drop=bool(dropped[i]),survived=not bool(dropped[i]),finish_step=int(finish[i]),last_color=snapshots[i]['color'],target=targets[i]) for i in range(n)]
 save(out/'report.json',dict(metadata=manifest,episodes=rows,drops=int(dropped.sum()),successes=int(success.sum()),wall_s=time.monotonic()-start))
except BaseException:
 save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 for w in writers:w.close()
 if env is not None:env.close()
 app.close()
