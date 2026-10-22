"""Train dense target-conditioned continuation with one-third-second options."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/s) for s in ('agentic','semantic_completion_v1','astra_runtime_v2','async_stability_v1','astra_control_prefix_v1','astra_retention_online_v1','semantic_completion_v3')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--num-envs',type=int,default=12);p.add_argument('--iterations',type=int,default=32)
p.add_argument('--mode',choices=['train','original','learned'],default='train');p.add_argument('--checkpoint',type=Path);p.add_argument('--source',type=Path);p.add_argument('--delay',type=float,default=6.)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import numpy as np
import torch
from model import ModeController,optimizers,update,command
from evidence_runtime import CachedDenseEvidence
from common import load_student,environment,sensor_frame,history_append,save,sha,STUDENT,ADAPTER,CONTROLLER
from placement import minimum_vertical_offsets
from runtime import ControllerRuntime
from recovery_controller_v3 import RecoveryController
from recovery_controller_v2 import observation
from latency_controller_v1 import normalized_target
from control_protocol import physical,pose_errors
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
out=a.output;out.mkdir(parents=True,exist_ok=False);env=None
try:
 n=a.num_envs;device=a.device;assert n in (6,12,24)
 evidence_path=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt';evidence=CachedDenseEvidence(evidence_path,device,n)
 student,text,pack=load_student(device);env,raw,symbols=environment(a.seed,1200,device,num_envs=n)
 oldpack=torch.load(CONTROLLER,map_location=device,weights_only=False);base=RecoveryController(616).to(device).eval();base.load_state_dict(oldpack['model']);base.requires_grad_(False)
 torch.manual_seed(124001);model=ModeController().to(device);model.initialize(oldpack['model'])
 if a.checkpoint:model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=False)['model'])
 opts=optimizers(model);train=a.mode=='train';history=[];started=time.time()
 protocol=dict(version='goal_maintenance_v2',evidence_sha256=sha(evidence_path),option_steps=10,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},student_sha256=sha(STUDENT),adapter_sha256=sha(ADAPTER),controller_sha256=sha(CONTROLLER),source_sha256=sha(__file__),model_sha256=sha(Path(__file__).with_name('model.py')),cloud_calls=0,
  actor='Measured 616-feature control input and frozen 125-feature dense representation queried by the delivered goal. No physical labels, future goal or remaining latency. Dense features and categorical options refresh every 10 steps, 3Hz.',
  actions=['original frozen V3/student command','0.15 student speed','hold current joint target'],
  schedule='Frozen original policy before first changed-goal reply. Execute 2s, wait 6s for same-goal reaffirmation, execute 2s. Categorical controller covers this full 10s after first reply.',
  reward='0.04 per correct pose step, +2 for first six-step correct hold, +12 for final six-step hold, -12 drop, -4 surviving final failure, +1 pose progress, -0.002 action change.',
  outcome='No early success termination. Final six consecutive steps <=.16rad and no drop; report full-horizon ready fraction and drops separately. All initial failures retained.',
  selection='Fixed final iteration. No outcome threshold selection, audit-state training or automatic deployment.',scope='Joint representation and action-duration development prototype. Not an isolated ablation or real Astra semantic result.')
 save(out/'protocol.json',protocol)
 for iteration in range(a.iterations if train else 1):
  seed=a.seed+iteration;delay=[2.,6.,12.][iteration%3] if train else a.delay;prefix=30*(1+seed%3);wait=round(delay*30);arrival=prefix+wait;horizon=arrival+300
  torch.manual_seed(seed);np.random.seed(seed);env.reset(seed=seed)
  oldgoals=(torch.arange(n,device=device)+seed)%6;newgoals=(oldgoals+1+seed%5)%6;goals=oldgoals.clone()
  for face in range(6):raw.cfg.fixed_target_face=face;raw._reset_target_pose((goals==face).nonzero().flatten())
  raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
  layout=np.stack([np.random.default_rng(seed+i).permutation(6) for i in range(n)]);symbols.set_layout(layout)
  offsets,_=minimum_vertical_offsets(raw);placed=raw.object.data.root_state_w.clone();placed[:,2]+=torch.tensor(offsets,device=device)
  raw.object.write_root_state_to_sim(placed);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
  runtime=ControllerRuntime(base,normalized_target(raw),False);active=torch.ones(n,dtype=torch.bool,device=device);ih=ph=None;rollout=[];prefix_actions=[]
  previous_error=pose_errors(raw.object_rot,goals);consecutive=torch.zeros(n,dtype=torch.long,device=device);bonus=torch.zeros_like(active);ready_steps=torch.zeros(n,dtype=torch.long,device=device)
  dropped=torch.zeros_like(active);pre_dropped=torch.zeros_like(active);counts=torch.zeros(n,3,dtype=torch.long,device=device);held_mode=torch.zeros(n,dtype=torch.long,device=device);trace=[];interventions=torch.zeros(n,dtype=torch.long,device=device)
  ref=torch.load(a.source/'branch.pt',map_location=device,weights_only=False) if a.source else None
  recorded=torch.load(a.source/'prefix_actions.pt',map_location=device,weights_only=True) if a.source else None
  if ref is not None:assert ref['arrival']==arrival and ref['seed']==seed
  initial=physical(raw)
  if ref is not None:
   assert all(torch.equal(v,ref['initial'][k].cpu()) for k,v in initial.items()),'Initial state mismatch'
  with torch.no_grad():
   for step in range(horizon):
    offset=step-arrival
    if ref is not None and step<arrival:
     if step%2==0:symbols.update();raw.capture_camera(refresh=True)
     env.step(recorded[step]);continue
    if step==arrival:
     goals=newgoals.clone()
     for face in range(6):raw.cfg.fixed_target_face=face;raw._reset_target_pose((goals==face).nonzero().flatten())
     raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();previous_error=pose_errors(raw.object_rot,goals)
    if ref is not None and step==arrival:
     actual=physical(raw);errors={k:float((v-ref['physical'][k].cpu()).abs().max()) for k,v in actual.items()};save(out/'prefix_audit.json',dict(errors=errors,maximum=max(errors.values())))
     if max(errors.values())!=0:raise RuntimeError('Physical reply state differs')
     ih=ref['ih'];ph=ref['ph'];prop=ref['prop'];active=ref['active'];dropped=ref['dropped'];pre_dropped=ref['pre_dropped']
     for key,value in ref['runtime'].items():setattr(runtime,key,value.to(device) if torch.is_tensor(value) else copy.deepcopy(value))
     torch.set_rng_state(ref['rng_cpu'].cpu());torch.cuda.set_rng_state(ref['rng_gpu'].cpu());evidence.retention_features=ref['dense'].to(device)
    elif step%2==0:
     camera,rgb,f,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,f,pack['history']);ph=history_append(ph,prop,pack['history']);runtime.update_visual(compact)
    if step>=arrival and offset%10==0 and not (ref is not None and step==arrival):
     evidence.update(rgb,ih,ph,camera['depth'])
    if step==arrival and not train:
     branch=dict(dense=evidence.retention_features.cpu().clone(),seed=seed,arrival=arrival,initial=initial,physical=physical(raw),ih=ih.cpu().clone(),ph=ph.cpu().clone(),prop=prop.cpu().clone(),active=active.cpu().clone(),dropped=dropped.cpu().clone(),pre_dropped=pre_dropped.cpu().clone(),
      runtime={k:v.cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in runtime.__dict__.items() if k!='model'},rng_cpu=torch.get_rng_state(),rng_gpu=torch.cuda.get_rng_state())
     torch.save(branch,out/'branch.pt')
     if not ref:torch.save(torch.stack(prefix_actions),out/'prefix_actions.pt')
    pending=torch.full((n,),prefix<=step<arrival or 60<=offset<240,dtype=torch.bool,device=device)
    delivered=[step in (arrival,arrival+240)]*n
    req_age=(step-prefix)/30 if prefix<=step<=arrival else (offset-60)/30 if 60<=offset<=240 else 0.
    age=torch.full((n,),req_age,device=device);proposal=student(VisualStudentBatch(ih,text[goals],ph))['action'];current=normalized_target(raw)
    baseline,diag=runtime.action(step,text[goals],prop,proposal,current,pending,torch.ones_like(active),delivered,age)
    if offset>=0:
     response_age=torch.tensor(diag['response_age'],device=device).clamp(max=2)
     obs=observation(runtime.history,prop,proposal,current,runtime.anchor,runtime.previous,torch.where(pending,age,runtime.wait_age),pending,response_age)
     obs=torch.cat((obs,evidence.retention_features[torch.arange(n,device=device),goals]),-1)
     if offset%10==0:
      dist=model.distribution(obs,text[goals]);held_mode=dist.sample() if train else dist.probs.argmax(-1)
      if train:
       priv=torch.cat((previous_error[:,None],(raw.object_pos-raw.in_hand_pos).norm(dim=-1)[:,None],raw.object_linvel,raw.object_angvel),-1)
       rollout.append(dict(obs=obs.clone(),goal=text[goals].clone(),priv=priv.clone(),mode=held_mode.clone(),logprob=dist.log_prob(held_mode),logits=dist.logits.clone(),value=model.value(obs,text[goals],priv),reward=torch.zeros(n,device=device),done=torch.zeros_like(active),valid=active.clone()))
     if a.mode=='original':held_mode.zero_()
     action=command(held_mode,baseline,proposal,current)
     counts.scatter_add_(1,held_mode[:,None],active.long()[:,None])
     interventions+=active&((action-baseline).abs().amax(-1)>1e-6)
    else:action=baseline
    action=torch.where(active[:,None],action,current);assert torch.isfinite(action).all() and float(action.abs().max())<=1.00001
    if step<arrival:prefix_actions.append(action.cpu().clone())
    runtime.commit(action);before=active.clone();_,_,term,trunc,info=env.step(action)
    error=pose_errors(raw.object_rot,goals);drop=before&info['semantic_metrics']['dropped'].bool();dropped|=drop
    if (trunc&before).any() or (term&before&~drop).any():raise RuntimeError('Unexpected environment end')
    if offset<0:pre_dropped|=drop
    else:
     ready=before&~drop&(error<=.16);consecutive=torch.where(ready,consecutive+1,torch.zeros_like(consecutive));ready_steps+=ready;first=(consecutive>=6)&~bonus;bonus|=first
     if train:
      reward=.04*ready.float()+2*first.float()-12*drop.float()+(previous_error-error)*before*~drop-.002*(action-current).square().mean(-1)*before
      rollout[-1]['reward']+=reward;rollout[-1]['done']|=drop
     if not train and offset%10==0:trace.append(dict(offset=offset,active=active.cpu().tolist(),mode=held_mode.cpu().tolist(),errors=error.cpu().tolist(),ready=ready.cpu().tolist()))
    active&=~drop;previous_error=error
    if step%100==0:save(out/'rollout_progress.json',dict(iteration=iteration+1,seed=seed,step=step,total_steps=horizon,alive=int(active.sum()),cloud_calls=0))
  success=active&(consecutive>=6)
  if train:
   rollout[-1]['reward']+=12*success.float()-4*(active&~success).float();rollout[-1]['done'][:]=True;stats=update(model,opts,rollout)
  else:stats={}
  row=dict(iteration=iteration+1,seed=seed,delay_s=delay,nominal=n,successes=int(success.sum()),drops=int(dropped.sum()),prefix_drops=int(pre_dropped.sum()),ready_fraction=float(ready_steps.sum())/(n*300),mode_counts=counts.sum(0).cpu().tolist(),intervention_steps=int(interventions.sum()),elapsed_s=time.time()-started,**stats)
  history.append(row);save(out/'history.json',history);save(out/'progress.json',dict(status='training' if train else 'evaluating',iteration=iteration+1,latest=row,updated=time.time()))
  if train:torch.save(dict(model=model.state_dict(),protocol=protocol,iteration=iteration+1),out/'controller_latest.pt')
  else:
   save(out/'trace.json',trace);save(out/'trials.json',[dict(case=i,success=bool(success[i]),drop=bool(dropped[i]),prefix_drop=bool(pre_dropped[i]),ready_steps=int(ready_steps[i]),intervention_steps=int(interventions[i]),mode_counts=counts[i].cpu().tolist()) for i in range(n)])
  print('MAINTENANCE '+json.dumps(row),flush=True)
 if train:torch.save(dict(model=model.state_dict(),protocol=protocol,iteration=a.iterations),out/'controller_final.pt')
 save(out/'result.json',dict(status='completed',mode=a.mode,final=history[-1],cloud_calls=0));save(out/'progress.json',dict(status='completed',updated=time.time()))
except BaseException:save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if env is not None:env.close()
 app.close()
