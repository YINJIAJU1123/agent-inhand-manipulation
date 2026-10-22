"""Counterfactual recovery at a recorded real Astra reply, with no new API calls.

Selected failure cases are development diagnostics. Future recorded replies are
not fed to the intervened hand. Only its delivered color is held for six seconds.
"""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/s) for s in ('agentic','semantic_completion_v1','astra_completion_v3','async_stability_v1','astra_runtime_v2','astra_control_prefix_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--case',type=int,required=True)
p.add_argument('--mode',choices=['original','handoff','smooth'],required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import numpy as np
import torch
from common import load_student,environment,sensor_frame,history_append,save,sha,STUDENT,ADAPTER,CONTROLLER,COLORS,SYMBOLS
from semantic_protocol_v3 import make_cases
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from latency_controller_v1 import normalized_target
from runtime import ControllerRuntime
from recovery_controller_v3 import RecoveryController
from control_protocol import pose_errors
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
from rollout_video import RolloutVideo
out=a.output;out.mkdir(parents=True,exist_ok=False);env=writer=None
try:
 report=json.loads((a.source/'report.json').read_text());meta=report['metadata'];seed=meta['seed'];n=6;i=a.case
 recorded=torch.load(a.source/'response_states.pt',map_location='cpu',weights_only=False);ref=recorded[a.step]
 actions=torch.load(a.source/'actions.pt',map_location=a.device,weights_only=True)
 assert ref['active'][i] and ref['delivered'][i] and a.step+180<len(actions)
 student,text,pack=load_student(a.device);env,raw,symbols=environment(seed,meta['max_steps'],a.device)
 cases=make_cases(seed,0,meta['template']);layout=split_layouts(seed,'test')
 for j,c in enumerate(cases):
  raw.cfg.fixed_target_face=layout[j].index(SYMBOLS.index(c['target_symbol']));raw._reset_target_pose(torch.tensor([j],device=a.device))
 raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
 initial=dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout)
 assert initial==json.loads((a.source/'initial.json').read_text())
 offsets,_=minimum_vertical_offsets(raw);placed=raw.object.data.root_state_w.clone();placed[:,2]+=torch.tensor(offsets,device=a.device)
 raw.object.write_root_state_to_sim(placed);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
 rng=torch.load(a.source/'initial_rng.pt',weights_only=True);torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
 path=ROOT/'outputs/astra_handoff_controller_v1/paired24_20261022/train/controller_final.pt' if a.mode=='handoff' else CONTROLLER
 model=RecoveryController(616).to(a.device).eval();model.load_state_dict(torch.load(path,map_location=a.device,weights_only=False)['model']);model.requires_grad_(False)
 runtime=ControllerRuntime(model,normalized_target(raw),False)
 ids=torch.tensor([s['decision']['color'] if s['decision']['color'] is not None else 1 for s in ref['snapshots']],device=a.device)
 assert ref['snapshots'][i]['decision']['color'] is not None
 delivered_event=next(x for x in report['episodes'][i]['async_events'] if x.get('delivery_step')==a.step and x['status']=='delivered')
 arrival_age=(a.step-delivered_event['request_step'])/30
 save(out/'protocol.json',dict(source=str(a.source),seed=seed,case=i,branch_step=a.step,mode=a.mode,commanded_color=int(ids[i]),reply_state=ref['snapshots'][i]['decision']['state'],horizon_steps=180,controller_sha256=sha(path),source_sha256=sha(__file__),actions_sha256=sha(a.source/'actions.pt'),cloud_calls=0,
  scope='Two previously inspected drop cases. Exact recorded physical prefix, same measured controller input at reply, no future Astra replies. Development intervention, not semantic success or independent validation.'))
 ih=ph=None;dropped=False;ready_count=consecutive=max_consecutive=0;trace=[];started=time.time();writer=RolloutVideo(out/'recovery.mp4')
 with torch.inference_mode():
  for step in range(a.step+180):
   if step<a.step:
    if step%2==0:symbols.update();raw.capture_camera(refresh=True)
    env.step(actions[step])
    if step%100==0:save(out/'progress.json',dict(status='replaying',step=step,branch=a.step,wall_s=time.time()-started))
    continue
   if step==a.step:
    actual=dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,cur_targets=raw.cur_targets,actions=raw.actions,prev_actions=raw.prev_actions)
    errors={k:float((v.cpu()-ref['physical'][k]).abs().max()) for k,v in actual.items()}
    save(out/'prefix_audit.json',dict(errors=errors,maximum=max(errors.values())))
    if max(errors.values())>1e-6:raise RuntimeError('Recorded action replay diverged')
    ih=ref['physical']['ih'].to(a.device);ph=ref['physical']['ph'].to(a.device);prop=ph[:,-1]
    for key,value in ref['controller'].items():setattr(runtime,key,value.to(a.device) if torch.is_tensor(value) else copy.deepcopy(value))
    torch.set_rng_state(ref['torch_rng']);torch.cuda.set_rng_state(ref['cuda_rng'])
    camera,rgb,_,_,_=sensor_frame(raw,symbols)
   elif step%2==0:
    camera,rgb,f,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,f,pack['history']);ph=history_append(ph,prop,pack['history']);runtime.update_visual(compact)
   proposal=student(VisualStudentBatch(ih,text[ids],ph))['action'];current=normalized_target(raw)
   pending=torch.zeros(n,dtype=torch.bool,device=a.device);known=torch.ones_like(pending);delivery=[False]*n;delivery[i]=step==a.step
   age=torch.zeros(n,device=a.device);age[i]=arrival_age if step==a.step else 0
   base,diag=runtime.action(step,text[ids],prop,proposal,current,pending,known,delivery,age)
   delta=step-a.step
   if a.mode=='smooth' and delta<60:
    rate=.15+.85*delta/60;base[i]=current[i]+rate*(proposal[i]-current[i])
   action=actions[step].clone();action[i]=current[i] if dropped else base[i]
   if step==a.step:
    action_delta=float((action[i]-actions[step,i]).abs().max())
    save(out/'first_action_audit.json',dict(max_abs_difference_from_recorded=action_delta,original_requires_match=a.mode=='original'))
    if a.mode=='original' and action_delta>1e-6:raise RuntimeError('Original first action did not reproduce recorded control')
   runtime.commit(action)
   if delta%2==0:writer.append(rgb[i].cpu().numpy())
   _,_,term,trunc,info=env.step(action);error=float(pose_errors(raw.object_rot,ids)[i]);dropped=dropped or bool(info['semantic_metrics']['dropped'][i])
   ready=not dropped and error<=.16;ready_count+=ready;consecutive=consecutive+1 if ready else 0;max_consecutive=max(max_consecutive,consecutive)
   trace.append(dict(offset=delta,error=error,drop=dropped,ready=ready,rate=float(diag['rate'][i])))
   if delta%30==0:save(out/'progress.json',dict(status='recovering',step=step,drop=dropped,ready_steps=ready_count))
   if dropped:break
 save(out/'trace.json',trace);save(out/'result.json',dict(status='completed',mode=a.mode,case=i,branch_step=a.step,drop=dropped,first_drop_offset=trace[-1]['offset']+1 if dropped else None,ready_steps=ready_count,ready_fraction_nominal=ready_count/180,max_consecutive_ready=max_consecutive,ever_stable_ready=max_consecutive>=6,final_ready=trace[-1]['ready'],cloud_calls=0))
except BaseException:save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if writer:writer.close()
 if env is not None:env.close()
 app.close()
