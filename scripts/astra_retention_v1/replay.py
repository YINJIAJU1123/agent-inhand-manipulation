"""Exact action replay of live Astra episodes. No cloud calls; labels for audit only."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/f) for f in ('agentic','semantic_completion_v1','astra_completion_v3','async_stability_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch
from common import *
from completion_runtime import DenseEvidenceRuntime
from semantic_protocol_v3 import make_cases
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from sensors import object_regions

def all_errors(q):
 s=2**-.5;targets=torch.tensor([[s,0,-s,0],[s,0,s,0],[s,s,0,0],[s,-s,0,0],[1,0,0,0],[0,1,0,0]],device=q.device)
 return 2*torch.acos((q[:,None]*targets[None]).sum(-1).abs().clamp(0,1))

env=None
try:
 src=Path(a.source);report=json.loads((src/'report.json').read_text());meta=report['metadata'];seed=meta['seed']
 actions=torch.load(src/'actions.pt',map_location='cpu',weights_only=True)
 recorded=torch.load(src/'response_states.pt',map_location='cpu',weights_only=False)
 student,text,pack=load_student(a.device)
 env,raw,symbols=environment(seed,meta['max_steps'],a.device)
 cases=make_cases(seed,0,meta['template']);layout=split_layouts(seed,'test')
 for i,c in enumerate(cases):
  raw.cfg.fixed_target_face=layout[i].index(SYMBOLS.index(c['target_symbol']));raw._reset_target_pose(torch.tensor([i],device=a.device))
 raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
 initial=dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout)
 assert initial==json.loads((src/'initial.json').read_text()),'Initial state mismatch'
 if meta['initialization']=='collision_clearance':
  offsets,audit=minimum_vertical_offsets(raw);state=raw.object.data.root_state_w.clone();state[:,2]+=torch.tensor(offsets,device=a.device)
  raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
 applied=dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist())
 assert applied==json.loads((src/'initial_applied.json').read_text()),'Applied initial state mismatch'
 evidence=DenseEvidenceRuntime(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt',a.device,6)
 rng=torch.load(src/'initial_rng.pt',weights_only=True);torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
 ih=ph=None;frames=[];labels=[];audits=[];started=time.time();rows=[];snaps=[dict(goal=None,goal_symbol=None,decision=dict(color=None,state='initial'))for _ in range(6)]
 with torch.inference_mode():
  for step,action in enumerate(actions):
   if step%2==0:
    camera,rgb,feat,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,feat,pack['history']);ph=history_append(ph,prop,pack['history'])
   if step in recorded:
    ref=recorded[step];snaps=ref['snapshots']
    actual=dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,cur_targets=raw.cur_targets,actions=raw.actions,prev_actions=raw.prev_actions)
    delta={k:float((v.cpu()-ref['physical'][k]).abs().max())for k,v in actual.items()}
    audits.append(dict(step=step,errors=delta));save(out/'replay_audit.json',dict(checked=len(audits),maximum=max(max(v['errors'].values())for v in audits),audits=audits))
    if max(delta.values())>1e-6:raise RuntimeError('Recorded action replay diverged at step '+str(step))
   if step%10==0:
    regions=object_regions(rgb,camera['depth']);x=evidence.encoder.candidates(ih,ph)
    predictions=evidence.model(x,regions['roi_rgb'],regions['roi_depth'],regions['roi_box'],regions['roi_found'])
    errors=all_errors(raw.object_rot)
    for i in range(6):
     if step>report['episodes'][i]['finish_step']:continue
     frames.append(dict(x=x[i].cpu(),**{k:v[i].cpu()for k,v in regions.items()}))
     labels.append(dict(error=errors[i].cpu(),rotation_label=raw.object_rot[i].cpu()))
     snap=snaps[i];rows.append(dict(seed=seed,case=i,step=step,source=str(src),skill=snap['decision']['color'],state=snap['decision']['state'],claimed_goal=snap['goal'],claimed_symbol=snap['goal_symbol'],probability=predictions['ordinal'][i,:,2].sigmoid().cpu().tolist(),target_color=report['episodes'][i]['target_color']))
   env.step(action.to(a.device))
   if step%100==0:
    save(out/'progress.json',dict(status='replaying',step=step,total_steps=len(actions),frames=len(rows),wall_s=time.time()-started));print(step,len(rows),flush=True)
 data={k:torch.stack([r[k]for r in frames])for k in frames[0]}
 data.update({k:torch.stack([r[k]for r in labels])for k in labels[0]});data['rows']=rows;torch.save(data,out/'data.pt')
 save(out/'manifest.json',dict(status='completed',source=str(src),frames=len(rows),cloud_calls=0,actions_sha256=sha(src/'actions.pt'),replay_max_error=max(max(v['errors'].values())for v in audits),source_sha256=sha(__file__),sensor_fields=list(frames[0]),supervision_only=['error','rotation_label','target_color'],note='Exact recorded actions with fresh RGB. No counterfactual success claim; used for failure analysis. These previously inspected seeds are development only.'))
 save(out/'progress.json',dict(status='completed',step=len(actions),frames=len(rows),wall_s=time.time()-started))
except BaseException:
 save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if env is not None:env.close()
 app.close()
