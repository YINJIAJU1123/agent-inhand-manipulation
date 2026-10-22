"""Completion supervision under public asynchronous control and goal switches.

RGB-D and robot measurements are model inputs. Exact object pose is saved only
as a supervised label. No high-level model or ground-truth action selector.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/x) for x in ('agentic','semantic_completion_v1','astra_runtime_v2','async_stability_v1','astra_retention_online_v1','astra_control_prefix_v1','semantic_completion_v3')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--split',choices=['train','calibration','audit','smoke'],required=True);p.add_argument('--max-steps',type=int,default=1200);p.add_argument('--cases',type=int,default=12)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch
from common import load_student,environment,sensor_frame,history_append,save,sha,STUDENT,ADAPTER,CONTROLLER
from evidence_runtime import CachedDenseEvidence
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from retention_runtime import RetentionRuntime
from latency_controller_v1 import normalized_target
from latent_data_v4 import split_layouts
from placement import minimum_vertical_offsets
from sensors import object_regions
from control_protocol import command_state
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
out=a.output;out.mkdir(parents=True,exist_ok=False);env=None
try:
    n=a.cases;assert n in (6,12,24)
    student,text,pack=load_student(a.device);env,raw,symbols=environment(a.seed,a.max_steps,a.device,num_envs=n)
    layout=split_layouts(a.seed,'train' if a.split=='train' else 'test',count=n);symbols.set_layout(layout)
    offsets,placement=minimum_vertical_offsets(raw);state=raw.object.data.root_state_w.clone();state[:,2]+=torch.tensor(offsets,device=a.device)
    raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers();raw.goal_markers.set_visibility(False)
    save(out/'placement.json',placement)
    evidence_path=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt';evidence=CachedDenseEvidence(evidence_path,a.device,n)
    controller=RecoveryController(616).to(a.device).eval();controller.load_state_dict(torch.load(CONTROLLER,map_location=a.device,weights_only=True)['model']);controller.requires_grad_(False)
    cold=normalized_target(raw).clone();runtime=ControllerRuntime(controller,cold,False);hold=RetentionRuntime(n,mode='hold',threshold=.7)
    # Predeclared case groups are interventions, not sensor or outcome features.
    startup=torch.tensor([(i+a.seed)%2==0 for i in range(n)],device=a.device)
    hold_group=torch.tensor([(i//2+a.seed)%2==0 for i in range(n)],device=a.device)
    active=torch.ones(n,dtype=torch.bool,device=a.device);ih=ph=None;frames=[];rows=[];trace=[];endings={};actions=[];started=time.monotonic()
    with torch.inference_mode():
        for step in range(a.max_steps):
            command=command_state(step,n);colors=command['colors'];known=torch.full((n,),colors[0] is not None,device=a.device,dtype=torch.bool)
            ids=torch.tensor([c if c is not None else 1 for c in colors],device=a.device)
            if step%2==0:
                camera,rgb,f,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,f,pack['history']);ph=history_append(ph,prop,pack['history']);runtime.update_visual(compact)
            if step%10==0:
                evidence.update(rgb,ih,ph,camera['depth']);regions=object_regions(rgb,camera['depth']);x=evidence.encoder.candidates(ih,ph)
                s=2**-.5;target=raw.object_rot.new_tensor([[s,0,-s,0],[s,0,s,0],[s,s,0,0],[s,-s,0,0],[1,0,0,0],[0,1,0,0]])
                errors=2*torch.acos((raw.object_rot[:,None]*target[None]).sum(-1).abs().clamp(0,1))
                segment=0 if step<180 else 1 if step<420 else 2 if step<660 else 3
                for i in active.nonzero().flatten().tolist():
                    frames.append(dict(x=x[i].cpu(),error=errors[i].cpu(),rotation_label=raw.object_rot[i].cpu(),**{k:v[i].cpu() for k,v in regions.items()}))
                    rows.append(dict(seed=a.seed,case=i,step=step,skill=int(ids[i]),segment=segment,phase=command['phase'],pending=command['pending'],split=a.split))
            goals=text[ids];proposal=student(VisualStudentBatch(ih,goals,ph))['action']
            current=normalized_target(raw);pending=active & command['pending'];age=torch.full((n,),command['age'],device=a.device)
            base,_=runtime.action(step,goals,prop,proposal,current,pending,known|startup,[command['delivered']]*n,age)
            retained=hold.apply(step=step,states=['execute' if command['phase']=='inspect' else command['phase']]*n,
                colors=colors,pending=pending,delivered=[command['delivered']]*n,request_ids=[command['request_id']]*n,active=active,
                ready_probability=evidence.last,feature_bank=evidence.retention_features,fresh=step%10==0,student_action=proposal,current=current,base_action=base)
            action=torch.where(hold_group[:,None],retained,base)
            action[~known & ~startup]=cold[~known & ~startup];action[~active]=current[~active]
            if not torch.isfinite(action).all():raise RuntimeError('Nonfinite action')
            runtime.commit(action);actions.append(action.cpu().clone());_,_,term,trunc,info=env.step(action)
            done=term|trunc|info['semantic_metrics']['dropped'].bool()
            for i in (active & done).nonzero().flatten().tolist():endings[i]=dict(step=step+1,drop=bool(info['semantic_metrics']['dropped'][i]),phase=command['phase'])
            active &= ~done
            if step%100==0:save(out/'progress.json',dict(status='collecting',step=step,frames=len(rows),active=int(active.sum()),wall_s=time.monotonic()-started))
            if not active.any():break
    data={k:torch.stack([r[k] for r in frames]) for k in frames[0]};data['rows']=rows;torch.save(data,out/'data.pt');torch.save(torch.stack(actions),out/'actions.pt')
    save(out/'manifest.json',dict(status='completed',seed=a.seed,split=a.split,frames=len(rows),cases=n,endings=endings,
        startup_groups=startup.cpu().tolist(),hold_groups=hold_group.cpu().tolist(),ready_counts=(data['error']<=.16).sum(0).tolist(),near_counts=((data['error']>.16)&(data['error']<=.4)).sum(0).tolist(),
        student_sha256=sha(STUDENT),adapter_sha256=sha(ADAPTER),controller_sha256=sha(CONTROLLER),evidence_sha256=sha(evidence_path),
        data_sha256=sha(out/'data.pt'),source_sha256=sha(__file__),cloud_calls=0,
        sensor_fields=['x','roi_rgb','roi_depth','roi_box','roi_found'],supervision_only_fields=['error','rotation_label'],
        scope='Public commands with continuous physics, delayed replies, two inspections and subsequent execution; fixed startup and hold interventions. No semantic-task success claim.'))
    save(out/'progress.json',dict(status='completed',frames=len(rows),wall_s=time.monotonic()-started))
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
