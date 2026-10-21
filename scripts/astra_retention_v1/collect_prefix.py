"""Independent, balanced known-goal prefixes for learning retention decisions.

A simulated correct goal is delivered at six seconds. It is not an Astra run.
The trigger uses frozen sensor evidence; exact pose only labels training outcomes.
"""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand')]+[str(ROOT/'scripts'/f)for f in ('agentic','semantic_completion_v1','astra_completion_v3','astra_runtime_v2','async_stability_v1')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--cases',type=int,default=24);p.add_argument('--split',choices=['train','calibration','audit'],required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;assert a.cases%6==0
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from common import *
from completion_runtime import DenseEvidenceRuntime
from placement import minimum_vertical_offsets
from latency_controller_v1 import normalized_target
from recovery_controller_v3 import RecoveryController
from runtime import ControllerRuntime
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch

env=None
try:
 n=a.cases;max_steps=1200;student,text,pack=load_student(a.device)
 controller=RecoveryController(616).to(a.device).eval();controller.load_state_dict(torch.load(CONTROLLER,map_location=a.device,weights_only=True)['model']);controller.requires_grad_(False)
 env,raw,symbols=environment(a.seed,max_steps,a.device,num_envs=n)
 rng=np.random.default_rng(a.seed+115);layout=np.stack([rng.permutation(6)for _ in range(n)]).tolist();targets=rng.permutation(np.tile(np.arange(6),n//6)).tolist()
 for i,c in enumerate(targets):raw.cfg.fixed_target_face=c;raw._reset_target_pose(torch.tensor([i],device=a.device))
 raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
 save(out/'initial.json',dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout))
 offsets,placement=minimum_vertical_offsets(raw);save(out/'placement.json',placement)
 state=raw.object.data.root_state_w.clone();state[:,2]+=torch.tensor(offsets,device=a.device);raw.object.write_root_state_to_sim(state);raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw._compute_intermediate_values();raw._update_face_markers()
 save(out/'initial_applied.json',dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist()))
 evidence=DenseEvidenceRuntime(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt',a.device,n)
 cold=normalized_target(raw).clone();runtime=ControllerRuntime(controller,cold,False)
 ih=ph=None;active=torch.ones(n,dtype=torch.bool,device=a.device);actions=[];states={};plan=[];rows=[dict(case=i,target_color=c,target_symbol=SYMBOLS[layout[i][c]],reason='untriggered',finish_step=max_steps)for i,c in enumerate(targets)]
 torch.save(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'initial_rng.pt');started=time.time()
 with torch.inference_mode():
  for step in range(900):
   if step%2==0:
    camera,rgb,f,compact,prop=sensor_frame(raw,symbols);ih=history_append(ih,f,pack['history']);ph=history_append(ph,prop,pack['history']);runtime.update_visual(compact)
   if step%10==0:
    evidence.update(rgb,ih,ph,camera['depth'])
    if step>=180:
     ready=sum(bool(active[i])and evidence.can_stop(i,c)for i,c in enumerate(targets))
     # A shared intervention time keeps every pre-branch action identical across
     # vectorized counterfactuals. Retain not-ready states for failure training.
     if ready>=max(1,n//2) or step>=720:
      for i,c in enumerate(targets):
       if active[i]:
        plan.append(dict(source=str(out),case=i,step=step,goal=c));active[i]=False;rows[i].update(reason='shared_sensor_branch',finish_step=step)
   if step%10==0:
    physical={k:v.detach().cpu().clone()for k,v in dict(joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets,cur_targets=raw.cur_targets,actions=raw.actions,prev_actions=raw.prev_actions).items()}
    states[step]=dict(physical=physical)
   ids=torch.tensor(targets if step>=180 else [0]*n,device=a.device);goals=text[ids];proposal=student(VisualStudentBatch(ih,goals,ph))['action'];current=normalized_target(raw)
   known=torch.full((n,),step>=180,device=a.device,dtype=torch.bool);pending=active&~known;delivered=[step==180]*n;age=torch.full((n,),min(step,180)/30 if step<=180 else 0.,device=a.device)
   command,_=runtime.action(step,goals,prop,proposal,current,pending,known,delivered,age);command[~known]=cold[~known];command[~active]=current[~active]
   runtime.commit(command);actions.append(command.cpu());_,_,term,trunc,info=env.step(command)
   for i in range(n):
    if active[i] and bool(term[i]or trunc[i]or info['semantic_metrics']['dropped'][i]):active[i]=False;rows[i].update(reason='prefix_drop',finish_step=step+1)
   if step%100==0:save(out/'progress.json',dict(status='collecting',step=step,active=int(active.sum()),triggered=len(plan),wall_s=time.time()-started));print(step,int(active.sum()),len(plan),flush=True)
   if not active.any():break
 torch.save(torch.stack(actions),out/'actions.pt');torch.save(states,out/'response_states.pt');save(out/'sensor_branch_plan.json',plan)
 save(out/'report.json',dict(metadata=dict(seed=a.seed,template='canonical',initialization='collision_clearance',max_steps=max_steps,num_envs=n,offline_known_goal=True,split=a.split),episodes=rows,cloud_calls=0,source_sha256=sha(__file__),student_sha256=sha(STUDENT),controller_sha256=sha(CONTROLLER),note='Balanced supplied goals, no external agent. All prefixes retained, including drops and no trigger. No task-success claim.'))
 save(out/'progress.json',dict(status='completed',triggered=len(plan),cases=n,wall_s=time.time()-started))
except BaseException:
 save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if env is not None:env.close()
 app.close()
