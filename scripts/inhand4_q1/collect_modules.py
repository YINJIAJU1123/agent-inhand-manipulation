"""Object-specific maintenance training trajectories. No formal outcomes or live API calls.

Uses published Actor labels for pending anchor holding and six counterfactual
canonical goals. Sensor inputs remain noisy. True poses appear only in labels.
"""
import pyglet
pyglet.options['headless']=True
import argparse,json,sys,time
from pathlib import Path
import torch
from runtime import DiceRuntime,add_arguments,sha
from viserdex.utils.runtime import add_runtime_args,parse_runtime_args
from viserdex.supermimic.simulation import TASK_ID,tensor
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.core import quat_mul,quat_unique,quat_conjugate,GOAL_ROT,HOLD
from viserdex.distillation.noise import pose_features
p=argparse.ArgumentParser(description=__doc__);add_arguments(p)
p.add_argument('--steps',type=int,default=2400);p.add_argument('--stride',type=int,default=4)
p.add_argument('--split',choices=('train','validation'),required=True)
add_runtime_args(p,TASK_ID);a=parse_runtime_args(p)
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True);runtime=None
try:
 runtime=DiceRuntime(a);r=runtime;dev=r.device;n=r.raw.num_envs
 teacher=Actor(r.student.config).to(dev).eval().requires_grad_(False);teacher.load_state_dict(r.payload['actor'],strict=True)
 r.reset(a.seed);episode=torch.zeros(n,dtype=torch.long,device=dev);behavior=torch.arange(n,device=dev)%4
 anchor_q=tensor(r.raw.scene['object'].data.root_quat_w).clone();frames=[];drop_count=0;started=time.monotonic()
 def counterfactual(qworld):
  h=r.adapter.history.clone()
  qbase=quat_unique(quat_mul(r.adapter.base_inverse,qworld))
  q=qbase[:,None].expand(-1,10,-1)
  h[...,105:109]=q
  h[...,113:124]=pose_features(h[...,113:116],h[...,116:120],q)
  return teacher(h)
 manifest=dict(format='inhand4_control_collection_v1',split=a.split,seed=a.seed,checkpoint_sha256=sha(a.checkpoint),teacher_sha256=sha(a.source_checkpoint),
  goal_bank=r.goal_manifest,runtime=r.metadata,steps=a.steps,stride=a.stride,num_envs=n,
  teacher_hold='Frozen original Actor conditioned on request-time true orientation, only supervision',
  teacher_execution='Frozen Actor, same noisy physical/action history with six counterfactual constant goal histories',
  behaviors=['continue','fixed_slow_0.15','anchor_action','teacher_anchor'],
  scope='scripted pending periods on training seeds, not live Astra evaluation',sources={x:sha(Path(__file__).with_name(x)) for x in ['collect_modules.py','runtime.py']})
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with torch.no_grad():
  for step in range(a.steps):
   # All worlds share a public goal and phase, with different randomized physics.
   cycle=step//200;phase=step%200;pending=phase<100;known=cycle%2==1
   if phase==0:
    r.set_skill(cycle%6);s=r.observe();r.anchor_pose=s['measured_pose'].clone();r.anchor_action=s['current'].clone()
    anchor_q=tensor(r.raw.scene['object'].data.root_quat_w).clone()
   values=r.values(pending,known,phase*.05 if pending else 0.)
   hold=counterfactual(anchor_q);proposal=values['proposal'];current=values['current']
   action=proposal.clone()
   if pending:
    mask=behavior==1;action[mask]=current[mask]+.15*(proposal[mask]-current[mask])
    mask=behavior==2;action[mask]=r.anchor_action[mask]
    mask=behavior==3;action[mask]=hold[mask]
   sample=None
   if step%a.stride==0:
    labels=torch.stack([counterfactual(r.rotations[g:g+1].expand(n,-1)) for g in range(6)],1)
    true_q=torch.nn.functional.normalize(tensor(r.raw.scene['object'].data.root_quat_w).double(),dim=-1)
    target=torch.nn.functional.normalize(r.rotations.double(),dim=-1)
    angles=2*torch.acos((true_q[:,None]*target[None]).sum(-1).abs().clamp(0,1))
    keys=['proprio','measured_pose','belief','proposal','current','anchor_pose','anchor_action','fit_error_m','known','pending','elapsed_pending_s']
    sample={k:values[k].detach().cpu().clone() for k in keys}
    sample.update(teacher_actions=labels.cpu(),teacher_hold_action=hold.cpu(),angles_rad=angles.float().cpu(),
      goal_index=torch.full((n,),r.active_goal,dtype=torch.long),episode_id=episode.cpu().clone(),env_id=torch.arange(n),
      sim_step=torch.full((n,),step,dtype=torch.long),episode_step=r.raw.episode_length_buf.cpu().clone(),behavior=behavior.cpu(),executed_action=action.cpu().clone())
   out=r.step(action);done=out['done'];drop_count+=int(out['drop'].sum());episode+=done
   if sample is not None:sample.update(next_done=done.cpu().clone(),next_drop=out['drop'].cpu().clone());frames.append(sample)
   if done.any():
    # Restore the public goal only for automatically reset training worlds.
    r.command.quat_command_w[done]=r.rotations[r.active_goal];r.command.time_left[done]=1e6
    r.command.hold_ticks[done]=0;r.command.just_completed[done]=False
    r.adapter.state[done,GOAL_ROT]=quat_unique(quat_mul(r.adapter.base_inverse[done],r.command.quat_command_w[done]));r.adapter.state[done,HOLD]=0
    s=r.observe();r.anchor_pose[done]=s['measured_pose'][done];r.anchor_action[done]=s['current'][done]
    anchor_q[done]=tensor(r.raw.scene['object'].data.root_quat_w)[done]
   if (step+1)%200==0:print(json.dumps(dict(step=step+1,samples=len(frames)*n,drops=drop_count,elapsed=time.monotonic()-started)),flush=True)
  data={k:torch.cat([f[k] for f in frames]) for k in frames[0]}
  torch.save(data,a.output/'samples_000.pt')
 (a.output/'completion.json').write_text(json.dumps(dict(completed=True,samples=len(data['goal_index']),drops=drop_count,elapsed=time.monotonic()-started,max_errors=r.adapter.max_errors),indent=2)+'\n')
finally:
 if runtime:runtime.close()
