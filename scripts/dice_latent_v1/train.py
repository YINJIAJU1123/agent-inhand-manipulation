"""Real Newton DAgger for the released dice actor's direct latent interface.

Only the training goal distribution changes. Physical configuration, noise,
action history, home-offset action semantics and reward audits are preserved.
"""
import argparse
from dataclasses import asdict
import hashlib,json,os,signal,time
from pathlib import Path
import torch
from viserdex.utils.runtime import SimulationRuntime,add_runtime_args,parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID,TransitionRecorderCfg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--env_config',type=Path,required=True)
p.add_argument('--bank',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--initial',type=Path)
p.add_argument('--num_envs',type=int,default=64)
p.add_argument('--iterations',type=int,default=200)
p.add_argument('--interface_iterations',type=int,default=60)
p.add_argument('--rollout_steps',type=int,default=32)
p.add_argument('--updates',type=int,default=16)
p.add_argument('--batch_size',type=int,default=256)
p.add_argument('--capacity',type=int,default=32768)
p.add_argument('--learning_rate',type=float,default=1e-4)
p.add_argument('--seed',type=int,default=186201)
p.add_argument('--evaluate',choices=('source','student'))
p.add_argument('--eval_steps',type=int,default=800)
p.add_argument('--collect',action='store_true',help='Save source-rollout distillation samples')
add_runtime_args(p,TASK_ID);args=parse_runtime_args(p)
if args.output.exists():raise FileExistsError(args.output)
if args.evaluate=='student' and not args.initial:p.error('Student evaluation requires --initial')
if args.evaluate!='source' and not args.bank:p.error('Latent student requires --bank')
if args.collect and args.evaluate!='source':p.error('Collection requires source actor')
if min(args.num_envs,args.iterations,args.rollout_steps,args.updates,args.batch_size,args.capacity)<1:p.error('Positive counts required')
payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
binding=SavedEnvironmentBinding(args.env_config)
if binding.contract_sha!=payload['metadata']['environment_contract']:raise ValueError('Released environment mismatch')
def configure(cfg):
    binding.configure(cfg,args)
    configure_clean_state_observations(cfg)
    cfg.recorders=TransitionRecorderCfg()
    cfg.scene.clone_in_fabric=False
    cfg.scene.robot.spawn.activate_contact_sensors=False
    cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False
    cfg.seed=args.seed
    binding.verify(cfg)
app=SimulationRuntime(args,configure=configure)
import gymnasium as gym
import numpy as np
from isaaclab_newton.physics import NewtonManager
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.core import quat_unique,quat_mul
from model import migrate,physical_history,configure_training,FORMAT
from goals import load_bank,install_training_goals,QUATERNIONS_XYZW,NORMALS

def main():
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True)
    model,source,migration=migrate(payload,args.seed)
    initial_pack=torch.load(args.initial,map_location='cpu',weights_only=False) if args.initial else None
    admitted_formats={FORMAT}
    if initial_pack and initial_pack['format']=='dice_supermimic_feature_latent_v3':
        if not args.evaluate:raise ValueError('Feature-adapter training uses its own feature distillation entrypoint')
        from feature_model import FeatureLatentActor
        model=FeatureLatentActor(source.config if hasattr(source,'config') else model.config)
        migration=initial_pack['metadata']['migration']
        admitted_formats.add(initial_pack['format'])
    env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
    device=env.device
    model.to(device);source.to(device)
    if args.evaluate=='source':
        vectors=None;quats=torch.tensor(QUATERNIONS_XYZW,device=device)
        goal_meta=dict(normals_body=NORMALS,target_quaternions_world_xyzw=QUATERNIONS_XYZW,latent_used=False)
    else:vectors,quats,goal_meta=load_bank(args.bank,device)
    if vectors is not None:model.configure_goal_statistics(vectors)
    if args.initial:
        initial=initial_pack
        if initial['format'] not in admitted_formats or initial['metadata']['goal_bank']['cache_sha256']!=goal_meta['cache_sha256']:
            raise ValueError('Initialization is not this latent architecture and goal bank')
        model.load_state_dict(initial['model'],strict=True)
    ids=install_training_goals(env.command_manager.get_term('object_pose'),quats)
    clean,_=env.reset()
    adapter=Adapter(env,PerceptionNoiseCfg(**payload['noise_config']),model.config.history)
    adapter.audit_observation(clean)
    goal_history=ids[:,None].expand(-1,model.config.history).clone()
    metadata=dict(migration=migration,goal_bank=goal_meta,source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        environment_contract=binding.contract_sha,noise_config=payload['noise_config'],seed=args.seed,
        num_envs=args.num_envs,pose_source='simulator with published sensor noise',
        goal_override='six canonical world-frame orientations; unchanged physical contract',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},pid=os.getpid())
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    metrics=dict(steps=0,success_events=0,drops=0,timeouts=0,action_mse_sum=0.)
    per_goal={n:dict(steps=0,success_events=0,drops=0) for n in range(6)}
    started=time.monotonic();hooks=[];optimizer=None;stage=None;physical_reference=None
    replay={};cursor=0;count=0
    collected=[]
    if not args.evaluate:
        replay=dict(physical=torch.empty(args.capacity,10,116,device=device),goals=torch.empty(args.capacity,10,dtype=torch.long,device=device),labels=torch.empty(args.capacity,21,device=device))
    def save(name,iteration):
        out=args.output/name;tmp=out.with_suffix('.tmp')
        torch.save(dict(format=FORMAT,model=model.state_dict(),config=asdict(model.config),metadata=metadata,iteration=iteration,
            stage=stage,optimizer=optimizer.state_dict() if optimizer else None,metrics=metrics),tmp)
        tmp.replace(out)
    try:
        total=args.eval_steps if args.evaluate else args.iterations*args.rollout_steps
        for step in range(total):
            iteration=step//args.rollout_steps
            next_stage='interface' if iteration<args.interface_iterations else 'adapt'
            if not args.evaluate and stage!=next_stage:
                for handle in hooks:handle.remove()
                stage=next_stage;hooks=configure_training(model,stage)
                optimizer=torch.optim.Adam([x for x in model.parameters() if x.requires_grad],lr=args.learning_rate,weight_decay=0)
                physical_reference=model.net[0].weight.detach().reshape(2048,10,-1)[...,:116].clone()
            with torch.no_grad():
                if step%32==0:
                    expected_goal=quat_unique(quat_mul(adapter.base_inverse[:,None].expand(-1,10,-1),quats[goal_history]))
                    torch.testing.assert_close(adapter.history[...,105:109],expected_goal,rtol=0,atol=2e-6)
                physical=physical_history(adapter.history)
                label=source(adapter.history)
                pred=label if args.evaluate=='source' else model(physical,vectors[goal_history])
                if not torch.isfinite(pred).all() or not torch.isfinite(label).all():raise FloatingPointError('Nonfinite action')
                if args.evaluate:action=label if args.evaluate=='source' else pred
                else:
                    beta=1. if stage=='interface' else max(.05,1.-(iteration-args.interface_iterations)/max(1,args.iterations-args.interface_iterations-1))
                    action=torch.where(torch.rand(args.num_envs,1,device=device)<beta,label,pred)
                    positions=(torch.arange(args.num_envs,device=device)+cursor)%args.capacity
                    replay['physical'][positions]=physical;replay['goals'][positions]=goal_history;replay['labels'][positions]=label
                    cursor=(cursor+args.num_envs)%args.capacity;count=min(args.capacity,count+args.num_envs)
                if args.collect:
                    collected.append((physical.cpu().clone(),goal_history.cpu().to(torch.uint8),label.cpu().clone(),adapter.history[..., [105,106,107,108,120,121,122,123]].cpu().clone()))
                active=ids.clone()
                record,_,_,clean=adapter.step(action)
                overflow=NewtonManager._solver.mjw_data.overflow.numpy()
                if int(np.bitwise_or.reduce(overflow.ravel())):raise RuntimeError('Physics solver capacity overflow')
                done=record['done']
                goal_history=torch.cat((goal_history[:,1:],ids[:,None]),1)
                goal_history[done]=ids[done,None].expand(-1,10)
                metrics['steps']+=args.num_envs
                metrics['success_events']+=int(record['success'].sum())
                metrics['drops']+=int(record['terminated'].sum())
                metrics['timeouts']+=int((done&~record['terminated']).sum())
                metrics['action_mse_sum']+=float((label-pred).square().mean(-1).sum())
                for g in range(6):
                    mask=active==g
                    per_goal[g]['steps']+=int(mask.sum());per_goal[g]['success_events']+=int((record['success']&mask).sum());per_goal[g]['drops']+=int((record['terminated']&mask).sum())
            if not args.evaluate and (step+1)%args.rollout_steps==0:
                losses=[]
                for _ in range(args.updates):
                    sample=torch.randint(count,(args.batch_size,),device=device)
                    prediction=model(replay['physical'][sample],vectors[replay['goals'][sample]])
                    loss=(prediction-replay['labels'][sample]).square().mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step();losses.append(float(loss))
                if stage=='interface':
                    torch.testing.assert_close(model.net[0].weight.detach().reshape(2048,10,-1)[...,:116],physical_reference,rtol=0,atol=0)
                row=dict(iteration=iteration+1,stage=stage,beta=beta,loss=sum(losses)/len(losses),elapsed=time.monotonic()-started,**metrics)
                with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if (iteration+1)%10==0:print(json.dumps(row),flush=True)
                if (iteration+1)%20==0:save('latest.pt',iteration+1)
            if stopping[0]:break
        report=dict(metrics=metrics,per_goal=per_goal,max_errors=adapter.max_errors,elapsed=time.monotonic()-started,
            mode=args.evaluate or 'mixed teacher/student training',complete=step+1==total,steps_executed=step+1,
            note='Continuous canonical-goal diagnostic, NOT formal semantic episode success rate')
        if args.collect:
            torch.save(dict(format='dice_source_distillation_v1',physical=torch.cat([x[0] for x in collected]),goals=torch.cat([x[1] for x in collected]),labels=torch.cat([x[2] for x in collected]),teacher_goal_features=torch.cat([x[3] for x in collected]),metadata=metadata),args.output/'samples.pt')
        if not args.evaluate:save('final.pt' if report['complete'] else 'interrupted.pt',iteration+1)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        env.close()
try:main()
finally:app.close()
