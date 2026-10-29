"""Fit maintenance/evidence heads from new-student trajectories, without formal trials."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from control_modules import features, FIELDS, TOLERANCES, CompletionEvidence, MaintenanceActor
from goals import load_bank

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    completion = json.loads((directory/'completion.json').read_text())
    if not completion['completed']:
        raise ValueError('Collection not completed')
    keys = set(FIELDS) | {'angles_rad','goal_index','teacher_actions','teacher_hold_action','episode_id','env_id','sim_step','behavior'}
    parts = []
    for path in sorted(directory.glob('samples_*.pt')):
        pack = torch.load(path,map_location='cpu',weights_only=True)
        parts.append({key:pack[key] for key in keys})
    data = {key:torch.cat([part[key] for part in parts]) for key in keys}
    if len(data['goal_index']) != completion['samples']:
        raise ValueError('Incomplete data shards')
    data['sensor'] = features(data)
    return manifest, data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--text_cache',type=Path,required=True)
    parser.add_argument('--original_asset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1200)
    parser.add_argument('--actor_goal_input',choices=('active','center'),default='active',
        help='Center retrains without explicit goal. Teacher and student goals remain correct.')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execution_speed_mode',choices=('fixed','fixed_matched','learned'),default='fixed',
        help='Fixed_matched retains the same extra head but keeps execution speed exactly one.')
    parser.add_argument('--actor_warm_start',type=Path)
    parser.add_argument('--paired_execution',type=Path,help='Dense same-state alternative-goal features for execution supervision')
    parser.add_argument('--balance_binding_state',action='store_true',
                        help='Balance semantic binding flags for identical physical action targets')
    parser.add_argument('--balance_execution_phases',action='store_true',
                        help='Balance unknown-goal inspection and known-goal execution states in the actor batch.')
    parser.add_argument('--align_actor_elapsed',action='store_true',
                        help='Use runtime zero elapsed-pending time for idle actor inputs; evidence fitting is unchanged.')
    parser.add_argument('--balance_evidence_goals',action='store_true',
                        help='Draw completion query goals uniformly within each binary class, with an independent RNG.')
    parser.add_argument('--evidence_replay',type=Path,
                        help='Replace half the evidence batch with aligned states from an earlier training capture.')
    parser.add_argument('--boundary_negatives',action='store_true',
                        help='Draw half the evidence-negative batch from (0.16, 0.32] rad using an independent RNG.')
    parser.add_argument('--seed',type=int,default=182841)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device=args.device
    bank,_,goal_manifest=load_bank(args.text_cache,args.original_asset,device)
    train_manifest,train=load_data(args.data/'train')
    val_manifest,val=load_data(args.data/'validation')
    if train_manifest['seed']==val_manifest['seed']:raise ValueError('Train/validation seed overlap')
    for key in ('checkpoint_sha256','teacher_sha256','goal_bank'):
        if train_manifest[key]!=val_manifest[key]:raise ValueError(f'Split contract differs: {key}')
    if train_manifest['goal_bank']['cache_sha256']!=goal_manifest['cache_sha256']:
        raise ValueError('Goal bank changed')
    mean=train['sensor'].mean(0).to(device)
    scale=train['sensor'].std(0).clamp_min(.01).to(device)
    evidence=CompletionEvidence(mean,scale,bank).to(device)
    actor=MaintenanceActor(mean.clone(),scale.clone(),bank,goal_input=args.actor_goal_input,
        execution_speed_mode=args.execution_speed_mode).to(device)
    if args.actor_warm_start:
        warm=torch.load(args.actor_warm_start,map_location='cpu',weights_only=True)
        if warm['manifest']['source_student_sha256']!=train_manifest['checkpoint_sha256']:
            raise ValueError('Warm-start actor student mismatch')
        torch.testing.assert_close(warm['bank'],bank.cpu(),atol=0,rtol=0)
        if args.execution_speed_mode!='fixed' and warm['manifest'].get('actor_execution_speed_mode','fixed')=='fixed':
            missing,unexpected=actor.load_state_dict(warm['model'],strict=False)
            if set(missing)!={'execution_speed_head.weight','execution_speed_head.bias'} or unexpected:
                raise ValueError('Unexpected warm-start incompatibility')
        else:
            actor.load_state_dict(warm['model'])
    optimizers={name:torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001)
                for name,model in [('evidence',evidence),('actor',actor)]}
    manifest=dict(format='pose_latent_control_modules_v1',seed=args.seed,updates=args.updates,
        actor_goal_input=args.actor_goal_input,actor_execution_speed_mode=args.execution_speed_mode,
        source_student_sha256=train_manifest['checkpoint_sha256'],goal_manifest=goal_manifest,
        train_collection=train_manifest,val_collection=val_manifest,
        observation_fields=list(FIELDS),feature_dim=len(mean),input_scope='measured sensors and public controller state only',
        actor_warm_start_sha256=sha(args.actor_warm_start) if args.actor_warm_start else None,
        fitting='supervised frozen-teacher DAgger; paired six-goal completion supervision',
        balance_binding_state=args.balance_binding_state,
        balance_execution_phases=args.balance_execution_phases,
        actor_idle_elapsed_zero=args.align_actor_elapsed,
        evidence_goal_balanced=args.balance_evidence_goals,
        evidence_boundary_negatives=args.boundary_negatives,
        evidence_negative_sampling='half forced to (0.16, 0.32] rad; half uniform over all negatives' if args.boundary_negatives else 'uniform over all negatives',
        binding_augmentation_scope='Actor only; same sensors, active latent and teacher action under either semantic-binding flag',
        scope='offline warmup, not admitted to autonomous agent evaluation',
        formal_episodes_used_for_selection=False,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('train_modules.py','control_modules.py','goals.py')})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    train={key:value.to(device) for key,value in train.items()}
    val={key:value.to(device) for key,value in val.items()}
    pairs=(train['angles_rad']<=.16).nonzero()
    negatives=(train['angles_rad']>.16).nonzero()
    if len(pairs)<50 or len(negatives)<50:raise ValueError('Insufficient physical completion classes')
    boundary_negatives=((train['angles_rad']>.16)&(train['angles_rad']<=.32)).nonzero()
    if args.boundary_negatives and len(boundary_negatives)<50:
        raise ValueError('Insufficient boundary-negative completion examples')
    boundary_rng=torch.Generator(device=device).manual_seed(args.seed+3000000)
    evidence_goal_rng=torch.Generator(device=device).manual_seed(args.seed+5000000)
    goal_pools=None
    if args.balance_evidence_goals:
        goal_pools=[[pool[pool[:,1]==goal] for goal in range(6)] for pool in (pairs,negatives)]
        if min(len(pool) for pools in goal_pools for pool in pools)<50:
            raise ValueError('Insufficient completion examples for a balanced query goal')
    pending_ids=train['pending'].nonzero().flatten()
    active_ids=(~train['pending']).nonzero().flatten()
    execution_pools=[((~train['pending'])&(train['known']==known)).nonzero().flatten() for known in (False,True)]
    execution_rng=torch.Generator(device=device).manual_seed(args.seed+6000000)
    if args.balance_execution_phases and min(len(pool) for pool in execution_pools)<50:
        raise ValueError('Insufficient inspection or bound execution states')
    manifest['actor_phase_sampling']=dict(pending_states=len(pending_ids),
        unknown_execution_states=len(execution_pools[0]),known_execution_states=len(execution_pools[1]),
        batch_fractions=[.5,.25,.25] if args.balance_execution_phases else None,
        scope='Actor only; evidence fitting and original global RNG draws unchanged')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if min(len(pending_ids),len(active_ids))<50:raise ValueError('Insufficient wait/execution coverage')
    paired=None
    pair_rng=torch.Generator(device=device).manual_seed(args.seed+1000000)
    binding_rng=torch.Generator(device=device).manual_seed(args.seed+2000000)
    binding_offset=sum(train[key].shape[-1] if train[key].ndim>1 else 1
                       for key in FIELDS[:FIELDS.index('known')])
    elapsed_offset=sum(train[key].shape[-1] if train[key].ndim>1 else 1
                       for key in FIELDS[:FIELDS.index('elapsed_pending_s')])
    replay=None
    replay_rng=torch.Generator(device=device).manual_seed(args.seed+4000000)
    if args.evidence_replay:
        replay_manifest,replay_data=load_data(args.evidence_replay/'train')
        if replay_manifest['seed'] in (train_manifest['seed'],val_manifest['seed']):
            raise ValueError('Replay must come from a distinct training seed')
        for key in ('checkpoint_sha256','teacher_sha256','goal_bank'):
            if replay_manifest[key]!=train_manifest[key]:
                raise ValueError(f'Replay contract differs: {key}')
        replay_sensor=replay_data['sensor']
        replay_sensor[~replay_data['pending'],elapsed_offset]=0.
        replay={'sensor':replay_sensor.to(device),'angles_rad':replay_data['angles_rad'].to(device)}
        replay['positive']=(replay['angles_rad']<=.16).nonzero()
        replay['negative']=(replay['angles_rad']>.16).nonzero()
        if min(len(replay['positive']),len(replay['negative']))<50:
            raise ValueError('Insufficient replay completion classes')
        manifest['evidence_replay']=dict(seed=replay_manifest['seed'],states=len(replay_sensor),fraction=.5,
            collection_sha256=sha(args.evidence_replay/'train/manifest.json'),idle_elapsed_zero=True,
            scope='Evidence only; original actor and global sampling RNG unchanged')
        del replay_data,replay_sensor
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.paired_execution:
        pair_manifest=json.loads((args.paired_execution/'manifest.json').read_text())
        if not pair_manifest['completed'] or pair_manifest['checkpoint_sha256']!=train_manifest['checkpoint_sha256']:
            raise ValueError('Invalid counterfactual feature provenance')
        paired=torch.load(args.paired_execution/'train.pt',map_location='cpu',weights_only=True)
        for key in ('sim_step','env_id','episode_id','goal_index'):
            torch.testing.assert_close(paired[key],train[key].cpu(),atol=0,rtol=0)
        if paired['source_manifest_sha256']!=sha(args.data/'train/manifest.json'):
            raise ValueError('Counterfactual features belong to a different collection')
        paired={k:paired[k].to(device) for k in ('features','proposals')}
        manifest['paired_execution_manifest_sha256']=sha(args.paired_execution/'manifest.json')
        manifest['actor_supervision']='Same sensor state and recurrent history, uniformly sampled correct goal during execution; original anchor holding targets during pending requests'
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tolerances=bank.new_tensor(TOLERANCES)
    start=time.monotonic()
    with (args.output/'training.jsonl').open('w') as log:
        for update in range(1,args.updates+1):
            p=pairs[torch.randint(len(pairs),(256,),device=device)]
            n=negatives[torch.randint(len(negatives),(256,),device=device)]
            if goal_pools is not None:
                for selected_class,pools in zip((p,n),goal_pools):
                    goals=torch.randint(6,(len(selected_class),),device=device,generator=evidence_goal_rng)
                    for goal,pool in enumerate(pools):
                        mask=goals==goal
                        selected_class[mask]=pool[torch.randint(len(pool),(int(mask.sum()),),
                            device=device,generator=evidence_goal_rng)]
            if args.boundary_negatives:
                count=len(n)//2
                n[:count]=boundary_negatives[torch.randint(len(boundary_negatives),(count,),
                    device=device,generator=boundary_rng)]
            selected=torch.cat((p,n))
            index,goal=selected.unbind(-1)
            evidence_sensor=train['sensor'][index]
            evidence_goal=goal.clone()
            error=train['angles_rad'][index,goal]
            if replay is not None:
                selections=[]
                for key in ('positive','negative'):
                    pool=replay[key]
                    selections.append(pool[torch.randint(len(pool),(128,),device=device,generator=replay_rng)])
                replay_index,replay_goal=torch.cat(selections).unbind(-1)
                evidence_sensor[::2]=replay['sensor'][replay_index]
                evidence_goal[::2]=replay_goal
                error[::2]=replay['angles_rad'][replay_index,replay_goal]
            result=evidence(evidence_sensor,bank[evidence_goal])
            binary=(error[:,None]<=tolerances).float()
            loss_e=F.binary_cross_entropy_with_logits(result['ordinal'],binary)+.2*F.smooth_l1_loss(result['log_error'],torch.log(error+.02))
            optimizers['evidence'].zero_grad(set_to_none=True)
            loss_e.backward()
            torch.nn.utils.clip_grad_norm_(evidence.parameters(),1.,error_if_nonfinite=True)
            optimizers['evidence'].step()
            index=torch.cat((pending_ids[torch.randint(len(pending_ids),(256,),device=device)],
                             active_ids[torch.randint(len(active_ids),(256,),device=device)]))
            if args.balance_execution_phases:
                for batch_start,pool in zip((256,384),execution_pools):
                    index[batch_start:batch_start+128]=pool[torch.randint(len(pool),(128,),device=device,generator=execution_rng)]
            goal=train['goal_index'][index].clone()
            pending=train['pending'][index]
            sensor=train['sensor'][index]
            proposal=train['proposal'][index]
            if paired is not None:
                active=~pending
                goal[active]=torch.randint(6,(int(active.sum()),),device=device,generator=pair_rng)
                sensor=sensor.clone();proposal=proposal.clone()
                sensor[active]=paired['features'][index[active],goal[active]]
                proposal[active]=paired['proposals'][index[active],goal[active]]
            if args.balance_binding_state:
                sensor=sensor.clone()
                sensor[:,binding_offset]=torch.randint(2,(len(index),),device=device,generator=binding_rng).float()
            if args.align_actor_elapsed:
                sensor=sensor.clone()
                sensor[:,elapsed_offset]=torch.where(pending,sensor[:,elapsed_offset],0.)
            result=actor(sensor,bank[goal],proposal,
                         train['proprio'][index,:21].clamp(-1,1),pending)
            target=torch.where(pending[:,None],train['teacher_hold_action'][index],
                               train['teacher_actions'][index,goal]).clamp(-1,1)
            loss_a=F.mse_loss(result['action'],target)+.01*result['residual'].square().mean()
            optimizers['actor'].zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
            optimizers['actor'].step()
            if update==1 or update%100==0:
                row=dict(update=update,evidence_loss=float(loss_e.detach()),actor_loss=float(loss_a.detach()),
                         elapsed_seconds=time.monotonic()-start)
                log.write(json.dumps(row)+'\n');log.flush()
                (args.output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
                print(json.dumps(row),flush=True)
    evidence.eval();actor.eval()
    predictions=[]
    actor_error=[];baseline_error=[]
    binding_errors={False:[],True:[]};binding_deltas=[]
    with torch.inference_mode():
        for indices in torch.arange(len(val['sensor']),device=device).split(256):
            x=val['sensor'][indices]
            logits=evidence(x[:,None].expand(-1,6,-1).reshape(-1,len(mean)),
                            bank[None].expand(len(indices),-1,-1).reshape(-1,768))['ordinal'][:,2]
            predictions.append(logits.sigmoid().reshape(-1,6))
            goal=val['goal_index'][indices];pending=val['pending'][indices]
            current=val['proprio'][indices,:21].clamp(-1,1)
            proposal=val['proposal'][indices]
            actor_x=x
            if args.align_actor_elapsed:
                actor_x=x.clone()
                actor_x[:,elapsed_offset]=torch.where(pending,actor_x[:,elapsed_offset],0.)
            pred=actor(actor_x,bank[goal],proposal,current,pending)['action']
            baseline=torch.where(pending[:,None],current+.15*(proposal.clamp(-1,1)-current),proposal.clamp(-1,1))
            target=torch.where(pending[:,None],val['teacher_hold_action'][indices],
                               val['teacher_actions'][indices,goal]).clamp(-1,1)
            actor_error.append((pred-target).square().mean(-1))
            baseline_error.append((baseline-target).square().mean(-1))
            if args.balance_binding_state:
                actions=[]
                for flag in (False,True):
                    counterfactual=actor_x.clone();counterfactual[:,binding_offset]=float(flag)
                    action=actor(counterfactual,bank[goal],proposal,current,pending)['action']
                    binding_errors[flag].append((action-target).square().mean(-1))
                    actions.append(action)
                binding_deltas.append((actions[0]-actions[1]).abs())
    probability=torch.cat(predictions)
    truth=val['angles_rad']<=.16
    # Whole env/episode groups are assigned to calibration or reporting.
    groups=list(zip(val['env_id'].cpu().tolist(),val['episode_id'].cpu().tolist()))
    assignments={group:int.from_bytes(hashlib.sha256(f"{val_manifest['seed']}:{group[0]}:{group[1]}".encode()).digest()[:4],'little')%2==0 for group in set(groups)}
    calibration=torch.tensor([assignments[group] for group in groups],device=device,dtype=torch.bool)
    report=~calibration
    candidates=[]
    for threshold in torch.linspace(.5,.999,200,device=device):
        accepted=(probability>=threshold)&calibration[:,None]
        count=int(accepted.sum())
        correct=int((accepted&truth).sum())
        if count>=20 and correct/count>=.95:
            candidates.append((count,float(threshold)))
    threshold=max(candidates,key=lambda item:item[0])[1] if candidates else None
    accepted=(probability>=threshold)&report[:,None] if threshold is not None else torch.zeros_like(truth)
    count=int(accepted.sum());correct=int((accepted&truth).sum())
    actor_error=torch.cat(actor_error);baseline_error=torch.cat(baseline_error)
    metrics=dict(validation_states=len(probability),physical_positive_queries=int(truth.sum()),
        calibration_states=int(calibration.sum()),report_states=int(report.sum()),threshold=threshold,
        report_accepts=count,report_correct_accepts=correct,report_precision=correct/count if count else None,
        report_recall=correct/max(1,int((truth&report[:,None]).sum())),
        actor_teacher_mse=float(actor_error.mean()),baseline_teacher_mse=float(baseline_error.mean()),
        actor_wait_mse=float(actor_error[val['pending']].mean()),baseline_wait_mse=float(baseline_error[val['pending']].mean()),
        actor_execution_mse=float(actor_error[~val['pending']].mean()),baseline_execution_mse=float(baseline_error[~val['pending']].mean()),
        evidence_offline_gate_passed=count>=20 and correct/count>=.95,
        episode_success=None,scope='correlated offline sensor states; requires separate closed-loop comparison')
    metrics['execution_phase_validation']={}
    for name,mask in [('inspection',(~val['pending'])&(~val['known'])),
                      ('bound_execution',(~val['pending'])&val['known'])]:
        metrics['execution_phase_validation'][name]=dict(states=int(mask.sum()),
            actor_teacher_mse=float(actor_error[mask].mean()) if mask.any() else None,
            student_teacher_mse=float(baseline_error[mask].mean()) if mask.any() else None)
    if args.balance_binding_state:
        binding_errors={flag:torch.cat(values) for flag,values in binding_errors.items()}
        binding_delta=torch.cat(binding_deltas)
        metrics['binding_state_validation']={}
        for phase,mask in [('pending',val['pending']),('executing',~val['pending'])]:
            delta=binding_delta[mask]
            metrics['binding_state_validation'][phase]=dict(states=int(mask.sum()),
                unknown_teacher_mse=float(binding_errors[False][mask].mean()),
                known_teacher_mse=float(binding_errors[True][mask].mean()),
                mean_abs_action_delta=float(delta.mean()),p95_abs_action_delta=float(delta.quantile(.95)),
                max_abs_action_delta=float(delta.max()))
    torch.save(dict(format=manifest['format'],model=evidence.state_dict(),feature_mean=mean.cpu(),feature_scale=scale.cpu(),
                    bank=bank.cpu(),threshold=threshold,metrics=metrics,manifest=manifest),args.output/'evidence.pt')
    torch.save(dict(format=manifest['format'],model=actor.state_dict(),feature_mean=actor.encoder.mean.cpu(),feature_scale=actor.encoder.scale.cpu(),
                    bank=bank.cpu(),metrics=metrics,manifest=manifest),args.output/'maintenance_actor.pt')
    torch.save(dict(probability=probability.cpu(),truth=truth.cpu(),calibration=calibration.cpu(),
                    env_id=val['env_id'].cpu(),episode_id=val['episode_id'].cpu(),sim_step=val['sim_step'].cpu()),
               args.output/'validation_predictions.pt')
    (args.output/'completion.json').write_text(json.dumps(dict(completed=True,metrics=metrics),indent=2)+'\n')
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
