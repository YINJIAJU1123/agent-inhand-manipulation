"""Calibrate learned pose-error correction against a tuned geometric baseline.

Training and calibration use development captures only. Report groups are
disjoint trajectories, but the capture split has already informed earlier
development decisions. This is not an untouched formal test set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation

from train_modules import load_data,sha
from control_modules import PoseCompletionCorrection
from goals import load_bank

def measured_errors(manifest,data,rotations):
    camera=manifest['camera']
    world_from_camera=torch.tensor(camera['T_world_camera_env0'],dtype=torch.float64)[:3,:3]
    base_from_camera=torch.tensor(camera['T_base_camera'],dtype=torch.float64)[:3,:3]
    world_from_base=world_from_camera@base_from_camera.T
    base_quaternion=torch.tensor(Rotation.from_matrix(world_from_base.numpy()).as_quat(),dtype=torch.float64)
    q=F.normalize(data['measured_pose'][:,3:7].double(),dim=-1)
    a=base_quaternion[:3].expand(len(q),-1)
    xyz=base_quaternion[3]*q[:,:3]+q[:,3,None]*a+torch.cross(a,q[:,:3],dim=-1)
    w=base_quaternion[3]*q[:,3]-(a*q[:,:3]).sum(-1)
    world=F.normalize(torch.cat((xyz,w[:,None]),-1),dim=-1)
    targets=F.normalize(rotations.cpu().double(),dim=-1)
    return (2*torch.acos((world[:,None]*targets[None]).sum(-1).abs().clamp(0,1))).float(),base_quaternion.tolist()

def metrics(accepted,truth,mask):
    selected=accepted&mask[:,None]
    correct=selected&truth
    tp=int(correct.sum());n=int(selected.sum());positive=int((truth&mask[:,None]).sum())
    return dict(accepted=n,correct=tp,false_accepts=n-tp,precision=tp/n if n else None,
                recall=tp/positive if positive else None,positives=positive)

def choose(scores,truth,calibration,thresholds,precision):
    candidates=[]
    for value in thresholds:
        m=metrics(scores(value),truth,calibration)
        if m['accepted']>=20 and m['precision']>=precision:
            candidates.append((m['correct'],float(value),m))
    return max(candidates,key=lambda x:(x[0],x[2]['precision'])) if candidates else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','text_cache','original_asset','output','reference_baseline'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=1200)
    p.add_argument('--seed',type=int,default=182851)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.manual_seed(a.seed);torch.set_num_threads(4)
    device='cuda:0'
    bank,rotations,bank_manifest=load_bank(a.text_cache,a.original_asset,device)
    tm,train=load_data(a.data/'train')
    vm,val=load_data(a.data/'validation')
    train['measured_angles'],base_quaternion=measured_errors(tm,train,rotations)
    val['measured_angles'],val_base=measured_errors(vm,val,rotations)
    torch.testing.assert_close(torch.tensor(base_quaternion),torch.tensor(val_base))
    groups=list(zip(val['env_id'].tolist(),val['episode_id'].tolist()))
    calibration=torch.tensor([int.from_bytes(hashlib.sha256(f"{vm['seed']}:{e}:{ep}".encode()).digest()[:4],'little')%2==0 for e,ep in groups],device=device)
    report=~calibration
    truth=(val['angles_rad']<=.16).to(device)
    raw_errors=val['measured_angles'].to(device)
    baseline=metrics(raw_errors<=.16,truth,report)
    reference=json.loads(a.reference_baseline.read_text())
    if baseline['accepted']!=reference['accepted'] or baseline['correct']!=reference['correct']:
        raise ValueError(f'Analytical reference did not reproduce: {baseline}')
    mean=train['sensor'].mean(0).to(device)
    scale=train['sensor'].std(0).clamp_min(.01).to(device)
    model=PoseCompletionCorrection(mean,scale,bank).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.001)
    keys=('sensor','angles_rad','measured_angles')
    train={k:train[k].to(device) for k in keys}
    val={k:val[k].to(device) for k in keys}
    near=(train['measured_angles']<.5).nonzero()
    far=(train['measured_angles']>=.5).nonzero()
    if len(near)<100:raise ValueError('Insufficient near-target training states')
    start=time.monotonic()
    with (a.output/'training.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.updates+1):
            selected=torch.cat((near[torch.randint(len(near),(384,),device=device)],
                                far[torch.randint(len(far),(128,),device=device)]))
            index,goal=selected.unbind(-1)
            pred=model(train['sensor'][index],bank[goal],train['measured_angles'][index,goal])
            error=train['angles_rad'][index,goal]
            loss=(.5*((error-pred['mean'])/pred['sigma']).square()+pred['sigma'].log()).mean()
            loss=loss+.05*pred['correction'].square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%100==0:
                row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-start)
                log.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    model.eval()
    parts=[]
    with torch.inference_mode():
        for indices in torch.arange(len(val['sensor']),device=device).split(256):
            n=len(indices)
            pred=model(val['sensor'][indices,None].expand(-1,6,-1).reshape(n*6,-1),
                       bank[None].expand(n,-1,-1).reshape(n*6,-1),
                       raw_errors[indices].reshape(-1))
            parts.append({k:v.reshape(n,6) for k,v in pred.items()})
    prediction={k:torch.cat([part[k] for part in parts]) for k in parts[0]}
    wanted=.99
    raw_selected=choose(lambda t:raw_errors<=t,truth,calibration,torch.linspace(.06,.16,201),wanted)
    learned_selected=choose(lambda b:prediction['mean']+b*prediction['sigma']<=.16,
                           truth,calibration,torch.linspace(0,4,201),wanted)
    raw_threshold=raw_selected[1] if raw_selected else None
    beta=learned_selected[1] if learned_selected else None
    simple=metrics(raw_errors<=raw_threshold,truth,report) if raw_threshold is not None else None
    corrected=metrics(prediction['mean']+beta*prediction['sigma']<=.16,truth,report) if beta is not None else None
    near_report=report[:,None]&(raw_errors<.5)
    comparison=dict(raw_pose=baseline,calibrated_pose=simple,learned_correction=corrected,
        calibration_precision_target=wanted,selected_pose_threshold=raw_threshold,selected_uncertainty_multiplier=beta,
        raw_near_target_mae=float((raw_errors-val['angles_rad']).abs()[near_report].mean()),
        corrected_near_target_mae=float((prediction['mean']-val['angles_rad']).abs()[near_report].mean()),
        accepted_for_closed_loop=False,scope='development-only offline trajectory split; no autonomous result',
        elapsed_seconds=time.monotonic()-start)
    (a.output/'result.json').write_text(json.dumps(comparison,indent=2)+'\n')
    manifest=dict(format='pose_completion_correction_v1',seed=a.seed,updates=a.updates,
        source_student_sha256=tm['checkpoint_sha256'],goal_bank=bank_manifest,
        train_collection=tm,val_collection=vm,world_from_base_quaternion_xyzw=base_quaternion,
        calibration='same SHA256 seed/env/episode trajectory groups as earlier development heads',
        report_split_previously_used_for_development=True,formal_data_used=False,
        independent_success_threshold=.16,
        purpose='correct measured orientation error and estimate uncertainty; no change to task success',
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('train_completion.py','control_modules.py','train_modules.py','goals.py')})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    torch.save(dict(model=model.state_dict(),feature_mean=mean.cpu(),feature_scale=scale.cpu(),bank=bank.cpu(),
        beta=beta,metrics=comparison,manifest=manifest),a.output/'correction.pt')
    torch.save(dict(**{k:v.cpu() for k,v in prediction.items()},measured_angles=raw_errors.cpu(),
        truth=truth.cpu(),calibration=calibration.cpu()),a.output/'validation_predictions.pt')
    print(json.dumps(comparison),flush=True)

if __name__=='__main__':main()
