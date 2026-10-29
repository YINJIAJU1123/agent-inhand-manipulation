"""Build same-state goal pairs by replaying the frozen student's recurrent history.

Each alternative uses the saved measured sensors and the same pre-decision
history, as at an actual goal switch. Simulator targets are teacher labels only.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from dataclasses import asdict
from control_modules import features,FIELDS
from goals import load_bank
from model import LatentStudent
from train_modules import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','checkpoint','text_cache','original_asset','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda:0'
    bank,_,goal_manifest=load_bank(a.text_cache,a.original_asset,device)
    from viserdex.distillation.config import StudentCfg
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg=StudentCfg(**checkpoint['original_student_cfg'])
    student=LatentStudent(cfg,checkpoint['latent_dim'],checkpoint['goal_width']).to(device).eval()
    student.load_state_dict(checkpoint['model'])
    student.requires_grad_(False)
    metadata=dict(format='pose_latent_counterfactual_features_v1',checkpoint_sha256=sha(a.checkpoint),
        goal_manifest=goal_manifest,source_sha256=sha(__file__),splits={},
        scope='offline same-state goal-switch features; no formal evaluation data')
    started=time.monotonic()
    for split in ('train','validation'):
        source=a.data/split
        manifest=json.loads((source/'manifest.json').read_text())
        completion=json.loads((source/'completion.json').read_text())
        if not completion['completed'] or not manifest['continuous_protocol']:
            raise ValueError('Completed continuous collection required')
        if manifest['checkpoint_sha256']!=metadata['checkpoint_sha256']:
            raise ValueError('Frozen student differs from the collection')
        for key in ('asset_sha256','cache_sha256'):
            if manifest['goal_bank'][key]!=goal_manifest[key]:raise ValueError('Goal bank mismatch')
        keys=set(FIELDS)|{'sim_step','env_id','episode_id','episode_step','goal_index','next_done'}
        parts=[{k:v for k,v in torch.load(path,map_location='cpu',weights_only=True).items() if k in keys}
               for path in sorted(source.glob('samples_*.pt'))]
        data={k:torch.cat([part[k] for part in parts]) for k in keys}
        count=len(data['goal_index'])
        n=int(data['env_id'].max())+1
        if count!=completion['samples'] or count%n:raise ValueError('Incomplete dense capture')
        steps=count//n
        torch.testing.assert_close(data['env_id'],torch.arange(n).repeat(steps),atol=0,rtol=0)
        torch.testing.assert_close(data['sim_step'],torch.arange(steps).repeat_interleave(n),atol=0,rtol=0)
        if not (data['episode_step'][:n]==0).all():raise ValueError('Missing initial recurrent history')
        for t in range(1,steps):
            expected=data['episode_id'][(t-1)*n:t*n]+data['next_done'][(t-1)*n:t*n].long()
            torch.testing.assert_close(data['episode_id'][t*n:(t+1)*n],expected,atol=0,rtol=0)
        data={k:v.to(device) for k,v in data.items()}
        hidden=student.initial_state(n,device)
        paired_features=[]
        paired_proposals=[]
        max_proposal_error=0.
        max_belief_error=0.
        with torch.inference_mode():
            for t in range(steps):
                index=slice(t*n,(t+1)*n)
                values={k:data[k][index] for k in FIELDS}
                goal=data['goal_index'][index]
                actual,_,next_hidden=student(values['proprio'],values['measured_pose'],bank[goal],hidden)
                actual_belief=student.core.belief(next_hidden[-1])
                max_proposal_error=max(max_proposal_error,float((actual-values['proposal']).abs().max()))
                max_belief_error=max(max_belief_error,float((actual_belief-values['belief']).abs().max()))
                if max(max_proposal_error,max_belief_error)>2e-4:
                    raise ValueError(f'Recurrent replay mismatch at step {t}: proposal={max_proposal_error}, belief={max_belief_error}')
                repeated={k:v.repeat_interleave(6,0) for k,v in values.items()}
                proposal,_,branch_hidden=student(repeated['proprio'],repeated['measured_pose'],
                    bank.repeat(n,1),hidden.repeat_interleave(6,1))
                repeated['proposal']=proposal
                repeated['belief']=student.core.belief(branch_hidden[-1])
                paired_features.append(features(repeated).reshape(n,6,-1).cpu())
                paired_proposals.append(proposal.reshape(n,6,-1).cpu())
                hidden=next_hidden*(~data['next_done'][index].bool())[None,:,None]
        record=dict(states=count,steps=steps,num_envs=n,source_manifest_sha256=sha(source/'manifest.json'),
                    max_replayed_proposal_abs_error=max_proposal_error,max_replayed_belief_abs_error=max_belief_error)
        torch.save(dict(format=metadata['format'],features=torch.cat(paired_features),
            proposals=torch.cat(paired_proposals),sim_step=data['sim_step'].cpu(),env_id=data['env_id'].cpu(),
            episode_id=data['episode_id'].cpu(),goal_index=data['goal_index'].cpu(),
            checkpoint_sha256=metadata['checkpoint_sha256'],goal_manifest=goal_manifest,**record),a.output/(split+'.pt'))
        metadata['splits'][split]=record
        print(json.dumps(dict(split=split,**record,elapsed_seconds=time.monotonic()-started)),flush=True)
        del data,parts,paired_features,paired_proposals
    metadata['completed']=True
    metadata['elapsed_seconds']=time.monotonic()-started
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
