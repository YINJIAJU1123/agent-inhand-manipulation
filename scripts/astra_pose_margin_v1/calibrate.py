"""Calibrate a conservative hold margin on frozen ordinal pose predictions.

Holding can be activated repeatedly, so first-stop precision is insufficient.
Use only the existing calibration trajectories to choose an inner tolerance
and confidence. Then evaluate the fixed choice on the existing audit split.
These are previously inspected development data, not fresh final validation.
"""
import argparse,json,sys,time,os,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/semantic_completion_v3'),str(ROOT/'scripts/semantic_completion_v2')]
from dense_model import load_model,TOLERANCES
from train_dense import SENSORS
from sequential import trajectories,THRESHOLD_GRID
from assets import save,sha
CHECKPOINT=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
CACHE=CHECKPOINT.parent.parent/'cache'


def metrics(probability,data,index,threshold):
    details=[]
    for key,ids in trajectories(data['rows']):
        color=key[-1];selected=probability[ids,color,index]>=threshold
        correct=data['error'][ids,color]<=.16
        details.append(dict(seed=key[0],case=key[1],segment=key[2],goal=color,
            accepted=int(selected.sum()),correct=int((selected&correct).sum()),wrong=int((selected&~correct).sum())))
    triggered=[x for x in details if x['accepted']]
    accepted=sum(x['accepted'] for x in details);correct=sum(x['correct'] for x in details)
    unsafe=sum(x['wrong']>0 for x in triggered)
    return dict(ordinal_index=index,inner_tolerance_rad=TOLERANCES[index],threshold=threshold,
        episodes=len(details),triggered_episodes=len(triggered),unsafe_episodes=unsafe,
        unsafe_episode_fraction=unsafe/len(triggered) if triggered else None,
        accepted_frames=accepted,correct_frames=correct,frame_precision=correct/accepted if accepted else None,
        details=details)


@torch.inference_mode()
def infer(model,path,out,name):
    data=torch.load(path,map_location='cpu',weights_only=True);all_prob=[]
    for start in range(0,len(data['rows']),64):
        output=model(*(data[k][start:start+64] for k in SENSORS));all_prob.append(output['ordinal'].sigmoid().cpu())
    probability=torch.cat(all_prob);torch.save(probability,out/(name+'_probabilities.pt'))
    save(out/'progress.json',dict(status='inferred',split=name,frames=len(data['rows']),pid=os.getpid(),updated=time.time()))
    return probability,data


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);model,pack=load_model(CHECKPOINT,'cpu');model.eval();model.requires_grad_(False)
    save(out/'protocol.json',dict(checkpoint_sha256=sha(CHECKPOINT),checkpoint=str(CHECKPOINT),
        calibration_sha256=sha(CACHE/'val_96311.pt'),audit_sha256=sha(CACHE/'val_96312.pt'),
        selection='At least eight triggered calibration trajectories and at most 5% containing any unsafe accepted frame. Maximize triggered trajectories, then correct accepted frames, then stricter inner tolerance.',
        use='Holding eligibility only. Original task tolerance remains 0.16rad; no new autonomous stop authority.',
        tolerances=[.08,.12,.16],confidence_grid=THRESHOLD_GRID,audit_gate='At least eight triggered audit trajectories, at most 10% containing any unsafe accepted frame.',
        scope='Previously inspected development data. Threshold frozen before audit, no deployment, no new Astra calls.',cloud_calls=0))
    prob,cal=infer(model,CACHE/'val_96311.pt',out,'calibration')
    curve=[metrics(prob,cal,i,t) for i in range(3) for t in THRESHOLD_GRID]
    eligible=[r for r in curve if r['triggered_episodes']>=8 and r['unsafe_episode_fraction']<=.05]
    selected=max(eligible,key=lambda r:(r['triggered_episodes'],r['correct_frames'],-r['inner_tolerance_rad'])) if eligible else None
    save(out/'calibration.json',dict(selected=selected,curve=curve,baseline=metrics(prob,cal,2,.7)))
    del prob,cal
    prob,audit=infer(model,CACHE/'val_96312.pt',out,'audit')
    scored=metrics(prob,audit,selected['ordinal_index'],selected['threshold']) if selected else None
    passed=bool(scored and scored['triggered_episodes']>=8 and scored['unsafe_episode_fraction']<=.1)
    result=dict(status='completed',calibration=selected,audit=scored,baseline_audit=metrics(prob,audit,2,.7),
        development_gate_passed=passed,cloud_calls=0,checkpoint_sha256=sha(CHECKPOINT),
        note='No model fitting, geometry inputs or deployment. Calibration and audit episodes were already used in earlier development; independent live validation still required.')
    save(out/'result.json',result);save(out/'progress.json',dict(status='completed',pid=os.getpid(),updated=time.time()))
    print(json.dumps({k:({x:y for x,y in v.items() if x!='details'} if isinstance(v,dict) else v) for k,v in result.items()},indent=2))

if __name__=='__main__':main()
