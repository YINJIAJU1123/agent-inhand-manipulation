"""Matched temporal/current-frame heads on existing development sensor trajectories.

No external model calls or policy deployment. Test frames do not select epochs
or thresholds. These historical splits have been inspected before, so an
improvement here requires fresh simulator validation before integration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT/'scripts/semantic_completion_v3'),str(ROOT/'scripts/semantic_completion_v2'),str(ROOT/'scripts/astra_retention_v1')])
from dense_model import load_model
from sequential import score,THRESHOLD_GRID,trajectories,first_stop_loss
from model import TemporalCompletion


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)


def windows(rows,length=8):
    history=defaultdict(list);indices=[]
    for j,r in enumerate(rows):
        key=(r['seed'],r['case'],r.get('segment',0));prior=history[key]
        if prior and rows[prior[-1]]['step']>=r['step']:raise ValueError('Noncausal sequence order')
        if prior and rows[prior[-1]]['step']+10!=r['step']:prior.clear()
        prior.append(j);recent=prior[-length:];indices.append([recent[0]]*(length-len(recent))+recent)
    return torch.tensor(indices)



def frozen_features(model,data):
    captured=[];hook=model.query.register_forward_hook(lambda module,args,value:captured.append(value))
    try:output=model(data['x'],data['roi_rgb'],data['roi_depth'],data['roi_box'],data['roi_found'])
    finally:hook.remove()
    result=torch.cat((captured[0],output['ordinal'],output['log_error'][...,None],data['x'][...,533:554]),-1)
    if result.shape[1:]!=(6,125) or not torch.isfinite(result).all():raise RuntimeError('Invalid measured feature bank')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
    torch.set_num_threads(2);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dense=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt';cache=dense.parent.parent/'cache'
    files=dict(train=sorted(cache.glob('train_*.pt')),calibration=[cache/'val_96311.pt'],audit=[cache/'val_96312.pt'])
    assert len(files['train'])==8
    provenance={str(p):sha(p) for paths in files.values() for p in paths}
    provenance.update({str(dense):sha(dense),str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('model.py')):sha(Path(__file__).with_name('model.py'))})
    save(out/'protocol.json',dict(source_sha256=provenance,epochs=a.epochs,variants=['current','temporal'],history_frames=8,
        measurements='Frozen dense query hidden, seven ordinal logits, log-error estimate, and 21 measured candidate features. No labels, seed, layout or outcome are inference inputs.',
        split='Eight existing training seeds, calibration96311, audit96312. Previously inspected development splits; not new held-out evidence.',
        objective='Weighted BCE plus log-error regression and first-stop trajectory objective. Matched data, optimizer and training budget.',
        selection='Minimum calibration balanced BCE across ready, near-negative and far-negative groups. No audit selection.',
        gate='Calibration and audit at least eight first-stop proposals, >=90% precision, audit reached recall >=30%. Temporal must reduce false first stops without losing correct stops relative to the frozen dense baseline.',
        cloud_calls=0,deployed=False))
    started=time.monotonic();model,_=load_model(dense,'cpu');data={}
    for split,paths in files.items():
        feats=[];errors=[];rows=[];base=[]
        for path in paths:
            d=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            with torch.inference_mode():
                for start in range(0,len(d['rows']),64):
                    end=start+64
                    feature=frozen_features(model,{k:d[k][start:end] for k in ('x','roi_rgb','roi_depth','roi_box','roi_found')})
                    feats.append(feature)
                    base.append(feature[...,98].sigmoid())
            errors.append(d['error']);rows+=d['rows']
            save(out/'progress.json',dict(status='extracting',split=split,file=str(path),pid=os.getpid(),elapsed_s=time.monotonic()-started))
        data[split]=dict(feature=torch.cat(feats),error=torch.cat(errors),rows=rows,base=torch.cat(base))
        data[split]['indices']=windows(rows)
        print(json.dumps(dict(split=split,frames=len(rows))),flush=True)
    del model
    train,cal,audit=[data[k] for k in ('train','calibration','audit')]
    # Frozen scalar features retain all six goal candidates for every frame.
    assert train['feature'].shape[1:]==(6,125)
    def assess(net,d):
        with torch.inference_mode():
            net.eval();pred=[]
            for ids in torch.arange(len(d['rows'])).split(128):pred.append(net(d['feature'][d['indices'][ids]])['ready'])
            logits=torch.cat(pred);y=d['error']<=.16
            bce=F.binary_cross_entropy_with_logits(logits,y.float(),reduction='none')
            masks=[y,(d['error']>.16)&(d['error']<=.5),d['error']>.5]
            return float(torch.stack([bce[m].mean() for m in masks]).mean()),logits.sigmoid()
    results={}
    for variant in ('current','temporal'):
        torch.manual_seed(202610223);net=TemporalCompletion(variant=='temporal')
        net.mean.copy_(train['feature'].flatten(0,1).mean(0));net.scale.copy_(train['feature'].flatten(0,1).std(0).clamp_min(.05))
        optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.03)
        dest=out/variant;dest.mkdir();history=[];best=float('inf');groups=trajectories(train['rows'])
        for epoch in range(a.epochs):
            net.train();losses=[]
            for ids in torch.randperm(len(train['rows'])).split(128):
                pred=net(train['feature'][train['indices'][ids]]);e=train['error'][ids];ready=e<=.16
                weights=1+3*ready.float()+2*((e>.16)&(e<=.4)).float()
                bce=F.binary_cross_entropy_with_logits(pred['ready'],ready.float(),reduction='none')
                loss=((bce+.15*F.smooth_l1_loss(pred['log_error'],e.clamp_min(.02).log(),reduction='none'))*weights).sum()/weights.sum()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
            # Causal full-trajectory supervision of the first decision, with the
            # commanded training color used only to select supervision labels.
            for group_index in torch.randperm(len(groups)):
                key,ids=groups[int(group_index)];color=key[-1];ids=torch.tensor(ids)
                pred=net(train['feature'][train['indices'][ids]])['ready'][:,color]
                loss=.05*first_stop_loss(pred,train['error'][ids,color]<=.16)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step()
            val,prob=assess(net,cal);row=dict(epoch=epoch+1,loss=sum(losses)/len(losses),calibration_balanced_bce=val)
            history.append(row);save(out/'progress.json',dict(status='training',variant=variant,pid=os.getpid(),**row,elapsed_s=time.monotonic()-started))
            save(dest/'history.json',history)
            if val<best:
                best=val;torch.save(dict(model=net.state_dict(),temporal=variant=='temporal',epoch=epoch+1,encoder_sha256=sha(dense),deployed=False),dest/'head.pt')
            if (epoch+1)%5==0:print(json.dumps(dict(variant=variant,**row)),flush=True)
        pack=torch.load(dest/'head.pt',map_location='cpu',weights_only=True);net.load_state_dict(pack['model'])
        _,prob=assess(net,cal);curve=[dict(threshold=t,**score(prob,cal,t)) for t in THRESHOLD_GRID]
        eligible=[r for r in curve if r['accepted']>=8 and r['precision']>=.9]
        selected=max(eligible,key=lambda r:(r['correct'],r['precision'],-r['threshold'])) if eligible else None
        _,prob=assess(net,audit);baseline=score(audit['base'],audit,.7)
        measured=score(prob,audit,selected['threshold']) if selected else None
        passed=bool(measured and measured['accepted']>=8 and measured['precision']>=.9 and measured['reached_recall']>=.3 and measured['false_first_stops']<baseline['false_first_stops'] and measured['correct']>=baseline['correct'])
        pack.update(threshold=selected['threshold'] if selected else 1.,calibration=selected,offline_gate_passed=passed,history_frames=8)
        torch.save(pack,dest/'head.pt')
        results[variant]=dict(epoch=pack['epoch'],calibration=selected,audit=measured,dense_baseline=baseline,offline_gate_passed=passed,checkpoint_sha256=sha(dest/'head.pt'))
        save(dest/'calibration_curve.json',curve);save(dest/'result.json',results[variant])
    save(out/'result.json',dict(status='completed',variants=results,cloud_calls=0,deployed=False,limitation='Previously inspected development split. Requires new simulator data and live shadow verification.'))
    save(out/'progress.json',dict(status='completed',pid=os.getpid(),elapsed_s=time.monotonic()-started,offline_gate_passed={k:v['offline_gate_passed'] for k,v in results.items()}))


if __name__=='__main__':main()
