"""Read-only development diagnostics. Does not choose a threshold or checkpoint."""
import argparse,json
from pathlib import Path
import torch
from assets import save,sha
from completion_model import load_model
from train import assess,statistics,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);root=Path(a.cache);model,pack=load_model(Path(a.training)/'head.pt','cpu');threshold=pack['threshold'];result={}
    for split in ['train','val']:
        paths=sorted(root.glob(split+'_*.pt'))
        groups=[(path.stem,[path]) for path in paths]
        if split=='train':groups=[('training_total',paths)]
        for name,files in groups:
            d=read(files);metrics,pred,raw=assess(model,d,'cpu');error=d['error'];truth=error<=.16
            bins={}
            for lo,hi in [(0,.08),(.08,.12),(.12,.16),(.16,.22),(.22,.4),(.4,1.),(1.,3.142)]:
                ids=(error>=lo)&(error<(hi if hi!=.16 else hi+1e-7));q=pred[ids]
                bins[f'{lo:g}-{hi:g}']=dict(count=int(ids.sum()),mean_probability=float(q.mean()) if len(q) else None,accepted=int((q>=threshold).sum()))
            per_goal=[statistics(pred[:,c],truth[:,c],threshold) for c in range(6)]
            rows=[]
            for file in files:
                source=torch.load(file,weights_only=True,map_location='cpu');rows.extend(source['rows'])
            trajectories={}
            for j,row in enumerate(rows):trajectories.setdefault((row['seed'],row['case']),[]).append(j)
            proposals=[]
            for key,indices in trajectories.items():
                indices.sort(key=lambda j:rows[j]['step']);reached=False;first=None
                for j in indices:
                    c=rows[j]['skill'];reached|=bool(error[j,c]<=.16)
                    if first is None and pack.get('calibration') and pred[j,c]>=threshold:
                        first=dict(step=rows[j]['step'],skill=c,error=float(error[j,c]),valid=bool(error[j,c]<=.16),probability=float(pred[j,c]))
                proposals.append(dict(seed=key[0],case=key[1],sampled_reach=reached,first_proposal=first))
            positive=truth.any(-1);ranking=(pred.argmax(-1)==error.argmin(-1))
            result[name]=dict(metrics=metrics,gate=statistics(pred,truth,threshold),per_goal=per_goal,error_bins=bins,
                ranking_accuracy_reached=float(ranking[positive].float().mean()) if positive.any() else None,
                replayed_trajectory_proposals=proposals)
    save(a.output,dict(checkpoint_sha256=sha(Path(a.training)/'head.pt'),threshold=threshold,calibrated=bool(pack.get('calibration')),splits=result,
        note='Read-only diagnosis after checkpoint and threshold selection. Correlated frames. Trajectory proposals replay recorded queried skills; they are not closed-loop intervention results. Formal test seeds absent.'))
if __name__=='__main__':main()
