"""Cache policy features and training labels in separate tensors, preserving splits."""
import argparse,json,time
from pathlib import Path
import torch
from assets import STUDENT,ADAPTER,save,sha
from features import FrozenFeatures

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);enc=FrozenFeatures(a.device);manifest=[]
    for path in sorted(Path(a.data).glob('*/data.pt')):
        d=torch.load(path,weights_only=True,map_location='cpu');xs=[]
        for start in range(0,len(d['rows']),32):xs.append(enc.candidates(d['features'][start:start+32].to(a.device),d['proprio'][start:start+32].to(a.device)).cpu())
        n=len(d['rows']);future=torch.zeros(n,6);valid=torch.zeros(n,6,dtype=torch.bool)
        lookup={(r['case'],r['step']):j for j,r in enumerate(d['rows'])}
        for j,r in enumerate(d['rows']):
            suffix=[lookup.get((r['case'],r['step']+delta)) for delta in [10,20,30]]
            if all(k is not None and d['rows'][k]['skill']==r['skill'] for k in suffix):
                c=r['skill'];end=suffix[-1];future[j,c]=float(d['error'][end,c]<d['error'][j,c]-.04 or d['error'][end,c]<=.16);valid[j,c]=True
        target=out/(path.parent.name+'.pt')
        torch.save(dict(x=torch.cat(xs),error=d['error'],future_progress=future,progress_valid=valid,rows=d['rows']),target)
        row=dict(shard=path.parent.name,frames=n,data_sha256=sha(path),cache_sha256=sha(target),ready_counts=(d['error']<=.16).sum(0).tolist())
        manifest.append(row);save(out/'progress.json',dict(status='caching',completed=len(manifest),current=row));print(json.dumps(row),flush=True)
    save(out/'manifest.json',dict(status='completed',student_sha256=sha(STUDENT),adapter_sha256=sha(ADAPTER),feature_source_sha256=sha(Path(__file__).parent/'features.py'),shards=manifest,
        contract='x contains frozen policy hidden state, policy command, robot proprioception, RGB marker statistics and the queried canonical language embedding projection. Simulator labels remain separate; neither true target color nor hidden layout is input. All six hypotheses are encoded.',
        progress='Observed same-skill 30-step suffix; endpoints missing after drops or horizon are censored, not assumed successful. Not an action-counterfactual value.'))
    save(out/'progress.json',dict(status='completed',completed=len(manifest)))
if __name__=='__main__':main()
