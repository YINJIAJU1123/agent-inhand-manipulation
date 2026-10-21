"""Train a small counterfactual retention value head on independent seed splits."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/semantic_completion_v1'),str(ROOT/'scripts/semantic_completion_v3')]
from common import save,sha
from dense_model import load_model
from value_model import RetentionValue,RATES,frozen_features
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(10422)
base=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt';encoder,_=load_model(base,'cuda:0')
parts={k:[]for k in ('train','calibration','audit')}
for prefix in sorted(Path(a.data).glob('*/prefix/report.json')):
 report=json.loads(prefix.read_text());meta=report['metadata'];split=meta['split'];parent=prefix.parent.parent
 branches={r:json.loads((parent/('rate'+str(r))/'report.json').read_text())for r in RATES}
 snapshots={r:torch.load(parent/('rate'+str(r))/'branches.pt',map_location='cpu',weights_only=True)for r in RATES}
 values={r:{row['case']:row for row in b['results']}for r,b in branches.items()}
 for case,example in snapshots[1.]['features'].items():
  if any(case not in values[r]for r in RATES):raise RuntimeError('Incomplete counterfactuals')
  reference=snapshots[1.]['states'][case]
  if any(float((snapshots[r]['states'][case][k]-v).abs().max())>1e-6 for r in RATES for k,v in reference.items()):raise RuntimeError('Counterfactual physical states differ')
  sensor={k:v[None].cuda()for k,v in example.items()};goal=torch.tensor([values[1.][case]['goal']],device='cuda:0')
  feature=frozen_features(encoder,sensor,goal)[0].cpu()
  parts[split].append(dict(feature=feature,quality=torch.tensor([values[r][case]['quality']for r in RATES]),drop=torch.tensor([float(values[r][case]['drop'])for r in RATES]),seed=meta['seed'],case=case))
 del snapshots
for split,min_count in [('train',32),('calibration',8),('audit',8)]:
 if len(parts[split])<min_count:raise RuntimeError('Too few independent '+split+' states: '+str(len(parts[split])))
sets={s:{r['seed']for r in rows}for s,rows in parts.items()}
assert not(sets['train']&sets['calibration']or sets['train']&sets['audit']or sets['calibration']&sets['audit'])
data={s:{k:torch.stack([r[k]for r in rows])for k in ('feature','quality','drop')}for s,rows in parts.items()}
torch.save(dict(data=data,keys={s:[dict(seed=r['seed'],case=r['case'])for r in rows]for s,rows in parts.items()}),out/'features.pt')
model=RetentionValue();model.mean.copy_(data['train']['feature'].mean(0));model.scale.copy_(data['train']['feature'].std(0).clamp_min(.05))
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.03);best=float('inf');history=[];start=time.time()
save(out/'protocol.json',dict(seeds={s:sorted(v)for s,v in sets.items()},states={s:len(v)for s,v in parts.items()},source_sha256=sha(__file__),model_source_sha256=sha(Path(__file__).with_name('value_model.py')),frozen_evidence_sha256=sha(base),epochs=100,selection='Minimum calibration value MSE; audit never selects checkpoint',rates=RATES,cloud_calls=0,supervision='Six-second counterfactual completion occupancy minus drop. Not autonomous Astra task success.'))
for epoch in range(100):
 model.train();indices=torch.randperm(len(parts['train']));losses=[]
 for batch in indices.split(32):
  d=data['train'];y=model(d['feature'][batch]);q=d['quality'][batch];v=y['value'];loss=F.smooth_l1_loss(v,q)+.2*F.binary_cross_entropy_with_logits(y['drop_logit'],d['drop'][batch])
  gap=q[:,:,None]-q[:,None,:];mask=gap>.05
  if mask.any():loss=loss+.2*F.softplus(.1-(v[:,:,None]-v[:,None,:]))[mask].mean()
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();losses.append(float(loss))
 model.eval()
 with torch.no_grad():validation=float(F.mse_loss(model(data['calibration']['feature'])['value'],data['calibration']['quality']))
 history.append(dict(epoch=epoch+1,loss=sum(losses)/len(losses),calibration_mse=validation));save(out/'progress.json',dict(status='training',**history[-1],elapsed_s=time.time()-start))
 if validation<best:
  best=validation;torch.save(dict(model={k:v.clone()for k,v in model.state_dict().items()},epoch=epoch+1,base_sha256=sha(base),rates=RATES,cloud_eligible=False),out/'head.pt')
model.load_state_dict(torch.load(out/'head.pt',weights_only=True)['model']);model.eval();metrics={}
for split in ('calibration','audit'):
 d=data[split]
 with torch.no_grad():choice=model(d['feature'])['value'].argmax(-1)
 idx=torch.arange(len(choice));q=d['quality'];drops=d['drop'];metrics[split]=dict(states=len(choice),chosen_mean_quality=float(q[idx,choice].mean()),chosen_drops=int(drops[idx,choice].sum()),choices=[int((choice==i).sum())for i in range(3)],fixed_mean_quality=q.mean(0).tolist(),fixed_drops=drops.sum(0).tolist(),oracle_mean_quality=float(q.max(-1).values.mean()))
best_fixed=max(range(3),key=lambda i:metrics['calibration']['fixed_mean_quality'][i]);audit=metrics['audit'];passed=audit['chosen_mean_quality']>audit['fixed_mean_quality'][best_fixed]+.03 and audit['chosen_drops']<=audit['fixed_drops'][best_fixed]
save(out/'history.json',history);save(out/'result.json',dict(metrics=metrics,calibration_selected_fixed_rate=RATES[best_fixed],offline_gate_passed=passed,cloud_eligible=False,checkpoint_sha256=sha(out/'head.pt'),note='Seed-held-out local counterfactual decision pilot. Requires closed-loop validation and real Astra evaluation.'))
save(out/'progress.json',dict(status='completed',offline_gate_passed=passed,elapsed_s=time.time()-start));print(json.dumps(metrics,indent=2))
