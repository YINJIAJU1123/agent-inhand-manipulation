"""Warm-start the direct-latent actor from disjoint source rollout training data."""
import argparse,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import torch
from model import migrate,configure_training,FORMAT
from goals import load_bank
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','data','bank','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--steps',type=int,default=2500);p.add_argument('--interface_steps',type=int,default=500)
p.add_argument('--batch_size',type=int,default=512);p.add_argument('--learning_rate',type=float,default=1e-4)
p.add_argument('--seed',type=int,default=186201);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
torch.set_num_threads(8);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model,source,migration=migrate(payload,a.seed);model.cuda();del source
vectors,quats,goal_meta=load_bank(a.bank,'cuda:0')
model.configure_goal_statistics(vectors)
data=torch.load(a.data,map_location='cpu',weights_only=False)
if data['format']!='dice_source_distillation_v1':raise ValueError('Unexpected data format')
source_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
if data['metadata']['source_checkpoint_sha256']!=source_hash:raise ValueError('Label source differs')
if data['metadata']['seed']==186202:raise ValueError('Development validation seed cannot train actor')
x=data['physical'].cuda();g=data['goals'].long().cuda();y=data['labels'].cuda()
if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite samples')
if x.shape[1:]!=(10,116) or g.shape!=x.shape[:2] or y.shape!=(len(x),21):raise ValueError('Sample shape mismatch')
metadata=dict(migration=migration,goal_bank=goal_meta,source_checkpoint_sha256=source_hash,data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
    data_metadata=data['metadata'],seed=a.seed,rows=len(x),pose_source='simulator with published sensor noise',stage='offline source warmup',
    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
(a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
hooks=[];stage=None;started=time.monotonic()
for step in range(a.steps):
 next_stage='interface' if step<a.interface_steps else 'adapt'
 if stage!=next_stage:
  for handle in hooks:handle.remove()
  stage=next_stage;hooks=configure_training(model,stage)
  optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=a.learning_rate,weight_decay=0)
  inherited=model.net[0].weight.detach().reshape(2048,10,-1)[...,:116].clone()
 ids=torch.randint(len(x),(a.batch_size,),device='cuda:0')
 prediction=model(x[ids],vectors[g[ids]])
 loss=(prediction-y[ids]).square().mean()
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
 optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
 if (step+1)%100==0:
  if stage=='interface':torch.testing.assert_close(model.net[0].weight.detach().reshape(2048,10,-1)[...,:116],inherited,rtol=0,atol=0)
  row=dict(step=step+1,stage=stage,training_action_mse=float(loss),elapsed=time.monotonic()-started)
  print(json.dumps(row),flush=True)
  with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if (step+1)%500==0 or step+1==a.steps:
  dest=a.output/('final.pt' if step+1==a.steps else 'latest.pt');tmp=dest.with_suffix('.tmp')
  torch.save(dict(format=FORMAT,model=model.state_dict(),config=asdict(model.config),metadata=metadata,iteration=step+1,stage=stage,optimizer=optimizer.state_dict()),tmp);tmp.replace(dest)
(a.output/'result.json').write_text(json.dumps(dict(complete=True,steps=a.steps,elapsed=time.monotonic()-started,scope='Supervised training only; closed-loop validation required'),indent=2)+'\n')
