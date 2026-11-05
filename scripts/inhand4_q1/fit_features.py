"""Distill teacher conditioning features while freezing the released control MLP."""
import argparse,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import torch
import sys
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/"dice_latent_v1"))
import feature_model
from feature_model import migrate,FORMAT
import goals
from model import TARGET_COLUMNS
from goals import load_bank
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','data','bank','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch_size',type=int,default=512)
p.add_argument('--seed',type=int,default=186205);p.add_argument('--learning_rate',type=float,default=0.0003)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True);torch.set_num_threads(8);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
bank,_,goal_meta=load_bank(a.bank,'cpu');model,migration=migrate(payload,bank,a.seed);model.cuda();bank=bank.cuda()
data=torch.load(a.data,map_location='cpu',weights_only=False)
source_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
if data['metadata']['source_checkpoint_sha256']!=source_hash or data['metadata']['seed']==186202:raise ValueError('Training source or split mismatch')
x=data['physical'].cuda();g=data['goals'].long().cuda();y=data['labels'].cuda()
target=(data['teacher_goal_features'].cuda()-model.obs_norm.mean[list(TARGET_COLUMNS)])/model.obs_norm.std[list(TARGET_COLUMNS)]
with torch.no_grad():
 source_batch=int(data['metadata']['num_envs'])
 reference=torch.cat([model.action_with_features(x[i:i+source_batch],target[i:i+source_batch]) for i in range(0,4*source_batch,source_batch)])
 parity=float((reference-y[:len(reference)]).abs().max())
 torch.testing.assert_close(reference,y[:len(reference)],atol=1e-6,rtol=0)
metadata=dict(migration=migration,goal_bank=goal_meta,source_checkpoint_sha256=source_hash,data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),data_metadata=data['metadata'],
 seed=a.seed,rows=len(x),teacher_feature_injection_action_parity_max=parity,actor_frozen=True,teacher_action_labels="frozen teacher re-evaluated at training batch shape; stored-label parity at original batch shape",
 source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(feature_model.__file__),Path(__file__),Path(goals.__file__)]},
 arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
(a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=a.learning_rate)
started=time.monotonic()
for step in range(a.steps):
 ids=torch.randint(len(x),(a.batch_size,),device='cuda:0')
 pred=model.conditioning_features(x[ids],bank[g[ids]])
 feature_loss=(pred-target[ids]).square().mean()
 if step>=2000:
  # BF16 GEMM changes with batch shape. Re-evaluate the frozen teacher at
  # this batch shape instead of penalizing numerical differences to batch-64 labels.
  with torch.no_grad(): teacher_action=model.action_with_features(x[ids],target[ids])
  action=model.action_with_features(x[ids],pred);action_loss=(action-teacher_action).square().mean()
  loss=feature_loss+.2*action_loss
 else:action_loss=feature_loss.detach()*0;loss=feature_loss
 if not torch.isfinite(loss):raise FloatingPointError('Feature loss')
 optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
 if (step+1)%500==0:
  row=dict(step=step+1,feature_mse=float(feature_loss),action_mse=float(action_loss),elapsed=time.monotonic()-started)
  print(json.dumps(row),flush=True)
  with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if (step+1)%5000==0 or step+1==a.steps:
  dest=a.output/('final.pt' if step+1==a.steps else 'latest.pt');tmp=dest.with_suffix('.tmp')
  torch.save(dict(format=FORMAT,model=model.state_dict(),config=asdict(model.config),metadata=metadata,iteration=step+1,optimizer=optimizer.state_dict()),tmp);tmp.replace(dest)
for key,value in model.net.state_dict().items():torch.testing.assert_close(value.cpu(),payload['actor']['net.'+key],atol=0,rtol=0)
(a.output/'result.json').write_text(json.dumps(dict(complete=True,steps=a.steps,actor_weights_unchanged=True,elapsed=time.monotonic()-started,scope='Feature/action distillation only; closed-loop verification required'),indent=2)+'\n')
