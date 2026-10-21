"""CPU visual-grounding repair on existing independent-layout render data.

Calibration chooses a checkpoint and abstention threshold; the separate audit
seed never participates in either selection. This is not an Astra task eval.
"""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha
# Explicit local-module import avoids collision with old model.py.
import importlib.util
spec=importlib.util.spec_from_file_location('query_symbol',Path(__file__).with_name('factorized.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SymbolEvidence=module.SymbolEvidence

class Patches(Dataset):
 def __init__(self,paths):
  arrays=[]
  for path in paths:
   d=torch.load(path,map_location='cpu',weights_only=True);labels=d['labels'].flatten();keep=labels>=0
   colors=torch.arange(6).repeat(len(d['labels']))
   arrays.append((d['patches'].flatten(0,1)[keep],d['geometry'].flatten(0,1)[keep],colors[keep],labels[keep]))
  self.fields=[torch.cat([r[i]for r in arrays])for i in range(4)]
 def __len__(self):return len(self.fields[-1])
 def __getitem__(self,i):return self.fields[0][i].float()/255.,*(f[i]for f in self.fields[1:])

@torch.inference_mode()
def predict(model,loader,legacy=False):
 model.eval();ys=[];ts=[]
 for patch,geo,color,label in loader:
  if legacy:
   e=torch.cat((model.patch(patch),geo),-1)[:,None].repeat(1,4,1);logits=model.symbol(model.evidence(e)[0][:,-1])
  else:logits=model(patch,geo,color)
  ys.append(logits);ts.append(label)
 return torch.cat(ys),torch.cat(ts)

def metrics(logits,label,threshold):
 probabilities=logits.softmax(-1);known=label<6;pred=logits.argmax(-1);conf,selected=probabilities[:,:6].max(-1);accept=conf>=threshold
 per=[float((pred[label==c]==c).float().mean())if(label==c).any()else None for c in range(7)]
 return dict(counts=torch.bincount(label,minlength=7).tolist(),per_class=per,macro_readable=sum(x for x in per[:6]if x is not None)/sum(x is not None for x in per[:6]),readable_accuracy=float((pred[known]==label[known]).float().mean()),conditional_symbol_accuracy=float((logits[known,:6].argmax(-1)==label[known]).float().mean()),accepted=int(accept.sum()),precision=float((selected[accept]==label[accept]).float().mean())if accept.any()else None,unknown_false_accepts=int((accept&~known).sum()),threshold=threshold)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=120);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(3);torch.manual_seed(20261022)
 data=ROOT/'outputs/semantic_completion_v1/corrected_repair_20261021/symbols/data'
 paths=sorted(data.glob('train_*/data.pt'));calpath=data/'val_96211/data.pt';auditpath=data/'val_96212/data.pt'
 train=Patches(paths);cal=Patches([calpath]);counts=torch.bincount(train.fields[-1],minlength=7);assert(counts>0).all()
 sampling=torch.Generator().manual_seed(20261022);sampler=WeightedRandomSampler((1/counts.float())[train.fields[-1]],4096,replacement=True,generator=sampling)
 loader=DataLoader(train,batch_size=128,sampler=sampler);cal_loader=DataLoader(cal,batch_size=128)
 model=SymbolEvidence();opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=.02);history=[];best=-1.;start=time.time()
 save(out/'protocol.json',dict(train_files=[str(x)for x in paths],calibration=str(calpath),audit=str(auditpath),source_sha256=sha(__file__),model_source_sha256=sha(Path(__file__).with_name('factorized.py')),input='Measured color crop, measured crop geometry, queried color. No layout, pose, symbol labels, seed, or frame index enters forward.',selection='Maximum six-class calibration macro accuracy, then calibrate accepted count >=30 and precision >=90%; evaluate separate audit after selection.',epochs=a.epochs,samples_per_epoch=4096,factorized_readability=True,device='cpu',threads=3,cloud_calls=0,scope='Development representation repair; old seed splits, no final real-agent success claim.'))
 try:
  for epoch in range(1,a.epochs+1):
   model.train();losses=[]
   for patch,geo,color,label in loader:
    # Brightness noise only; measured inference preprocessing is unchanged.
    patch=(patch*(.85+.3*torch.rand(len(patch),1,1,1))).clamp(0,1)
    logits=model(patch,geo,color);loss=F.nll_loss(logits,label)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
   logits,label=predict(model,cal_loader);m=metrics(logits,label,.7);score=m['macro_readable']
   row=dict(epoch=epoch,loss=sum(losses)/len(losses),calibration=m,elapsed_s=time.time()-start);history.append(row);save(out/'history.json',history);save(out/'progress.json',dict(status='training',pid=os.getpid(),**row));print(json.dumps(row),flush=True)
   if score>best:
    best=score;torch.save(dict(model={k:v.detach().clone()for k,v in model.state_dict().items()},epoch=epoch,cloud_eligible=False),out/'head.pt')
  pack=torch.load(out/'head.pt',weights_only=True);model.load_state_dict(pack['model']);cl,ct=predict(model,cal_loader)
  curve=[metrics(cl,ct,t)for t in (.5,.6,.7,.8,.85,.9,.95,.98,.99)]
  eligible=[r for r in curve if r['accepted']>=30 and r['precision']>=.9];selected=max(eligible,key=lambda r:r['accepted'])if eligible else None;threshold=selected['threshold']if selected else 1.
  audit=Patches([auditpath]);audit_loader=DataLoader(audit,batch_size=128);al,at=predict(model,audit_loader);am=metrics(al,at,threshold)
  passed=bool(selected and am['accepted']>=30 and(am['precision']or 0)>=.9 and am['macro_readable']>=.7)
  pack.update(threshold=threshold,calibration=selected,audit=am,offline_gate_passed=passed);torch.save(pack,out/'head.pt')
  # Read-only comparison of the previous selected semantic branch, same audit views.
  from model import load_model
  oldpath=ROOT/'outputs/semantic_completion_v1/corrected_repair_20261021/final/learning/train/evidence.pt';old,_=load_model(oldpath,'cpu');ol,ot=predict(old,audit_loader,legacy=True)
  result=dict(status='completed',best_epoch=pack['epoch'],calibration=selected,calibration_curve=curve,audit=am,legacy_audit_fixed_threshold=metrics(ol,ot,.7),offline_gate_passed=passed,cloud_eligible=False,checkpoint_sha256=sha(out/'head.pt'),note='Still needs new-seed/current online-image shadow tests. Cannot replace semantic binding or raise task-success claims from this classifier audit.')
  save(out/'result.json',result);save(out/'progress.json',dict(status='completed',pid=None,elapsed_s=time.time()-start,offline_gate_passed=passed));print(json.dumps(result),flush=True)
 except BaseException:
  save(out/'failure.json',dict(error=traceback.format_exc()));save(out/'progress.json',dict(status='failed',pid=None));raise

if __name__=='__main__':main()
