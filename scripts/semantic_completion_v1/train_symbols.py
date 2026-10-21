"""Repair semantic evidence using balanced render-layout augmentation.

Freeze the validated pose branch; train the patch evidence and observation
heads on training-only symbol views and the original skill transitions.
"""
import argparse,itertools,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from train import Frames,predict,evaluate
from model import load_model
from common import save,sha

class Symbols(Dataset):
    def __init__(self,paths):
        ds=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        self.patches=torch.cat([d['patches'] for d in ds]).flatten(0,1)
        self.geo=torch.cat([d['geometry'] for d in ds]).flatten(0,1)
        self.labels=torch.cat([d['labels'] for d in ds]).flatten()
        usable=self.labels>=0
        self.patches=self.patches[usable];self.geo=self.geo[usable];self.labels=self.labels[usable]
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return self.patches[i].float()/255,self.geo[i],self.labels[i]

def symbol_forward(model,p,g):
    e=torch.cat((model.patch(p),g),-1)[:,None].repeat(1,4,1)
    return model.symbol(model.evidence(e)[0][:,-1])

@torch.inference_mode()
def check_symbols(model,loader,device):
    model.eval();ys=[];ts=[]
    for p,g,t in loader:ys.append(symbol_forward(model,p.to(device),g.to(device)).cpu());ts.append(t)
    y=torch.cat(ys);t=torch.cat(ts);pred=y.argmax(-1);prob=y.softmax(-1);known=t<6
    confident=prob[:,:6].max(-1).values>=.7;select=prob[:,:6].argmax(-1)
    per=[float((pred[t==c]==c).float().mean()) if (t==c).any() else None for c in range(7)]
    return dict(counts=torch.bincount(t,minlength=7).tolist(),per_class_accuracy=per,
        macro_readable=sum(v for v in per[:6] if v is not None)/sum(v is not None for v in per[:6]),
        readable_accuracy=float((pred[known]==t[known]).float().mean()),
        confident_count=int(confident.sum()),
        confident_precision=float((select[confident]==t[confident]).float().mean()) if confident.any() else None,
        unknown_false_accepts=int((confident&~known).sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--learning',required=True);p.add_argument('--symbols',required=True)
    p.add_argument('--initial',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=80)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(20261021);device='cuda:0'
    root=Path(a.learning);aug=Path(a.symbols);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    tr=Frames(sorted((root/'data').glob('train_*/data.pt')));va=Frames(sorted((root/'data').glob('val_*/data.pt')))
    train_files=sorted((aug/'data').glob('train_*/data.pt'));val_files=sorted((aug/'data').glob('val_*/data.pt'))
    st=Symbols(train_files);sv=Symbols(val_files);counts=torch.bincount(st.labels,minlength=7)
    if (counts==0).any() or (torch.bincount(sv.labels,minlength=7)==0).any():raise RuntimeError('Incomplete symbol coverage')
    weights=1/counts.float();sampler=WeightedRandomSampler(weights[st.labels],len(tr)*2,replacement=True)
    sl=DataLoader(st,batch_size=64,sampler=sampler);vl=DataLoader(sv,batch_size=128)
    dl=DataLoader(tr,batch_size=32,shuffle=True);vdl=DataLoader(va,batch_size=64)
    model,old=load_model(a.initial,device)
    for name in ['patch','evidence','symbol','visibility','view']:getattr(model,name).requires_grad_(True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    real_counts=torch.zeros(7)
    for i in range(len(tr)):
        label=tr[i]['symbol'];real_counts+=torch.bincount(label[label>=0],minlength=7)
    real_weights=(1/real_counts.clamp_min(1)).to(device)
    save(out/'protocol.json',dict(source_sha256=sha(__file__),initial_sha256=sha(a.initial),
         augmentation_sha256={str(p):sha(p) for p in train_files+val_files},train_counts=counts.tolist(),
         frozen=['sensor','temporal','pose','ready','sensor_mean','sensor_std'],
         selection='Harmonic mean of held-out original temporal readable accuracy and balanced-layout six-class macro accuracy. No formal test data.',
         augmentation='Mild intensity scale only. Symbol-layout variants are single-view auxiliary examples, not temporal continuations.'))
    history=[];best=-1;start=time.time()
    for epoch in range(a.epochs):
        model.train();sit=iter(sl);losses=[]
        for b in dl:
            b={k:v.to(device) for k,v in b.items()}
            try:patch,geo,label=next(sit)
            except StopIteration:sit=iter(sl);patch,geo,label=next(sit)
            patch=patch.to(device);geo=geo.to(device);label=label.to(device)
            patch=(patch*(.9+.2*torch.rand(len(patch),1,1,1,device=device))).clamp(0,1)
            logits=symbol_forward(model,patch,geo);y=predict(model,b)
            loss=F.cross_entropy(logits,label)+.2*F.cross_entropy(y['symbol'].flatten(0,1),b['symbol'].flatten(),weight=real_weights)
            loss+=.1*F.binary_cross_entropy_with_logits(y['visible'],b['visible'])
            valid=b['future_valid']
            if valid.any():
                loss+=.3*(F.binary_cross_entropy_with_logits(y['future_visible'][valid,b['skill'][valid]],b['future'][valid])+F.binary_cross_entropy_with_logits(y['survival'][valid,b['skill'][valid]],b['survived'][valid]))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,vdl,device);symbols=check_symbols(model,vl,device)
        x=metrics['symbol_accuracy_readable'];y=symbols['macro_readable'];score=2*x*y/max(x+y,1e-9)
        row=dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),validation=metrics,symbols=symbols,selection_score=score,wall_s=time.time()-start)
        history.append(row);save(out/'history.json',history);print(json.dumps(row),flush=True)
        if score>best:
            best=score
            pack=dict(old,model={k:v.cpu().clone() for k,v in model.state_dict().items()},epoch=epoch+1,metrics=metrics,
                auxiliary_symbol_metrics=symbols,thresholds=dict(ready=metrics['gate_threshold'],symbol=.7),source='balanced_symbol_layout_auxiliary_v3')
            torch.save(pack,out/'evidence.pt')
    pack=torch.load(out/'evidence.pt',weights_only=True,map_location='cpu')
    m=pack['metrics'];s=pack['auxiliary_symbol_metrics']
    eligible=(m['gate_calibrated'] and m['symbol_accuracy_readable']>=.7 and s['macro_readable']>=.7 and s['confident_count']>=30 and (s['confident_precision'] or 0)>=.9)
    save(out/'result.json',dict(status='completed',best_epoch=pack['epoch'],metrics=m,auxiliary_symbol_metrics=s,
         checkpoint_sha256=sha(out/'evidence.pt'),cloud_eligible=eligible,train_seeds=tr.seeds,val_seeds=va.seeds,
         note='Development repair. Original pilot failures retained. Formal four-arm benchmark has not run.'))

if __name__=='__main__':main()
