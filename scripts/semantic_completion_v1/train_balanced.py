"""Development repair: class-balanced symbol loss, geometry-preserving RGB jitter.

Uses the same training/validation split. No test outcomes are consumed.
"""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train import Frames,predict,evaluate,loss_fn
from model import load_model
from common import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--learning',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261021);device='cuda:0'
    root=Path(a.learning);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    train=Frames(sorted((root/'data').glob('train_*/data.pt')));val=Frames(sorted((root/'data').glob('val_*/data.pt')))
    counts=torch.zeros(7)
    for i in range(len(train)):label=train[i]['symbol'];counts+=torch.bincount(label[label>=0],minlength=7)
    weights=(1/counts.clamp_min(1));weights=weights/weights[:6].mean();weights=weights.to(device)
    model,old=load_model(root/'train/evidence.pt',device);model.requires_grad_(True)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    train_loader=DataLoader(train,batch_size=32,shuffle=True);val_loader=DataLoader(val,batch_size=64)
    started=time.time();history=[];best=None
    save(out/'protocol.json',dict(train_counts=counts.tolist(),weights=weights.tolist(),train_seeds=train.seeds,val_seeds=val.seeds,
        epochs=a.epochs,initial_checkpoint_sha256=sha(root/'train/evidence.pt'),source_sha256=sha(__file__),
        selection='Validation readable accuracy among calibrated pose gates, then validation loss. Development set only.',
        augmentation='One shared temporal/spatial 90-degree rotation and flip per minibatch, mild intensity scale. Geometry unchanged. Pose branch uses original sensor features.'))
    for epoch in range(a.epochs):
        model.train();losses=[]
        for b in train_loader:
            b={k:v.to(device) for k,v in b.items()}
            b['patches']=torch.rot90(b['patches'],int(torch.randint(4,())),(-2,-1))
            if bool(torch.rand(())<.5):b['patches']=b['patches'].flip(-1)
            b['patches']=(b['patches']*(.85+.3*torch.rand(len(b['patches']),1,1,1,1,1,device=device))).clamp(0,1)
            y=predict(model,b)
            base_ce=F.cross_entropy(y['symbol'].flatten(0,1),b['symbol'].flatten(),weight=torch.tensor([1.]*6+[.35],device=device))
            balanced_ce=F.cross_entropy(y['symbol'].flatten(0,1),b['symbol'].flatten(),weight=weights)
            loss=loss_fn(y,b)-base_ce+balanced_ce
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,val_loader,device)
        row=dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),validation=metrics,wall_s=time.time()-started)
        history.append(row);save(out/'history.json',history);print(json.dumps(row),flush=True)
        score=(bool(metrics['gate_calibrated']),metrics['symbol_accuracy_readable'],-metrics['loss'])
        if best is None or score>best:
            best=score
            pack=dict(old,model={k:v.cpu().clone() for k,v in model.state_dict().items()},epoch=epoch+1,metrics=metrics,
                      thresholds=dict(ready=metrics['gate_threshold'],symbol=.7),source='class_balanced_development_v2')
            torch.save(pack,out/'evidence.pt')
    pack=torch.load(out/'evidence.pt',weights_only=True,map_location='cpu')
    save(out/'result.json',dict(status='completed',best_epoch=pack['epoch'],metrics=pack['metrics'],checkpoint_sha256=sha(out/'evidence.pt'),
         train_frames=len(train),val_frames=len(val),train_seeds=train.seeds,val_seeds=val.seeds))

if __name__=='__main__':main()
