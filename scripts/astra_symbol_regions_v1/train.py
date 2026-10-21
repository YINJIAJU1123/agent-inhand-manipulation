"""Train RGB sticker localization with glyph supervision; select on calibration only."""
import argparse, json, sys, time, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha
from model import StickerReader,losses


class Views(Dataset):
    def __init__(self,paths):
        self.rgb=[];self.boxes=[];self.labels=[];self.seeds=set()
        for path in paths:
            d=torch.load(path,map_location='cpu',weights_only=True)
            self.rgb.append(d['rgb']);self.boxes.append(d['boxes']);self.labels.append(d['labels'])
            self.seeds.update(r['seed']for r in d['rows'])
        if not self.rgb:raise ValueError('No input RGB shards')
        self.rgb=torch.cat(self.rgb);self.boxes=torch.cat(self.boxes);self.labels=torch.cat(self.labels)
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return self.rgb[i],self.boxes[i],self.labels[i]


def iou(a,b):
    lo=torch.maximum(a[...,:2]-a[...,2:]/2,b[...,:2]-b[...,2:]/2)
    hi=torch.minimum(a[...,:2]+a[...,2:]/2,b[...,:2]+b[...,2:]/2)
    intersection=(hi-lo).clamp_min(0).prod(-1)
    return intersection/(a[...,2:].prod(-1)+b[...,2:].prod(-1)-intersection).clamp_min(1e-8)


@torch.inference_mode()
def predict(model,loader,device):
    model.eval();p=[];t=[];overlap=[]
    for rgb,boxes,labels in loader:
        y=model(rgb.to(device));p.append(y['probability'].cpu());t.append(labels)
        overlap.append(iou(y['boxes'].cpu(),boxes))
    return torch.cat(p),torch.cat(t),torch.cat(overlap)


def metrics(probability,labels,overlap,threshold):
    usable=labels>=0;known=usable&(labels<6);prediction=probability.argmax(-1)
    confidence,selected=probability[...,:6].max(-1);accepted=(confidence>=threshold)&usable
    by_class=[float((prediction[labels==c]==c).float().mean())if(labels==c).any()else None for c in range(6)]
    correct=accepted&(selected==labels)
    return dict(views=len(labels),readable=int(known.sum()),per_class=by_class,
                macro_readable=sum(v for v in by_class if v is not None)/sum(v is not None for v in by_class),
                accepted=int(accepted.sum()),precision=float(correct.sum()/accepted.sum())if accepted.any()else None,
                unknown_false_accepts=int((accepted&~known).sum()),readable_mean_box_iou=float(overlap[known].mean()),
                readable_box_iou50=float((overlap[known]>=.5).float().mean()),threshold=threshold)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(10423)
    paths=sorted(a.data.glob('train_*/data.pt'));calpath=a.data/'val_96211/data.pt';auditpath=a.data/'val_96212/data.pt'
    train=Views(paths);cal=Views([calpath]);audit_seeds={r['seed']for r in torch.load(auditpath,map_location='cpu',weights_only=True)['rows']}
    assert not(train.seeds&cal.seeds or train.seeds&audit_seeds or cal.seeds&audit_seeds)
    counts=torch.bincount(train.labels[(train.labels>=0)&(train.labels<6)],minlength=6).float();assert(counts>0).all()
    known=(train.labels>=0)&(train.labels<6);weights=((1/counts)[train.labels.clamp(0,5)]*known).sum(-1)+.05/counts.max()
    sampler=WeightedRandomSampler(weights,2048,replacement=True,generator=torch.Generator().manual_seed(10423))
    loader=DataLoader(train,batch_size=12,sampler=sampler,num_workers=0)
    cal_loader=DataLoader(cal,batch_size=12);model=StickerReader().to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.02)
    source_hash={name:sha(Path(__file__).with_name(name))for name in ['model.py','train.py','collect.py','labels.py']}
    save(out/'protocol.json',dict(train_files=[str(x)for x in paths],train_seeds=sorted(train.seeds),calibration_seeds=sorted(cal.seeds),audit_seeds=sorted(audit_seeds),
         train_views=len(train),calibration_views=len(cal),epochs=a.epochs,source_sha256=source_hash,cloud_calls=0,
         input='Measured native RGB only. Deployed forward predicts regions and symbols without labels, projection, layout, pose, or reference crop.',
         supervision='Training-only projected glyph boxes, readability and symbol identity. Ground-truth crops train an auxiliary classifier loss only.',
         selection='Maximum calibration readable macro accuracy. Calibration threshold requires >=50 accepted predictions and >=95% precision. Audit evaluated after selection only.',
         scope='Localization repair on development seeds. No autonomous task success claim.'))
    best=-1.;history=[];start=time.time()
    try:
        for epoch in range(1,a.epochs+1):
            model.train();values=[]
            for rgb,boxes,labels in loader:
                rgb=rgb.to(a.device).float()/255.;boxes=boxes.to(a.device).clone();labels=labels.to(a.device)
                if not((labels>=0)&(labels<6)).any():continue
                flip=torch.rand(len(rgb),device=a.device)<.5;rgb[flip]=rgb[flip].flip(-1);boxes[flip,:,0]=1-boxes[flip,:,0]
                rgb=(rgb*(.85+.3*torch.rand(len(rgb),1,1,1,device=a.device))).clamp(0,1)
                prediction=model(rgb);loss,parts=losses(model,rgb,boxes,labels,prediction,predicted_weight=0. if epoch<=5 else .3)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();values.append(float(loss.detach()))
            pr,lab,ov=predict(model,cal_loader,a.device);m=metrics(pr,lab,ov,.7)
            row=dict(epoch=epoch,loss=sum(values)/len(values),calibration=m,elapsed_s=time.time()-start);history.append(row)
            save(out/'progress.json',dict(status='training',**row));save(out/'history.json',history);print('SYMBOL_REGION_EPOCH '+json.dumps(row),flush=True)
            if m['macro_readable']>best:
                best=m['macro_readable'];torch.save(dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},epoch=epoch,cloud_eligible=False,input='native_rgb_only'),out/'head.pt')
        pack=torch.load(out/'head.pt',weights_only=True,map_location=a.device);model.load_state_dict(pack['model'])
        pr,lab,ov=predict(model,cal_loader,a.device)
        curve=[metrics(pr,lab,ov,t)for t in [.5,.6,.7,.8,.85,.9,.95,.98,.99]]
        eligible=[m for m in curve if m['accepted']>=50 and(m['precision']or 0)>=.95]
        selected=max(eligible,key=lambda m:m['accepted'])if eligible else None;threshold=selected['threshold']if selected else 1.1
        audit=Views([auditpath]);pr,lab,ov=predict(model,DataLoader(audit,batch_size=12),a.device);am=metrics(pr,lab,ov,threshold)
        passed=bool(selected and am['accepted']>=50 and(am['precision']or 0)>=.95 and am['macro_readable']>=.75 and am['readable_mean_box_iou']>=.6)
        pack.update(threshold=threshold,offline_gate_passed=passed);pack['model']={k:v.cpu()for k,v in pack['model'].items()};torch.save(pack,out/'head.pt')
        result=dict(status='completed',best_epoch=pack['epoch'],calibration=selected,calibration_curve=curve,audit=am,offline_gate_passed=passed,
                    checkpoint_sha256=sha(out/'head.pt'),cloud_eligible=False,cloud_calls=0,note='Requires real-video sensor-only shadow and fresh-seed validation before integration; no semantic task-success claim.')
        save(out/'result.json',result);save(out/'progress.json',dict(status='completed',offline_gate_passed=passed,elapsed_s=time.time()-start));print('SYMBOL_REGION_RESULT '+json.dumps(result),flush=True)
    except BaseException:
        save(out/'failure.json',dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
