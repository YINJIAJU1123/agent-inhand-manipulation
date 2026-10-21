"""Fit only glyph identity, with affine/projective view variation and no color input."""
import argparse,hashlib,importlib.util,json,math,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from model import StickerReader,crop_regions
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reader_training_metrics',ROOT/'scripts/astra_symbol_regions_v1/train.py')
metrics_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_module)
metrics=metrics_module.metrics;predict=metrics_module.predict;iou=metrics_module.iou
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def augment(x):
    n=len(x);angle=torch.rand(n,device=x.device)*2*math.pi
    scale=.7+.6*torch.rand(n,2,device=x.device)
    shear=(torch.rand(n,device=x.device)-.5)*.6
    c,s=angle.cos(),angle.sin();matrix=x.new_zeros(n,2,3)
    matrix[:,0,0]=c*scale[:,0];matrix[:,1,0]=s*scale[:,0]
    matrix[:,0,1]=c*shear-s*scale[:,1];matrix[:,1,1]=s*shear+c*scale[:,1]
    matrix[:,:,2]=(torch.rand(n,2,device=x.device)-.5)*.18
    grid=F.affine_grid(matrix,x.shape,align_corners=False)
    perspective=(torch.rand(n,1,1,2,device=x.device)-.5)*.4
    grid=grid/(1+(grid*perspective).sum(-1,keepdim=True)).clamp_min(.5)
    x=F.grid_sample(x,grid,padding_mode='border',align_corners=False)
    small=int(torch.randint(16,65,()).item())
    x=F.interpolate(F.interpolate(x,(small,small),mode='bilinear',align_corners=False),(64,64),mode='bilinear',align_corners=False)
    if bool(torch.rand(())<.5):x=F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)
    return (x+.015*torch.randn_like(x)).clamp(0,1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(10523);device='cuda:0'
    model=StickerReader().to(device).eval();model.load_state_dict(torch.load(a.base,map_location='cpu',weights_only=True)['model'])
    for param in model.parameters():param.requires_grad_(False)
    for module in (model.glyph,model.symbol):
        for param in module.parameters():param.requires_grad_(True)
    examples=[];targets=[];seeds=set();started=time.time()
    with torch.inference_mode():
        for path in sorted(a.data.glob('train_*/data.pt')):
            d=torch.load(path,map_location='cpu',weights_only=True);seeds.update(r['seed']for r in d['rows'])
            for begin in range(0,len(d['rgb']),8):
                rgb=d['rgb'][begin:begin+8].to(device);boxes=d['boxes'][begin:begin+8].to(device);labels=d['labels'][begin:begin+8].to(device)
                known=(labels>=0)&(labels<6);prediction=model(rgb)
                truth=crop_regions(rgb.float()/255.,boxes).reshape(len(rgb),6,3,64,64)
                predicted=prediction['crops'].reshape(len(rgb),6,3,64,64);matched=known&(iou(prediction['boxes'],boxes)>=.5)
                for images,mask in [(truth,known),(predicted,matched)]:
                    examples.append((images[mask]*255).round().clamp(0,255).to(torch.uint8).cpu());targets.append(labels[mask].cpu())
    x,y=torch.cat(examples),torch.cat(targets);counts=torch.bincount(y,minlength=6).float();assert(counts>0).all()
    cal=metrics_module.Views([a.data/'val_96211/data.pt']);audit=metrics_module.Views([a.data/'val_96212/data.pt'])
    assert not(seeds&cal.seeds or seeds&audit.seeds or cal.seeds&audit.seeds)
    sampler=WeightedRandomSampler((1/counts)[y],4096,replacement=True,generator=torch.Generator().manual_seed(10523))
    loader=DataLoader(TensorDataset(x,y),batch_size=64,sampler=sampler);cal_loader=DataLoader(cal,batch_size=12)
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=4e-4,weight_decay=.02)
    sources={name:sha(Path(__file__).with_name(name))for name in ('model.py','train.py')}
    save(out/'protocol.json',dict(base_sha256=sha(a.base),train_seeds=sorted(seeds),calibration_seeds=sorted(cal.seeds),audit_seeds=sorted(audit.seeds),training_crops=len(x),class_counts=counts.tolist(),epochs=a.epochs,source_sha256=sources,base_model_source_sha256=sha(ROOT/'scripts/astra_symbol_regions_v1/model.py'),cloud_calls=0,selection='Calibration readable macro accuracy; no video labels are used in fitting or selection.',changes='Frozen RGB localization and presence. Only glyph identity learns grayscale contrast-normalized affine/projective, resolution and blur variation. Glyph color query is zero.',scope='Repeated development split; neither audit nor actual video is a final unseen test. Requires fresh-seed validation before deployment.'))
    best=-1.;history=[]
    try:
        for epoch in range(1,a.epochs+1):
            model.eval();model.glyph.train();losses=[]
            for rgb,label in loader:
                rgb=augment(rgb.to(device).float()/255.);label=label.to(device)
                logits=model.glyph_logits(rgb);loss=F.cross_entropy(logits,label,label_smoothing=.03)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite glyph loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
            prob,lab,overlap=predict(model,cal_loader,device);m=metrics(prob,lab,overlap,.7)
            row=dict(epoch=epoch,loss=sum(losses)/len(losses),calibration=m,elapsed_s=time.time()-started);history.append(row);save(out/'progress.json',dict(status='training',**row));save(out/'history.json',history)
            print('SYMBOL_INVARIANT_EPOCH '+json.dumps(row),flush=True)
            if m['macro_readable']>best:
                best=m['macro_readable'];torch.save(dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},epoch=epoch,cloud_eligible=False,input='native_rgb_only',architecture='color_independent_glyph_v1'),out/'head.pt')
        pack=torch.load(out/'head.pt',map_location=device,weights_only=True);model.load_state_dict(pack['model']);model.eval()
        pr,lab,ov=predict(model,cal_loader,device);curve=[metrics(pr,lab,ov,t)for t in [.5,.6,.7,.8,.85,.9,.95,.98,.99]]
        admissible=[m for m in curve if m['accepted']>=50 and(m['precision']or 0)>=.95];selected=max(admissible,key=lambda m:m['accepted'])if admissible else None;threshold=selected['threshold']if selected else 1.1
        pr,lab,ov=predict(model,DataLoader(audit,batch_size=12),device);am=metrics(pr,lab,ov,threshold)
        passed=bool(selected and am['accepted']>=50 and(am['precision']or 0)>=.95 and am['macro_readable']>=.75 and am['readable_mean_box_iou']>=.6)
        pack.update(threshold=threshold,offline_gate_passed=passed);pack['model']={k:v.cpu()for k,v in pack['model'].items()};torch.save(pack,out/'head.pt')
        result=dict(status='completed',best_epoch=pack['epoch'],calibration=selected,calibration_curve=curve,audit=am,offline_gate_passed=passed,checkpoint_sha256=sha(out/'head.pt'),cloud_eligible=False,cloud_calls=0,note='Development learning candidate. Actual recorded-video shadow and fresh-seed validation are still required.')
        save(out/'result.json',result);save(out/'progress.json',dict(status='completed',offline_gate_passed=passed));print('SYMBOL_INVARIANT_RESULT '+json.dumps(result),flush=True)
    except BaseException:
        save(out/'failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
