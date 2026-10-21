"""Goal-paired completion learning with independent calibration and audit seeds."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from assets import save,sha
from completion_model import CompletionModel,THRESHOLDS

def read(paths):
    ds=[torch.load(p,weights_only=True,map_location='cpu') for p in paths]
    return {k:torch.cat([d[k] for d in ds]) for k in ['x','error','future_progress','progress_valid']}

def statistics(prob,true,threshold):
    accepts=prob>=threshold;n=int(accepts.sum());correct=int((accepts&true).sum());positive=int(true.sum())
    return dict(accepted=n,correct=correct,false_accepts=n-correct,precision=correct/n if n else None,recall=correct/positive if positive else None,positive_labels=positive)

@torch.inference_mode()
def assess(model,data,device):
    model.eval();ys=[]
    for x in data['x'].split(128):ys.append({k:v.cpu() for k,v in model(x.to(device)).items()})
    y={k:torch.cat([d[k] for d in ys]) for k in ys[0]}
    error=data['error'];true=error<=.16;p=y['ordinal'][...,2].sigmoid();near=(error>.16)&(error<=.5)
    bce=F.binary_cross_entropy_with_logits(y['ordinal'][...,2],true.float(),reduction='none')
    loss=.5*bce[true].mean()+.5*bce[near].mean()
    metrics=dict(selection_loss=float(loss),completion_brier=float((p-true.float()).square().mean()),log_error_mae=float((y['log_error']-error.clamp_min(.02).log()).abs().mean()),
        positives=int(true.sum()),near_negatives=int(near.sum()))
    return metrics,p,y

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=80);p.add_argument('--kind',choices=['mlp','linear'],default='mlp');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261021);device=a.device;root=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    train_files=sorted(root.glob('train_*.pt'));val_files=sorted(root.glob('val_*.pt'));assert len(val_files)==2
    train,cal,audit=read(train_files),read(val_files[:1]),read(val_files[1:])
    model=CompletionModel(kind=a.kind).to(device);model.mean.copy_(train['x'].flatten(0,1).mean(0));model.scale.copy_(train['x'].flatten(0,1).std(0).clamp_min(.02))
    error=train['error'];weight=1+8*((error>.16)&(error<.4)).any(-1).float()+12*(error<=.16).any(-1).float()
    ds=TensorDataset(*[train[k] for k in ['x','error','future_progress','progress_valid']])
    loader=DataLoader(ds,batch_size=64,sampler=WeightedRandomSampler(weight,len(ds),replacement=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02);thresholds=torch.tensor(THRESHOLDS,device=device)
    history=[];best=float('inf');start=time.time()
    save(out/'protocol.json',dict(train_files=[str(x) for x in train_files],calibration_file=str(val_files[0]),audit_file=str(val_files[1]),
        epochs=a.epochs,kind=a.kind,cache_manifest_sha256=sha(root/'manifest.json'),source_sha256=sha(__file__),
        model_source_sha256=sha(Path(__file__).parent/'completion_model.py'),feature_source_sha256=sha(Path(__file__).parent/'features.py'),
        selection='Lowest balanced completion log loss on calibration seed. Fit threshold on that seed only, then report audit seed once.',
        goal_pairing='All six candidate language goals evaluated at the same recorded physical state; positive versus wrong-target ranking.',
        labels='Exact full-quaternion error; ready <=0.16 radians. Near misses (0.16,0.4) receive higher loss/sample weight. Policy stays frozen.',
        limits='Conditional on Astra binding. Cannot detect an incorrect symbol-to-color binding from pose alone. Development pilot, not task success.'))
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,e,future,valid in loader:
            x,e,future,valid=[v.to(device) for v in [x,e,future,valid]];y=model(x)
            truth=(e[...,None]<=thresholds).float();w=1+3*(e<=.16).float()+2*((e>.16)&(e<=.4)).float()
            ordinal=F.binary_cross_entropy_with_logits(y['ordinal'],truth,reduction='none').mean(-1)
            pose=F.smooth_l1_loss(y['log_error'],e.clamp_min(.02).log(),reduction='none')
            loss=((ordinal+.3*pose)*w).sum()/w.sum()
            logits=y['ordinal'][...,2];positive=e<=.16
            pair=positive[:,:,None]&~positive[:,None,:]
            if pair.any():loss+=.2*F.softplus(1-logits[:,:,None]+logits[:,None,:])[pair].mean()
            loss+=.1*F.relu(y['ordinal'][...,:-1]-y['ordinal'][...,1:]).mean()
            if valid.any():loss+=.25*F.binary_cross_entropy_with_logits(y['progress'][valid],future[valid])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        metrics,_,_=assess(model,cal,device);row=dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),calibration=metrics,wall_s=time.time()-start)
        history.append(row);save(out/'history.json',history)
        if epoch%5==0:print(json.dumps(row),flush=True)
        if metrics['selection_loss']<best:
            best=metrics['selection_loss'];torch.save(dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},epoch=epoch+1,kind=a.kind),out/'head.pt')
    pack=torch.load(out/'head.pt',weights_only=True,map_location='cpu');model.load_state_dict(pack['model'])
    cm,cp,_=assess(model,cal,device);true=cal['error']<=.16
    curve=[dict(threshold=t,**statistics(cp,true,t)) for t in [.5,.6,.7,.8,.85,.9,.95,.975,.99,.995,.999]]
    candidates=[r for r in curve if r['accepted']>=10 and r['precision']>=.9]
    selected=max(candidates,key=lambda r:(r['recall'],r['precision'])) if candidates else None
    threshold=selected['threshold'] if selected else 1.0
    am,ap,ay=assess(model,audit,device);a_stats=statistics(ap,audit['error']<=.16,threshold)
    per_goal=[statistics(ap[:,i],audit['error'][:,i]<=.16,threshold) for i in range(6)]
    valid=audit['progress_valid'];pt=audit['future_progress'][valid];pp=ay['progress'][valid].sigmoid()
    progress_metrics=dict(examples=int(valid.sum()),brier=float((pp-pt).square().mean()),constant_base_brier=float((pt-pt.mean()).square().mean()))
    positive=audit['error']<=.16;ids=positive.any(-1);correct_goal=audit['error'].argmin(-1)
    goal_ranking=float((ap.argmax(-1)[ids]==correct_goal[ids]).float().mean()) if ids.any() else None
    eligible=bool(selected and a_stats['accepted']>=10 and a_stats['precision']>=.9 and a_stats['recall']>=.15)
    pack.update(threshold=threshold,calibration=selected,audit=a_stats,cloud_eligible=eligible,feature_manifest=json.loads((root/'manifest.json').read_text()),source='frozen_policy_goal_conditioned_completion_v2')
    torch.save(pack,out/'head.pt')
    save(out/'result.json',dict(status='completed',best_epoch=pack['epoch'],threshold=threshold,calibration_curve=curve,calibration_metrics=cm,
         calibrated=bool(selected),audit=a_stats,audit_metrics=am,audit_per_goal=per_goal,goal_ranking_accuracy_on_reached=goal_ranking,
         progress_metrics=progress_metrics,cloud_eligible=eligible,checkpoint_sha256=sha(out/'head.pt'),
         note='Conditional goal completion only. No Astra request or autonomous-task success measurement. Audit seed not used for selection.'))
    print(json.dumps(json.loads((out/'result.json').read_text())),flush=True)
if __name__=='__main__':main()
