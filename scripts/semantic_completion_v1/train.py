"""Train on complete held-out-seed splits; never inspect formal eval labels."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from common import save,sha
from model import EvidenceModel


class Frames(Dataset):
    def __init__(self, paths):
        self.shards = []
        self.index = []
        self.seeds = []
        for path in paths:
            d = torch.load(path,map_location="cpu",weights_only=True)
            rows = d["rows"]
            self.seeds.append(rows[0]["seed"])
            histories = []
            for j,r in enumerate(rows):
                hist = [k for k in range(max(0,j-24),j+1)
                        if rows[k]["case"]==r["case"]][-4:]
                histories.append([hist[0]]*(4-len(hist))+hist)
            d["histories"] = histories
            shard = len(self.shards)
            self.shards.append(d)
            self.index.extend((shard,j) for j in range(len(rows)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s,j = self.index[idx]
        d = self.shards[s]
        h = d["histories"][j]
        seen = d["readable"][h].any(0)
        usable = d.get("usable",torch.ones_like(d["readable"]))[h].all(0)
        labels = torch.where(seen,d["layout"][j],torch.where(usable,6,-100))
        return dict(patches=d["patches"][h].float()/255.,
                    geometry=d["geometry"][h],
                    features=d["features"][j,-4:],
                    proprio=d["proprio"][j,-4:],
                    symbol=labels,visible=d["readable"][j].float(),
                    error=d["error"][j],skill=torch.tensor(d["rows"][j]["skill"]),
                    future=d["future_readable"][j],
                    future_valid=d["future_valid"][j],
                    survived=d["survived"][j].float())


def predict(model,b):
    return model(b["patches"],b["geometry"],b["features"],b["proprio"])


def loss_fn(y,b):
    ready = (b["error"]<=.16).float()
    classification = F.cross_entropy(y["symbol"].flatten(0,1),b["symbol"].flatten(),
                                    weight=torch.tensor([1.]*6+[.35],device=ready.device))
    pose = F.smooth_l1_loss(y["log_error"],torch.log(b["error"].clamp_min(.02)))
    completion = F.binary_cross_entropy_with_logits(y["ready"],ready,
                                                    pos_weight=torch.tensor(12.,device=ready.device))
    visible = F.binary_cross_entropy_with_logits(y["visible"],b["visible"])
    valid = b["future_valid"]
    future_loss = ready.sum()*0.
    if valid.any():
        v = y["future_visible"][valid,b["skill"][valid]]
        s = y["survival"][valid,b["skill"][valid]]
        future_loss = F.binary_cross_entropy_with_logits(v,b["future"][valid]) + F.binary_cross_entropy_with_logits(s,b["survived"][valid])
    return classification + pose + .5*completion + .3*visible + .5*future_loss


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval()
    losses, ys, bs = [], [], []
    for batch in loader:
        b={k:v.to(device) for k,v in batch.items()}
        y=predict(model,b)
        losses.append(float(loss_fn(y,b)))
        ys.append({k:v.cpu() for k,v in y.items()})
        bs.append({k:v.cpu() for k,v in b.items() if k in ("symbol","error","future_valid","skill","future","survived")})
    y={k:torch.cat([r[k] for r in ys]) for k in ys[0]}
    b={k:torch.cat([r[k] for r in bs]) for k in bs[0]}
    visible=(b["symbol"]>=0)&(b["symbol"]!=6)
    true=b["error"]<=.16
    probability=y["ready"].sigmoid()
    # Select the lowest validation threshold meeting a prespecified precision,
    # otherwise retain 0.95 and disclose the lack of a reliable gate.
    choices=[]
    for threshold in [.5,.6,.7,.8,.85,.9,.95,.98,.99]:
        accepted=probability>=threshold
        n=int(accepted.sum())
        precision=float(true[accepted].float().mean()) if n else None
        if n>=10 and precision>=.9:
            choices.append((threshold,n,precision))
    selected=choices[0] if choices else (.95,0,None)
    mask=b["future_valid"]
    view=y["future_visible"][mask,b["skill"][mask]].sigmoid()
    metrics=dict(loss=sum(losses)/len(losses),
        symbol_accuracy_readable=float((y["symbol"].argmax(-1)[visible]==b["symbol"][visible]).float().mean()) if visible.any() else None,
        pose_log_mae=float((y["log_error"]-b["error"].clamp_min(.02).log()).abs().mean()),
        ready_brier=float((probability-true.float()).square().mean()),
        positive_completion_labels=int(true.sum()),
        future_visibility_brier=float((view-b["future"][mask]).square().mean()) if mask.any() else None,
        gate_threshold=selected[0],gate_validation_count=selected[1],gate_precision=selected[2],
        gate_calibrated=bool(choices))
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261020)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    train_paths=sorted(Path(a.data).glob("train_*/data.pt"))
    val_paths=sorted(Path(a.data).glob("val_*/data.pt"))
    train,val=Frames(train_paths),Frames(val_paths)
    assert train_paths and val_paths and not set(train.seeds)&set(val.seeds)
    model=EvidenceModel().to(a.device)
    sensors=torch.cat([torch.cat((d["features"][:,-4:],d["proprio"][:,-4:]),-1).flatten(0,1) for d in train.shards])
    model.sensor_mean.copy_(sensors.mean(0))
    model.sensor_std.copy_(sensors.std(0).clamp_min(.03))
    dl=DataLoader(train,batch_size=32,shuffle=True,num_workers=0)
    vl=DataLoader(val,batch_size=64,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    history=[];best=float("inf");started=time.time()
    for epoch in range(a.epochs):
        model.train();losses=[]
        for batch in dl:
            b={k:v.to(a.device) for k,v in batch.items()}
            opt.zero_grad();loss=loss_fn(predict(model,b),b)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,vl,a.device)
        row=dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),validation=metrics,wall_s=time.time()-started)
        history.append(row);save(out/"history.json",history);print(json.dumps(row),flush=True)
        if metrics["loss"]<best:
            best=metrics["loss"]
            pack=dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},
                epoch=epoch+1,metrics=metrics,train_seeds=train.seeds,val_seeds=val.seeds,
                source="sensor_only_temporal_completion_v1",
                dataset_sha256={str(p):sha(p) for p in train_paths+val_paths},
                thresholds=dict(ready=metrics["gate_threshold"],symbol=.7),
                limits="Pilot; future-head predicts 90-step readability/survival under random skill rollouts, not final task value")
            torch.save(pack,out/"evidence.pt")
    pack=torch.load(out/"evidence.pt",map_location="cpu",weights_only=True)
    save(out/"result.json",dict(status="completed",train_frames=len(train),val_frames=len(val),
        best_epoch=pack["epoch"],metrics=pack["metrics"],checkpoint_sha256=sha(out/"evidence.pt"),
        train_seeds=train.seeds,val_seeds=val.seeds))


if __name__=="__main__":
    main()
