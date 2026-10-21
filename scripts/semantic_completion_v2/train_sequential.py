"""Matched continuation control versus a trajectory-aware stop objective.

Uses only training trajectories for gradients and the calibration seed for all
selection. The audit seed is inspected after selection. This is development,
not a fresh confirmatory experiment or a full Astra agent evaluation.
"""
import argparse,json,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from assets import save,sha
from completion_model import load_model,THRESHOLDS
from train import assess,read,statistics
from sequential import trajectories,first_stop_loss,score,calibrate


def dataset(paths):
    data = read(paths)
    data['rows'] = [row for p in paths for row in torch.load(p, weights_only=True, map_location='cpu')['rows']]
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True); p.add_argument('--initial', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=30)
    a = p.parse_args(); torch.set_num_threads(4)
    root, out = Path(a.cache), Path(a.output); out.mkdir(parents=True, exist_ok=False)
    train = dataset(sorted(root.glob('train_*.pt')))
    validation = sorted(root.glob('val_*.pt')); assert len(validation) == 2
    cal, audit = dataset(validation[:1]), dataset(validation[1:])
    sequences = [(key, ids) for key, ids in trajectories(train['rows']) if 96121 <= key[0] <= 96128]
    assert len(sequences) == 48
    protocol = dict(initial_sha256=sha(a.initial), cache_manifest_sha256=sha(root/'manifest.json'), epochs=a.epochs,
        calibration=str(validation[0]), audit=str(validation[1]), sequence_training_seeds=list(range(96121,96129)),
        source_sha256=sha(__file__), sequential_source_sha256=sha(Path(__file__).with_name('sequential.py')),
        change='Matched continuation training with and without 0.2 * differentiable first-stop negative log likelihood.',
        calibration_rule='At least 3 first stops and >=90% first-stop precision on calibration trajectories. Maximize correct first stops, then precision.',
        selection='Maximize correctly completed calibration trajectories subject to calibration rule; break ties by balanced pose BCE.',
        limits='Both seeds are development data already inspected for the original model. Audit never used for variant checkpoint/threshold selection. No new confirmatory or Astra result.',
        cloud_calls=0)
    save(out/'protocol.json', protocol)
    base, base_pack = load_model(a.initial, a.device)
    _, cp, _ = assess(base, cal, a.device); selected, curve = calibrate(cp, cal)
    _, ap, _ = assess(base, audit, a.device)
    t = selected['threshold'] if selected else 1.0
    save(out/'recalibration_only.json', dict(calibration=selected, curve=curve, audit=score(ap, audit, t), threshold=t))
    thresholds = torch.tensor(THRESHOLDS, device=a.device)
    e = train['error']; weight = 1+8*((e>.16)&(e<.4)).any(-1).float()+12*(e<=.16).any(-1).float()
    ds = TensorDataset(*[train[k] for k in ['x','error','future_progress','progress_valid']])
    for name, coefficient in [('continuation_control', 0.), ('first_stop', .2)]:
        torch.manual_seed(20261021); rng = random.Random(20261021)
        model, pack = load_model(a.initial, a.device); model.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.02)
        loader = DataLoader(ds, batch_size=64, sampler=WeightedRandomSampler(weight,len(ds),replacement=True))
        dest = out/name; dest.mkdir(); history=[]; best_key=None; started=time.time()
        for epoch in range(a.epochs):
            model.train(); losses=[]; stopping=[]
            for x,error,future,valid in loader:
                x,error,future,valid = [v.to(a.device) for v in (x,error,future,valid)]
                y=model(x); truth=(error[...,None]<=thresholds).float()
                w=1+3*(error<=.16).float()+2*((error>.16)&(error<=.4)).float()
                ordinal=F.binary_cross_entropy_with_logits(y['ordinal'],truth,reduction='none').mean(-1)
                pose=F.smooth_l1_loss(y['log_error'],error.clamp_min(.02).log(),reduction='none')
                loss=((ordinal+.3*pose)*w).sum()/w.sum()
                z=y['ordinal'][...,2]; positive=error<=.16; pair=positive[:,:,None]&~positive[:,None,:]
                if pair.any(): loss+=.2*F.softplus(1-z[:,:,None]+z[:,None,:])[pair].mean()
                loss+=.1*F.relu(y['ordinal'][...,:-1]-y['ordinal'][...,1:]).mean()
                if valid.any(): loss+=.25*F.binary_cross_entropy_with_logits(y['progress'][valid],future[valid])
                # The same sequence forward pass is performed in both arms.
                chosen=rng.sample(sequences,8); lengths=[len(ids) for _,ids in chosen]
                sx=torch.cat([train['x'][ids,key[-1]] for key,ids in chosen]).to(a.device)
                sz=model(sx)['ordinal'][...,2].split(lengths)
                sl=torch.stack([first_stop_loss(logit,(train['error'][ids,key[-1]]<=.16).to(a.device)) for logit,(key,ids) in zip(sz,chosen)]).mean()
                loss=loss+coefficient*sl
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                losses.append(float(loss.detach())); stopping.append(float(sl.detach()))
            metrics, cp, _ = assess(model, cal, a.device); selected, curve = calibrate(cp, cal)
            key=(selected['correct'] if selected else -1, selected['precision'] if selected else 0, -metrics['selection_loss'])
            row=dict(epoch=epoch+1,loss=sum(losses)/len(losses),first_stop_loss=sum(stopping)/len(stopping),calibration=selected,pose=metrics,wall_s=time.time()-started)
            history.append(row); save(dest/'history.json',history)
            if best_key is None or key>best_key:
                best_key=key
                candidate=dict(pack,model={k:v.cpu().clone() for k,v in model.state_dict().items()},epoch=epoch+1,
                    threshold=selected['threshold'] if selected else 1.,calibration=selected,progress_enabled=False,
                    source='trajectory_stop_v2_'+name,cloud_eligible=False)
                torch.save(candidate,dest/'head.pt')
            if epoch%5==0:print(json.dumps(dict(arm=name,**row)),flush=True)
        final, pack = load_model(dest/'head.pt',a.device)
        cm, cp, _=assess(final,cal,a.device); am, ap, _=assess(final,audit,a.device)
        t=pack['threshold']; cs=score(cp,cal,t); aus=score(ap,audit,t)
        admissible=bool(pack['calibration'] and aus['accepted']>=3 and aus['precision']>=.9)
        pack.update(audit=aus,cloud_eligible=False,local_first_stop_gate=admissible)
        torch.save(pack,dest/'head.pt')
        save(dest/'result.json',dict(arm=name,best_epoch=pack['epoch'],threshold=t,calibration=cs,audit=aus,
            frame_audit=statistics(ap,audit['error']<=.16,t),calibration_metrics=cm,audit_metrics=am,
            local_first_stop_gate=admissible,cloud_eligible=False,progress_enabled=False,checkpoint_sha256=sha(dest/'head.pt'),
            note='Recorded-trajectory development result. Requires fresh simulator validation. Progress estimates disabled; no Astra calls.'))
    save(out/'completed.json',dict(status='completed',cloud_calls=0))
if __name__=='__main__':main()
