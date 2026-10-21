"""Held-out diagnostics only. Does not read formal test episodes or tune weights."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train import Frames,predict
from model import load_model
from common import save

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--learning',required=True);a=p.parse_args()
    torch.set_num_threads(4);root=Path(a.learning)
    model,pack=load_model(root/'train/evidence.pt','cpu')
    rows=[]
    for path in sorted((root/'data').glob('val_*/data.pt')):
        ds=Frames([path]);ys=[];bs=[]
        for b in DataLoader(ds,batch_size=32):
            ys.append({k:v for k,v in predict(model,b).items() if k in ('symbol','ready','future_visible','survival')})
            bs.append({k:v for k,v in b.items() if k in ('symbol','error','future_valid','skill','future','survived')})
        y={k:torch.cat([x[k] for x in ys]) for k in ys[0]};b={k:torch.cat([x[k] for x in bs]) for k in bs[0]}
        prob=y['symbol'].softmax(-1);confidence,pred=prob[...,:6].max(-1)
        accepted=(confidence>=pack['thresholds']['symbol'])&(b['symbol']>=0);visible=(b['symbol']>=0)&(b['symbol']!=6)
        ready=y['ready'].sigmoid()>=pack['thresholds']['ready'];true=b['error']<=.16
        def frac(num,den):return num/den if den else None
        mask=b['future_valid'];future=y['future_visible'][mask,b['skill'][mask]].sigmoid();truth=b['future'][mask]
        rows.append(dict(seed=ds.seeds[0],frames=len(ds),readable=int(visible.sum()),
            symbol_accuracy_readable=frac(int(((y['symbol'].argmax(-1)==b['symbol'])&visible).sum()),int(visible.sum())),
            symbol_top_six_accuracy_readable=frac(int(((pred==b['symbol'])&visible).sum()),int(visible.sum())),
            confident_symbols=int(accepted.sum()),confident_readable=int((accepted&visible).sum()),
            confident_symbol_precision=frac(int(((pred==b['symbol'])&accepted).sum()),int(accepted.sum())),
            completion_labels=int(true.sum()),completion_accepts=int(ready.sum()),
            completion_true_accepts=int((ready&true).sum()),completion_false_accepts=int((ready&~true).sum()),
            completion_precision=frac(int((ready&true).sum()),int(ready.sum())),
            completion_recall=frac(int((ready&true).sum()),int(true.sum())),
            future_examples=int(mask.sum()),future_brier=float((future-truth).square().mean()),
            never_visible_brier=float(truth.square().mean()),always_visible_brier=float((1-truth).square().mean())))
    result=dict(checkpoint_epoch=pack['epoch'],thresholds=pack['thresholds'],per_validation_seed=rows,
        note='Correlated frames; empirical diagnostics, not statistical confidence intervals. No test-set threshold selection.')
    save(root/'train/validation_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
