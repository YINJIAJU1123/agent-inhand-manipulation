"""Local render augmentation at replayed training states. Never uses test seeds.

At each sampled physical state, render multiple disjoint-bank symbol layouts.
This balances symbol identities without falsely treating changed layouts as a
continuous observation history. Output is single-view auxiliary supervision.
"""
import argparse,itertools,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['train','val'],required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from common import *
from latent_data_v4 import split_layouts,visibility_labels

# Same corrected pixel/facing rule as the original collector, without importing
# its executable entry point.
def readable(raw,camera,rgb,without,layout):
    labels=visibility_labels(raw,camera)
    diff=(rgb.float()-without.float()).abs().mean(-1)>20
    changed=diff & ((rgb.max(-1).values-rgb.min(-1).values)<45) & ((rgb.max(-1).values<80)|(rgb.min(-1).values>190))
    fractions=[.315,.78,.3825,.4375,.4,.3825]
    result=torch.zeros(6,6,dtype=torch.bool)
    usable=torch.zeros(6,6,dtype=torch.bool)
    for i in range(6):
        for c,points in enumerate(labels['projected_points'][i]):
            xx,yy=zip(*points);x0,x1=max(0,int(min(xx))-2),min(512,int(max(xx))+3);y0,y1=max(0,int(min(yy))-2),min(512,int(max(yy))+3)
            pixels=int(changed[i,y0:y1,x0:x1].sum()) if x1>x0 and y1>y0 else 0
            expected=labels['projected_area'][i][c]*fractions[layout[i][c]]*(.0075/.009)**2
            result[i,c]=labels['facing'][i][c]>=.5 and pixels>=max(35,.65*expected)
            usable[i,c]=bool(result[i,c]) or pixels<=3 or labels['facing'][i][c]<=0
    return result,usable

out=Path(a.output);out.mkdir(parents=True,exist_ok=False);env=None
try:
    source=Path(a.source);actions=torch.load(source/'actions.pt',weights_only=True,map_location=a.device)
    env,raw,symbols=environment(a.seed,900,a.device)
    original=split_layouts(a.seed,a.split);symbols.set_layout(original)
    bank=np.asarray(list(itertools.permutations(range(6))));bank=bank[np.random.default_rng(31990).permutation(len(bank))]
    lo,hi={'train':(0,480),'val':(480,600)}[a.split];bank=bank[lo:hi]
    rng=np.random.default_rng(a.seed+20261021);active=torch.ones(6,dtype=torch.bool,device=a.device)
    patches=[];geometry=[];labels=[];rows=[];start=time.time()
    with torch.inference_mode():
        for step in range(min(480,len(actions))):
            if step%60==0:
                symbols.update();symbols.markers.set_visibility(False)
                without=raw.capture_camera(refresh=True)['rgb'][...,:3].to(torch.uint8)
                symbols.markers.set_visibility(True)
                coverage=np.zeros((6,6,6),dtype=np.int32)
                for view in range(12):
                    layout=[]
                    for i in range(6):
                        values=coverage[i,np.arange(6)[None],bank].sum(-1)
                        candidates=np.flatnonzero(values==values.min());choice=bank[rng.choice(candidates)]
                        layout.append(choice.tolist());coverage[i,np.arange(6),choice]+=1
                    symbols.set_layout(layout);symbols.update()
                    camera=raw.capture_camera(refresh=True);rgb=camera['rgb'][...,:3].to(torch.uint8)
                    p,g=color_patches(rgb);valid,usable=readable(raw,camera,rgb,without,layout)
                    for i in range(6):
                        if not active[i]:continue
                        patches.append((p[i]*255).byte().cpu());geometry.append(g[i].cpu())
                        labels.append(torch.where(valid[i],torch.tensor(layout[i]),torch.where(usable[i],6,-100)))
                        rows.append(dict(seed=a.seed,step=step,case=i,view=view,split=a.split,layout=layout[i]))
                symbols.set_layout(original);symbols.update();raw.capture_camera(refresh=True)
                save(out/'progress.json',dict(step=step,views=len(rows),active=int(active.sum()),wall_s=time.time()-start))
                print(f'step={step} views={len(rows)}',flush=True)
            _,_,term,trunc,info=env.step(actions[step])
            active &= ~(term|trunc|info['semantic_metrics']['dropped'].bool())
            if not active.any():break
    data=dict(patches=torch.stack(patches),geometry=torch.stack(geometry),labels=torch.stack(labels),rows=rows)
    torch.save(data,out/'data.pt');counts=torch.bincount(data['labels'][data['labels']>=0].flatten(),minlength=7)
    save(out/'manifest.json',dict(status='completed',seed=a.seed,split=a.split,views=len(rows),label_counts=counts.tolist(),ambiguous_labels=int((data['labels']<0).sum()),
        label_rule='Same existing prepared-dataset ink area correction (7.5/9)^2; exclude ambiguous partial evidence.',
        source_actions_sha256=sha(source/'actions.pt'),data_sha256=sha(out/'data.pt'),source_sha256=sha(__file__),
        scope='Single-view counterfactual symbol-layout rendering at replayed physical states. Independent of formal eval. Not action-value supervision.'))
except BaseException:
    import traceback
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
