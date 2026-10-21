"""Keep native RGB and train-only symbol boxes to repair measured-crop failures."""
import argparse, itertools, json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic'),str(ROOT/'scripts/semantic_completion_v1')]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--split',choices=['train','val'],required=True)
p.add_argument('--max-steps',type=int,default=480)
p.add_argument('--views',type=int,default=12)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.enable_cameras=True
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import numpy as np
import torch
from PIL import Image,ImageDraw
from common import environment,save,sha,color_patches,COLORS,SYMBOLS
from latent_data_v4 import split_layouts,visibility_labels
from labels import readable

out=Path(a.output);out.mkdir(parents=True,exist_ok=False);env=None
try:
    source=Path(a.source);actions=torch.load(source/'actions.pt',weights_only=True,map_location=a.device)
    env,raw,symbols=environment(a.seed,900,a.device)
    original=split_layouts(a.seed,a.split);symbols.set_layout(original)
    bank=np.asarray(list(itertools.permutations(range(6))))
    bank=bank[np.random.default_rng(31990).permutation(len(bank))]
    lo,hi={'train':(0,480),'val':(480,600)}[a.split];bank=bank[lo:hi]
    rng=np.random.default_rng(a.seed+20261021);active=torch.ones(6,dtype=torch.bool,device=a.device)
    images=[];boxes=[];labels=[];rows=[];geometry=[];start=time.time();samples=[]
    with torch.inference_mode():
        for step in range(min(a.max_steps,len(actions))):
            if step%60==0:
                symbols.update();symbols.markers.set_visibility(False)
                without=raw.capture_camera(refresh=True)['rgb'][...,:3].to(torch.uint8)
                symbols.markers.set_visibility(True)
                coverage=np.zeros((6,6,6),dtype=np.int32)
                for view in range(a.views):
                    layout=[]
                    for i in range(6):
                        score=coverage[i,np.arange(6)[None],bank].sum(-1)
                        candidates=np.flatnonzero(score==score.min());choice=bank[rng.choice(candidates)]
                        layout.append(choice.tolist());coverage[i,np.arange(6),choice]+=1
                    symbols.set_layout(layout);symbols.update()
                    camera=raw.capture_camera(refresh=True);rgb=camera['rgb'][...,:3].to(torch.uint8)
                    valid,usable=readable(raw,camera,rgb,without,layout)
                    projection=visibility_labels(raw,camera)
                    points=torch.tensor(projection['projected_points'])
                    lower=points.min(2).values;upper=points.max(2).values
                    center=(lower+upper)/2
                    # Include the color around the glyph. These are labels only.
                    extent=(upper-lower)*1.7+4
                    lower=(center-extent/2).clamp(0,511);upper=(center+extent/2).clamp(1,512)
                    center=(lower+upper)/2;extent=(upper-lower).clamp_min(1)
                    target_boxes=torch.cat((center,extent),-1)/512
                    _,observed_geometry=color_patches(rgb)
                    for i in range(6):
                        if not active[i]:continue
                        lab=torch.where(valid[i],torch.tensor(layout[i]),torch.where(usable[i],6,-100))
                        images.append(rgb[i].permute(2,0,1).cpu().clone());boxes.append(target_boxes[i]);labels.append(lab)
                        geometry.append(observed_geometry[i].cpu())
                        rows.append(dict(seed=a.seed,step=step,case=i,view=view,split=a.split))
                        if len(samples)<12 and valid[i].any():
                            im=Image.fromarray(rgb[i].cpu().numpy());draw=ImageDraw.Draw(im)
                            for c in range(6):
                                if valid[i,c]:
                                    b=target_boxes[i,c]*512;x,y,w,h=b.tolist()
                                    draw.rectangle((x-w/2,y-h/2,x+w/2,y+h/2),outline='white',width=2)
                                    draw.text((max(0,x-w/2),max(0,y-h/2-14)),COLORS[c]+' '+SYMBOLS[int(lab[c])],fill='white',stroke_width=1,stroke_fill='black')
                            samples.append(im)
                symbols.set_layout(original);symbols.update();raw.capture_camera(refresh=True)
                save(out/'progress.json',dict(status='collecting',step=step,views=len(rows),active=int(active.sum()),wall_s=time.time()-start))
            _,_,term,trunc,info=env.step(actions[step])
            active &= ~(term|trunc|info['semantic_metrics']['dropped'].bool())
            if not active.any():break
    data=dict(rgb=torch.stack(images),boxes=torch.stack(boxes),labels=torch.stack(labels),rows=rows,legacy_geometry=torch.stack(geometry))
    assert data['rgb'].dtype==torch.uint8 and data['rgb'].shape[1:]==(3,512,512)
    assert torch.isfinite(data['boxes']).all() and ((data['boxes']>=0)&(data['boxes']<=1)).all()
    torch.save(data,out/'data.pt')
    known=(data['labels']>=0)&(data['labels']<6)
    counts=torch.bincount(data['labels'][data['labels']>=0].flatten(),minlength=7)
    canvas=Image.new('RGB',(1024,((len(samples)+3)//4)*256))
    for i,im in enumerate(samples):canvas.paste(im.resize((256,256)),((i%4)*256,(i//4)*256))
    canvas.save(out/'label_preview.png')
    manifest=dict(status='completed',seed=a.seed,split=a.split,views=len(rows),rgb_shape=list(data['rgb'].shape),label_counts=counts.tolist(),readable=int(known.sum()),
                  legacy_empty_readable=int((known&(data['legacy_geometry'][...,0]==0)).sum()),ambiguous=int((data['labels']<0).sum()),
                  source_actions_sha256=sha(source/'actions.pt'),data_sha256=sha(out/'data.pt'),source_sha256=sha(__file__),labels_source_sha256=sha(Path(__file__).with_name('labels.py')),
                  cloud_calls=0,scope='Native measured RGB is the only deployed model input. Privileged projected boxes and visibility are training labels, never runtime observations. Old development seeds, not final evaluation.')
    save(out/'manifest.json',manifest);save(out/'progress.json',dict(status='completed',views=len(rows),wall_s=time.time()-start));print('SYMBOL_REGION_DATA '+json.dumps(manifest),flush=True)
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
