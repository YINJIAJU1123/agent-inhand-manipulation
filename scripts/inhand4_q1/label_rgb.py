"""Depth-tested glyph-surface supervision, separated from model RGB inputs."""
import argparse
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from goals import NORMALS
from semantic import INSTRUCTIONS,LETTERS

p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--allow_missing',action='store_true');a=p.parse_args()
m=json.loads((a.dataset/'manifest.json').read_text())
if m['format']!='inhand4_marked_rgb_v1' or not m['completed']:raise ValueError('Require completed actual RGB')
asset_path=Path(m['assets_manifest'])
if hashlib.sha256(asset_path.read_bytes()).hexdigest()!=m['assets_manifest_sha256']:raise ValueError('Changed asset declaration')
asset=json.loads(asset_path.read_text());letters=asset['conditions'][m['layout_condition']]['letters_by_physical_skill']
surface=asset['surface_samples'];surface_path=Path(surface['path'])
if hashlib.sha256(surface_path.read_bytes()).hexdigest()!=surface['sha256']:raise ValueError('Changed surface samples')
geo=np.load(surface_path)
regions=[]
for i in range(6):
    uv=geo[f'uv_{i}'];mask=(np.abs(uv-.5)<=.30).all(-1)
    # Deterministic subsampling over the real curved letter-bearing surface.
    region=geo[f'points_{i}'][mask][::3]
    if len(region)<30:raise ValueError('Insufficient glyph surface points')
    regions.append(region)
a.output.mkdir(parents=True,exist_ok=False)
rows=[];counts=[[0,0] for _ in range(6)];samples={}
with h5py.File(a.dataset/'samples.h5','r') as data:
    for frame in range(len(data['rgb'])):
        pose=data['object_pose'][frame];r=Rotation.from_quat(pose[3:]).as_matrix();pos=pose[:3]
        view=data['view'][frame];proj=data['projection'][frame];depth=data['depth'][frame];height,width=depth.shape
        toward=data['camera_position'][frame]-pos;toward/=np.linalg.norm(toward)
        for goal,normal in enumerate(NORMALS):
            body=regions[goal];n=len(body);world=body@r.T+pos
            camera=np.column_stack((world,np.ones(n)))@view.T;clip=camera@proj.T
            ndc=clip[:,:3]/clip[:,3:];pixels=np.column_stack(((ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2))
            xy=np.rint(pixels).astype(int)
            inside=(camera[:,2]<0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)&(np.abs(ndc[:,2])<=1)
            visible=np.zeros(n,dtype=bool)
            measured=proj[2,3]/(2*depth[xy[inside,1],xy[inside,0]]-1+proj[2,2])
            visible[inside]=measured>=-camera[inside,2]-.0015
            fraction=float(visible.mean());facing=float((r@normal)@toward)
            extent=np.ptp(pixels,axis=0);area=float(np.prod(extent))
            positive=facing>=.55 and fraction>=.9 and area>=100
            negative=facing<=-.1 or (facing>=.55 and fraction<=.05)
            if not (positive or negative):continue
            letter=letters[goal]
            row=dict(frame=frame,goal_index=goal,letter=letter,readable=positive,split=m['split'],seed=m['seed'],
                env_id=int(data['env_id'][frame]),episode_id=int(data['episode_id'][frame]),sim_step=int(data['sim_step'][frame]),
                face_cosine=facing,visible_fraction=fraction,area_pixels=area,instruction=INSTRUCTIONS[LETTERS.index(letter)])
            rows.append(row);counts[goal][int(positive)]+=1
            key=(goal,bool(positive))
            if key not in samples:samples[key]=(row,data['rgb'][frame].copy(),pixels.copy())
with (a.output/'frames.jsonl').open('w') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
meta=dict(format='inhand4_visibility_grounding_v1',dataset=str(a.dataset.resolve()),split=m['split'],dataset_manifest=m,
    counts=counts,labels=len(rows),label_source='actual curved glyph surfaces projected against rendered depth',
    scope='weak visibility labels requiring visual audit, not task success',human_audit_status='pending')
(a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
sheet=Image.new('RGB',(6*256,2*284),'white')
for (goal,visible),(row,rgb,pixels) in samples.items():
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    for xy in pixels[::8]:
        if np.isfinite(xy).all():d.ellipse((xy[0]-1,xy[1]-1,xy[0]+1,xy[1]+1),fill=(255,0,0))
    x=goal*256;y=int(visible)*284;sheet.paste(im.resize((256,256)),(x,y))
    ImageDraw.Draw(sheet).text((x+4,y+260),f"skill {goal+1} letter {row['letter']} {'visible' if visible else 'hidden'}",fill='black')
sheet.save(a.output/'audit.jpg',quality=90)
print(json.dumps(dict(counts=counts,labels=len(rows),output=str(a.output))))
if not a.allow_missing and any(min(c)==0 for c in counts):raise ValueError('Missing visibility strata')
