"""Depth-tested visibility supervision for dice faces; no labels enter VLM inputs."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from goals import NORMALS,PROMPTS
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--allow_missing',action='store_true');a=p.parse_args()
m=json.loads((a.dataset/'manifest.json').read_text())
if m['format']!='dice_newton_rgb_v1' or not m['completed']:raise ValueError('Require completed actual RGB collection')
a.output.mkdir(parents=True,exist_ok=False)
rows=[];counts=[[0,0] for _ in range(6)];samples={}
with h5py.File(a.dataset/'samples.h5','r') as data:
 for frame in range(len(data['rgb'])):
  pose=data['object_pose'][frame];r=Rotation.from_quat(pose[3:]).as_matrix();pos=pose[:3]
  v=data['view'][frame];p=data['projection'][frame];depth=data['depth'][frame];rgb=data['rgb'][frame];height,width=depth.shape
  toward=data['camera_position'][frame]-pos;toward/=np.linalg.norm(toward)
  for goal,normal in enumerate(NORMALS):
   normal=np.asarray(normal,dtype=float);axis=np.argmax(np.abs(normal));tangents=[j for j in range(3) if j!=axis]
   grid=np.stack(np.meshgrid(np.linspace(-.018,.018,9),np.linspace(-.018,.018,9)),-1).reshape(-1,2)
   body=np.zeros((81,3));body[:,axis]=normal[axis]*.0349;body[:,tangents]=grid
   world=body@r.T+pos;camera=np.column_stack((world,np.ones(81)))@v.T;clip=camera@p.T
   ndc=clip[:,:3]/clip[:,3:];pixels=np.column_stack(((ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2))
   xy=np.rint(pixels).astype(int)
   inside=(camera[:,2]<0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)&(np.abs(ndc[:,2])<=1)
   visible=np.zeros(81,dtype=bool)
   measured=p[2,3]/(2*depth[xy[inside,1],xy[inside,0]]-1+p[2,2])
   visible[inside]=measured>=-camera[inside,2]-.003
   fraction=float(visible.mean());facing=float((r@normal)@toward)
   corners=pixels[[0,8,80,72]];area=float(abs(np.dot(corners[:,0],np.roll(corners[:,1],1))-np.dot(corners[:,1],np.roll(corners[:,0],1)))/2)
   positive=facing>=.55 and fraction>=.9 and area>=100
   negative=facing<=-.1 or (facing>=.55 and fraction<=.05)
   if not(positive or negative):continue
   row=dict(frame=frame,goal_index=goal,number=goal+1,readable=positive,split=m['split'],seed=m['seed'],env_id=int(data['env_id'][frame]),episode_id=int(data['episode_id'][frame]),sim_step=int(data['sim_step'][frame]),face_cosine=facing,visible_fraction=fraction,area_pixels=area,instruction=PROMPTS[goal])
   rows.append(row);counts[goal][int(positive)]+=1
   key=(goal,bool(positive))
   if key not in samples:samples[key]=(row,rgb.copy(),corners.copy())
with (a.output/'frames.jsonl').open('w') as f:
 for row in rows:f.write(json.dumps(row)+'\n')
meta=dict(format='dice_visibility_grounding_v1',dataset=str(a.dataset.resolve()),split=m['split'],dataset_manifest=m,counts=counts,labels=len(rows),label_source='projected central face region tested against actual rendered scene depth',scope='weak visibility labels, require visual audit; not semantic success',human_audit_status='pending')
(a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
sheet=Image.new('RGB',(6*256,2*284),'white')
for (goal,visible),(row,rgb,corners) in samples.items():
 im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.line([tuple(x) for x in np.vstack((corners,corners[:1]))],fill=(255,0,0),width=2)
 x=goal*256;y=int(visible)*284;sheet.paste(im.resize((256,256)),(x,y));ImageDraw.Draw(sheet).text((x+4,y+260),f"face {goal+1} {'visible' if visible else 'hidden'}",fill='black')
sheet.save(a.output/'audit.jpg',quality=90)
print(json.dumps({'counts':counts,'labels':len(rows),'output':str(a.output)}))
if not a.allow_missing and any(min(x)==0 for x in counts):raise ValueError('Missing visibility strata')
