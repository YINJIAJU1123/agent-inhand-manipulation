"""Offline coarse verification audit. GT annotation never enters VLM requests."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from visual_agent import VLMClient,COLORS
p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('--output',required=True); a=p.parse_args()
root=Path(a.directory); client=VLMClient(); records=[]
for path in sorted(root.glob('seed*_oracle/report.json')):
    for episode in json.loads(path.read_text())['episodes']:
        # First pre-action view with an unambiguously upward face, plus a side.
        for snap in episode['snapshots']:
            z=np.asarray(snap['scorer_face_normal_z']); top=int(z.argmax())
            if z[top]<.85: continue
            sides=np.where((z>=-.25)&(z<.4))[0]
            if not len(sides): continue
            rgb=np.asarray(Image.open(path.parent/snap['path']))
            for color in [top,int(sides[0])]:
                result=client.verify(rgb,COLORS[color])
                row={'seed':episode['seed'],'case':episode['case'],'image':str(path.parent/snap['path']),'queried_color':COLORS[color],'scorer_normal_z':float(z[color]),'scorer_upward':bool(color==top),'prediction':result['parsed'].get('upward','unknown'),'result':result}
                records.append(row)
                Path(a.output).write_text(json.dumps({'records':records},indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='result'}),flush=True)
            break
