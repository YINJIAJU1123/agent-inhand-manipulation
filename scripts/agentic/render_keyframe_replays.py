"""Turn recorded snapshots into honest held-frame replay videos, not new rollouts."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

COLORS=('red','green','blue','yellow','magenta','cyan')
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
choices=[(1013,'periodic',0,'success_active_search'),(912,'event_memory',2,'success_event_confirmation'),(1013,'periodic',1,'failure_drop'),(1013,'event_memory',3,'failure_unresolved')]
index=[]
for seed,mode,case,name in choices:
 folder=root/f'seed{seed}_{mode}';report=folder/'report.json';r=json.loads(report.read_text())['episodes'][case]
 snaps=sorted(r['snapshots'],key=lambda x:x['step']);frames=[Image.open(folder/s['path']).convert('RGB') for s in snaps]
 result='SUCCESS' if r['success'] else ('FAILED: DROP' if r['drop'] else 'FAILED: UNRESOLVED / TIME LIMIT')
 def draw(j,final=False):
  snap=snaps[j];canvas=Image.new('RGB',(768,832),(19,24,32));d=ImageDraw.Draw(canvas)
  d.text((20,10),'RECORDED KEYFRAMES (~1 Hz) - NOT continuous video',font=font,fill=(255,206,90))
  d.text((20,42),f'{mode} | seed {seed} | case {case} | task: star upward',font=font,fill='white')
  d.text((20,70),f'Correct sticker: {COLORS[case]} (scorer annotation only)',font=small,fill=(180,190,210))
  canvas.paste(frames[j].resize((640,640)),(64,100))
  decision=snap['decision'];color=decision['color'];label=COLORS[color] if color is not None else 'unknown'
  d.text((20,745),f'Saved step {snap["step"]} | state: {decision["state"]} | skill: {label}',font=font,fill='white')
  if final:
   d.text((20,774),f'Report outcome: {result}; {r["calls"]} VLM calls',font=font,fill=(100,230,160) if r['success'] else (255,150,140))
   d.text((20,804),'Last saved frame held here; terminal image was not recorded.',font=small,fill=(190,190,190))
  else:
   calls=[t for t in r['trace'] if t['step']<=snap['step']]
   latest=calls[-1]['color'] if calls else 'none'
   d.text((20,777),f'Latest VLM observation: {latest} | calls so far: {len(calls)}',font=font,fill=(180,200,230))
  return np.asarray(canvas)
 output=dest/(name+'.mp4');writer=imageio_ffmpeg.write_frames(str(output),(768,832),fps=10,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','22','-movflags','+faststart'],ffmpeg_log_level='error');writer.send(None)
 j=0;duration=max(r['steps']/30,1)
 try:
  for tick in range(math.ceil(duration*10)):
   step=tick*3
   while j+1<len(snaps) and snaps[j+1]['step']<=step:j+=1
   writer.send(draw(j))
  final=draw(len(snaps)-1,True)
  for _ in range(30):writer.send(final)
 finally:writer.close()
 Image.fromarray(draw(len(snaps)-1,True)).save(dest/(name+'_preview.png'))
 index.append({'video':str(output.resolve()),'source_report':str(report.resolve()),'case':case,'success':r['success'],'drop':r['drop'],'keyframe_count':len(snaps),'type':'held-keyframe replay; no synthesized or interpolated motion'})
 print(json.dumps(index[-1]),flush=True)
(dest/'index.json').write_text(json.dumps(index,indent=2)+'\n')
