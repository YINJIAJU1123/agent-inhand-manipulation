"""Offline sensor-only semantic predictions at recorded real Astra request frames.

Video compression may perturb RGB. Layout is used after predictions for scoring
only. This diagnostic neither calls Astra nor changes an outcome or a model.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/semantic_completion_v1'),str(ROOT/'outputs/video_dependencies')]
from common import save,sha,color_patches,SYMBOLS
import imageio_ffmpeg
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('shadow_symbols',Path(__file__).with_name('augmented_model.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(1);pack=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=module.SymbolEvidence();model.load_state_dict(pack['model']);model.eval();threshold=pack['threshold'];rows=[];started=time.time();episodes=0
for report in sorted(Path(a.source).glob('eval_*/report.json')):
 data=json.loads(report.read_text());layout=json.loads((report.parent/'initial.json').read_text())['layout']
 for episode in data['episodes']:
  case=episode['case'];video=report.parent/f'case{case}.mp4'
  if not video.exists():continue
  requests={e['request_step']//2:e for e in episode['async_events']if e['request_step']%2==0 and e['request_step']<episode['finish_step']}
  tools={t['step']:t for t in episode['trace']};reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);w,h=meta['size'];assert abs(meta['fps']-15)<.01
  for index,frame in enumerate(reader):
   if index not in requests:continue
   event=requests[index];step=event['request_step'];rgb=torch.from_numpy(np.frombuffer(frame,dtype=np.uint8).reshape(h,w,3).copy())[None]
   with torch.inference_mode():
    patches,geo=color_patches(rgb);probability=model(patches[0],geo[0],torch.arange(6)).softmax(-1);confidence,pred=probability[:,:6].max(-1);accepted=confidence>=threshold
   # Only scoring below this point accesses labels or the actual tool response.
   instruction_symbol=SYMBOLS.index(episode['target_symbol']);candidates=[c for c in range(6)if accepted[c] and pred[c]==instruction_symbol]
   prediction=dict(colors=[dict(color=c,symbol=int(pred[c]),confidence=float(confidence[c]),accepted=bool(accepted[c]),correct=int(pred[c])==layout[case][c])for c in range(6)],target_candidates=candidates,unique_correct_target=len(candidates)==1 and candidates[0]==episode['target_color'])
   tool=tools.get(step,{});args=tool.get('arguments',{});chosen=args.get('color');colors=('red','green','blue','yellow','magenta','cyan');chosen_id=colors.index(chosen)if chosen in colors else None;claimed=args.get('goal_symbol');semantic_tool=tool.get('tool')in ('reorient_target','verify_target','stop')
   contradiction=semantic_tool and chosen_id is not None and bool(accepted[chosen_id]) and SYMBOLS[int(pred[chosen_id])]!=claimed
   astra_correct=semantic_tool and chosen_id==episode['target_color'] and claimed==episode['target_symbol']
   rows.append(dict(source=str(report.parent),case=case,step=step,**prediction,astra_tool=tool.get('tool'),astra_semantic_tool=semantic_tool,astra_binding_correct=astra_correct,contradicts_astra=bool(contradiction),target_color=episode['target_color'],target_symbol=episode['target_symbol']))
  reader.close();episodes+=1;save(out/'progress.json',dict(status='shadowing',episodes=episodes,request_frames=len(rows),elapsed_s=time.time()-started))
accepted=[c for r in rows for c in r['colors']if c['accepted']];target=[r for r in rows if len(r['target_candidates'])==1];semantic=[r for r in rows if r['astra_semantic_tool']]
result=dict(status='completed',videos=episodes,request_frames=len(rows),threshold=threshold,accepted_bindings=len(accepted),correct_bindings=sum(c['correct']for c in accepted),precision=sum(c['correct']for c in accepted)/len(accepted)if accepted else None,unique_target_frames=len(target),correct_unique_target_frames=sum(r['unique_correct_target']for r in target),astra_semantic_frames=len(semantic),astra_incorrect_binding_frames=sum(not r['astra_binding_correct']for r in semantic),contradicted_wrong_binding_frames=sum(r['contradicts_astra'] and not r['astra_binding_correct']for r in semantic),contradicted_correct_binding_frames=sum(r['contradicts_astra'] and r['astra_binding_correct']for r in semantic),checkpoint_sha256=sha(a.checkpoint),source_sha256=sha(__file__),cloud_calls=0,rows=rows,note='Diagnostic on already inspected development recordings. Predictions precede label access; no change to Astra task outcomes. Compressed video frames, not raw request PNGs. No pose or stop gate conclusion.')
save(out/'result.json',result);save(out/'progress.json',dict(status='completed',videos=episodes,request_frames=len(rows),elapsed_s=time.time()-started));print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))
