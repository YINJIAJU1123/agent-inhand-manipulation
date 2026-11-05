"""Development-only check of fixed-goal 120s runtime, never a semantic score."""
import pyglet
pyglet.options['headless']=True
import argparse,json,time
from pathlib import Path
import torch
from runtime import DiceRuntime,add_arguments
from viserdex.utils.runtime import add_runtime_args,parse_runtime_args
from viserdex.supermimic.simulation import TASK_ID
p=argparse.ArgumentParser(description=__doc__);add_arguments(p)
p.add_argument('--steps',type=int,default=600);p.add_argument('--goals',type=int,default=6)
add_runtime_args(p,TASK_ID);a=parse_runtime_args(p)
if a.output.exists():raise FileExistsError(a.output)
if a.num_envs!=1:raise ValueError('Use single env for deployment runtime check')
a.output.mkdir(parents=True);r=None;results=[];start=time.monotonic()
try:
 r=DiceRuntime(a);(a.output/'manifest.json').write_text(json.dumps(r.metadata,indent=2)+'\n')
 for goal in range(a.goals):
  init=r.reset(a.seed+goal);r.set_skill(goal)
  sensor=r.observe(include_rgb=True)
  from PIL import Image
  Image.fromarray(sensor['rgb']).save(a.output/f'initial_{goal+1}.png')
  errors=[];events=0;dropped=False;done=False
  for i in range(a.steps):
   sensor=r.observe();out=r.step(sensor['proposal'])
   error=float(2*torch.acos((out['quaternion']*r.rotations[goal]).sum(-1).abs().clamp(0,1))[0]);errors.append(error)
   events+=int(out['record']['success'].sum());dropped=bool(out['drop'][0]);done=bool(out['done'][0])
   if done:break
  row=dict(goal=goal+1,steps=r.steps,min_true_error_rad=min(errors),final_true_error_rad=errors[-1],native_hold_events=events,
   drop=dropped,done=done,initial_state_sha256=init['sha256'],max_errors=r.adapter.max_errors)
  results.append(row);print(json.dumps(row),flush=True)
 (a.output/'result.json').write_text(json.dumps(dict(complete=True,scope='fixed-goal runtime development check, not semantic main experiment',rows=results,elapsed=time.monotonic()-start),indent=2)+'\n')
finally:
 if r:r.close()
