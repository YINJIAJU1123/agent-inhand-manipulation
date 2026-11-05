"""Four marked objects, matched identity/permuted Q1. Real model latency elapses under active physics.

Formal runs require frozen cases and code/artifact hashes. Service errors remain
in the raw ledger and are excluded by matched case across methods at aggregation.
"""
import pyglet
pyglet.options['headless']=True
import argparse,base64,io,json,math,os,signal,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
import torch
from PIL import Image
from runtime import DiceRuntime,add_arguments,sha
from semantic import SYSTEM,validate_action
import semantic as semantic_contract
if Path(semantic_contract.__file__).resolve()!=Path(__file__).with_name("semantic.py").resolve():
 raise ValueError("Study must load its own semantic contract before any model call")
from semantic import INSTRUCTIONS as PROMPTS,LETTERS
from control_modules import CompletionGate,features,load_maintenance_actor
from viserdex.utils.runtime import add_runtime_args,parse_runtime_args
from viserdex.supermimic.simulation import TASK_ID,tensor
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);add_arguments(p)
p.add_argument('--provider',choices=('astra','vlm','rule'),required=True)
p.add_argument('--actor_checkpoint',type=Path);p.add_argument('--stop_checkpoint',type=Path)
p.add_argument('--maintenance',choices=('continue','fixed_slow','learned'),default='continue')
p.add_argument('--service_url');p.add_argument('--release',type=Path);p.add_argument('--development',action='store_true')
p.add_argument('--language_manifest',type=Path)
p.add_argument('--episodes',type=int,default=40);p.add_argument('--start_episode',type=int,default=0)
add_runtime_args(p,TASK_ID);a=parse_runtime_args(p)
if not a.assets_manifest:raise ValueError('Marked object asset manifest required')
if a.num_envs!=1:raise ValueError('Formal agent runtime requires one world')
if a.output.exists():raise FileExistsError(a.output)
if a.language_manifest:
 if a.release or a.development:raise ValueError('Language manifests are a separate frozen evaluation mode')
else:
 if a.development==bool(a.release):raise ValueError('Choose explicit development or a frozen release')
if a.provider!='astra' and (a.maintenance!='continue' or a.actor_checkpoint or a.stop_checkpoint):raise ValueError('Bare VLM baselines cannot receive extra modules')
if a.maintenance=='learned' and not a.actor_checkpoint:raise ValueError('Learned maintenance checkpoint required')
language_manifest=json.loads(a.language_manifest.read_text()) if a.language_manifest else None
if language_manifest:
 if (language_manifest.get('format')!='multilingual_case_manifest_v1'
     or language_manifest.get('object') not in ('duck','hedgehog')
     or a.provider not in language_manifest.get('method_conditions',[])
     or a.layout_condition!='identity'
     or len(language_manifest.get('cases',[]))!=60):raise ValueError('Invalid marked-object language case manifest')
 cases=language_manifest['cases']
 if any(c.get('index')!=i or c.get('prompt_condition')!=language_manifest['prompt_condition']
        or c.get('language') not in ('en','zh') for i,c in enumerate(cases)):
  raise ValueError('Malformed duck language cases')
 for case in cases:
  case['requested_label_index']=case['goal_index']+1
else:
 cases=None
release=json.loads(a.release.read_text()) if a.release else None
if release:
 if release['format']!='inhand4_marked_q1_v1' or len(release['cases'])!=40:raise ValueError('Formal release format')
 for path,digest in release['source_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Frozen source changed: '+path)
 for flag in ['source_checkpoint','checkpoint','env_config','bank','assets_manifest','relocation']:
  if sha(getattr(a,flag))!=release['shared_artifact_sha256'][flag]:raise ValueError('Frozen artifact changed: '+flag)
 condition=release['conditions'][a.provider]
 if condition['maintenance']!=a.maintenance:raise ValueError('Wrong method configuration')
 for flag in ['actor_checkpoint','stop_checkpoint']:
  value=getattr(a,flag)
  if (sha(value) if value else None)!=condition.get(flag+'_sha256'):raise ValueError('Wrong method weight: '+flag)
 cases=release['cases_by_layout'][a.layout_condition]
else:
 if language_manifest is None:
  if a.seed in [187501,187502]:raise ValueError('Reserved formal seed')
  letters=json.loads(a.assets_manifest.read_text())['conditions'][a.layout_condition]['letters_by_physical_skill']
  layout=[LETTERS.index(letter)+1 for letter in letters]
  # This table constructs language/scoring labels only. It never transforms an action.
  cases=[dict(index=i,base_seed=a.seed,episode=i,trial_seed=a.seed*1000+i,goal_index=i%6,requested_label_index=layout[i%6],instruction=PROMPTS[layout[i%6]-1],language='en') for i in range(a.start_episode+a.episodes)]
limit=60 if language_manifest else 40
if not 0<a.episodes<=limit or a.start_episode+a.episodes>len(cases):raise ValueError('Case subset invalid')
if release and any(case['base_seed']!=a.seed for case in cases[a.start_episode:a.start_episode+a.episodes]):raise ValueError('Each formal process must use its registered base seed')
a.output.mkdir(parents=True)

def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def angle(q,target):
 q=torch.nn.functional.normalize(q.double(),dim=-1);target=torch.nn.functional.normalize(target.double(),dim=-1)
 return float((2*torch.acos((q*target).sum(-1).abs().clamp(0,1)))[0])

class RemotePredictor:
 def __init__(self,url):
  if not url or not url.startswith('http://127.0.0.1:'):raise ValueError('Local VLM service required')
  self.url=url;response=requests.get(url+'/health',timeout=10);response.raise_for_status();self.metadata=response.json()
  if self.metadata.get('kind')!=a.provider or self.metadata.get('frame_size')!=512 or self.metadata.get('backend')!='inhand4_marked_q1_v1':raise ValueError('Wrong VLM service')
  if self.metadata.get('semantic_sha256')!=sha(Path(__file__).with_name('semantic.py')):raise ValueError('Wrong VLM semantic contract')
 def predict(self,rgb,instruction,language='en'):
  buf=io.BytesIO();Image.fromarray(rgb).save(buf,format='PNG')
  response=requests.post(self.url+'/predict',json=dict(image=base64.b64encode(buf.getvalue()).decode(),instruction=instruction,language=language),timeout=(10,180),allow_redirects=False)
  response.raise_for_status();return response.json()

def request(client,instruction,rgb,feedback,history,language='en'):
 start=time.monotonic()
 try:
  result=client.ask(instruction,rgb,feedback,*history) if a.provider=='astra' else client.predict(rgb,instruction,language)
  return dict(result=result,error=None,elapsed_s=time.monotonic()-start)
 except Exception as e:return dict(result=None,error=type(e).__name__+': '+str(e),elapsed_s=time.monotonic()-start)

r=None;rows=[];status='failed';start=time.monotonic();stopping=[False]
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
try:
 if a.provider=='astra':
  sys.path[:0]=[str(ROOT/'scripts/astra_transport_v2'),str(ROOT/'scripts/agentic')]
  from bridge import BridgeClient,bridge_health
  bridge_health();upstream=dict(backend='live_gpt_6_astra',protocol_sha256=sha(Path(__file__).with_name('semantic.py')))
 else:predictor=RemotePredictor(a.service_url);upstream=predictor.metadata
 if release and a.provider=='vlm':
  if upstream['checkpoint_sha256']!=release['conditions']['vlm']['goal_head_sha256'] or upstream['adapter_sha256']!=release['conditions']['vlm']['adapter_sha256']:raise ValueError('Wrong frozen VLM weights')
 r=DiceRuntime(a)
 actor=load_maintenance_actor(a.actor_checkpoint,sha(a.checkpoint),r.vectors,r.device) if a.maintenance=='learned' else None
 gate=CompletionGate(a.stop_checkpoint,sha(a.checkpoint),r.vectors,r.device) if a.stop_checkpoint else None
 manifest=dict(runtime=r.metadata,provider=a.provider,upstream=upstream,maintenance=a.maintenance,
  semantic_sha256=sha(Path(__file__).with_name("semantic.py")),
  study_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob("*.py")},
  stop_mode='learned' if gate else 'measured',stop_hz=3,call_limit=6,development=a.development,
  scope='frozen bilingual language-coverage study' if language_manifest else 'frozen Q1 comparison' if release else 'development only',
  prompt_condition=language_manifest['prompt_condition'] if language_manifest else None,
  language_manifest_sha256=sha(a.language_manifest) if language_manifest else None,
  release_sha256=sha(a.release) if a.release else None,success='correct semantic binding AND true full canonical orientation error <= .16 rad at autonomous stop',
  service_exclusion='transport error or request latency >120s; matched across methods; never drop or wrong instruction',
  actor_sha256=sha(a.actor_checkpoint) if a.actor_checkpoint else None,stop_sha256=sha(a.stop_checkpoint) if a.stop_checkpoint else None)
 save(a.output/'manifest.json',manifest)
 if language_manifest:save(a.output/'case_manifest.json',language_manifest)
 with ThreadPoolExecutor(max_workers=1) as pool,torch.no_grad(),(a.output/'steps.jsonl').open('w',buffering=1) as log:
  for case in cases[a.start_episode:a.start_episode+a.episodes]:
   if stopping[0]:break
   goal=case['goal_index'];requested_label_index=case['requested_label_index'];instruction=case['instruction'];language=case.get('language','en');target=r.rotations[goal:goal+1]
   initial=r.reset(case['trial_seed']);epdir=a.output/f"episode_{case['index']:03d}";epdir.mkdir()
   client=BridgeClient() if a.provider=='astra' else predictor
   known=False;bound=None;declared=None;mode='startup';due=0;history=(None,None)
   future=None;ready=None;calls=[];submitted=0;request_step=0;request_wall=0.;first_delivery=None
   reason=None;excluded=None;drop_phase=None;stop_measurement=None;next_stop_tick=0
   final_q=tensor(r.raw.scene['object'].data.root_quat_w).clone()
   error=angle(final_q,target);minimum=error;reached=False
   while reason is None:
    if stopping[0]:reason='interrupted';break
    sensor=r.observe()
    if future is None and ready is None and r.steps>=due and submitted<6:
     rgb=r.observe(include_rgb=True)['rgb'];submitted+=1;request_step=r.steps;request_wall=time.monotonic()
     r.anchor_pose=sensor['measured_pose'].clone();r.anchor_action=sensor['current'].clone()
     Image.fromarray(rgb).save(epdir/f'request_{submitted:02d}.png')
     feedback=dict(control_step=r.steps,elapsed_simulation_s=r.steps*.05,previous_state=mode,current_number_skill=r.active_goal+1,
      bound_number=bound+1 if known else None,calls_remaining=6-submitted,
      sensor_estimated_error_rad_by_skill={str(i+1):float(sensor['estimated_errors'][0,i]) for i in range(6)},
      caution='No true orientation or success signal. These local sensor estimates include noise, delay and failures.')
     save(epdir/f'agent_input_{submitted:02d}.json',dict(instruction=instruction,language=language,feedback=feedback,
      image_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),
      fields_sent=['instruction','image','feedback','history'] if a.provider=='astra' else ['instruction','image']))
     future=pool.submit(request,client,instruction,rgb,feedback,history,language)
    if future is not None and future.done():
     ready=future.result();future=None;ready.update(call=submitted,request_step=request_step,delivery_step=request_step+math.ceil(ready['elapsed_s']*20))
     if ready['error']:excluded='service_error'
     elif ready['elapsed_s']>120:excluded='request_latency_over_120s'
     if excluded:calls.append(ready);ready=None;reason='service_excluded';break
    if ready is not None and r.steps>=ready['delivery_step']:
     result=ready['result'];ready['applied_step']=r.steps;calls.append(ready);ready=None
     if a.provider=='astra':
      if result.get('system_prompt')!=SYSTEM:raise ValueError('Wrong Astra system prompt')
      tool=result['tool'];args=validate_action(tool,result['arguments']);history=(result['response_id'],result['call_id'])
      if tool=='abstain':reason='abstain';break
      selected=args['number']-1;r.set_skill(selected);mode=tool
      if tool=='inspect_face':known=False;bound=None;declared=None
      else:known=True;bound=selected;declared=args['goal_number']
      due=r.steps+args.get('steps',20)
     else:
      if bool(result['accepted']):
       selected=result['skill']
       if type(selected) is not int or selected not in range(6):raise ValueError('Malformed VLM skill')
       vector=result['goal_embedding'] if a.provider=='vlm' else None
       r.set_skill(selected,vector);known=True;bound=selected;declared=requested_label_index;mode='reorient_target'
      elif known:mode='verify_target'
      else:r.set_skill((1,3,0,2,4,5)[(submitted-1)%6]);mode='inspect_face'
      due=r.steps+60
     if first_delivery is None:first_delivery=r.steps
     sensor=r.observe()
    pending=future is not None or ready is not None
    values=r.values(pending,known,(r.steps-request_step)*.05 if pending else 0.)
    eligible=known and mode!='inspect_face' and declared==requested_label_index and (a.provider=='astra' or not pending)
    accepted=False;estimate=None
    # Integer accumulator gives exactly 3Hz eligibility, not a rounded 20/3 divisor.
    if r.steps*3>=next_stop_tick:
     next_stop_tick=((r.steps*3)//20+1)*20
     if eligible:
      measured=values['estimated_errors'][:,bound]
      estimate=gate.estimate(values,r.latent,measured) if gate else None
      accepted=bool(estimate['accept'][0]) if gate else float(measured[0])<=.16
      if accepted:
       final_q=tensor(r.raw.scene['object'].data.root_quat_w).clone()
       error=angle(final_q,target)
       reason='success' if bound==goal and error<=.16 else 'false_stop'
       stop_measurement=dict(step=r.steps,pending=pending,measured_error_rad=float(measured[0]),learned=gate is not None)
       break
    current=values['current'];proposal=values['proposal'];action=proposal;speed=1.;residual=0.
    if a.maintenance=='fixed_slow' and pending:action=current+.15*(proposal-current);speed=.15
    elif actor is not None:
     out=actor(features(values),r.latent,proposal,current,values['pending']);action=out['action'];speed=float(out['speed'][0]);residual=float(out['residual'].norm())
    out=r.step(action);final_q=out['quaternion'].clone();error=angle(final_q,target);minimum=min(minimum,error);reached|=error<=.16
    if bool(out['drop'][0]):reason='drop';drop_phase='startup_wait' if first_delivery is None else 'wait' if pending else 'execution'
    elif bool(out['done'][0]) or r.steps>=2400:reason='horizon'
    log.write(json.dumps(dict(case=case['index'],step=r.steps,mode=mode,pending=pending,known=known,active_goal=r.active_goal,
     true_error_rad=error,measured_active_error_rad=float(values['estimated_errors'][0,r.active_goal]),speed=speed,residual_l2=residual,reason=reason))+'\n')
    if pending and reason is None:
     delay=(r.steps-request_step)*.05-(time.monotonic()-request_wall)
     if delay>0:time.sleep(delay)
    if r.steps%200==0:save(a.output/'progress.json',dict(status='running',completed=len(rows),case=case['index'],steps=r.steps,calls=submitted,elapsed=time.monotonic()-start))
   if future is not None:
    ready=future.result();ready.update(call=submitted,request_step=request_step,delivery_step=None)
   if ready is not None:
    if ready['error']:excluded='service_error'
    elif ready['elapsed_s']>120:excluded='request_latency_over_120s'
    calls.append(ready)
   save(epdir/'calls.json',calls)
   row=dict(**case,initial_state=initial,steps=r.steps,reason=reason,success=reason=='success',exclusion_reason=excluded,
    drop_phase=drop_phase,first_skill_delivery_step=first_delivery,stop_measurement=stop_measurement,minimum_error_rad=minimum,
    final_error_rad=error,final_quaternion_xyzw=final_q[0].cpu().tolist(),target_quaternion_xyzw=target[0].cpu().tolist(),reached=reached,declared_number=declared,declared_goal=bound,actual_model_calls=submitted,latencies_s=[c['elapsed_s'] for c in calls])
   rows.append(row)
   with (a.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   save(a.output/'progress.json',dict(status='running',completed=len(rows),success=sum(x['success'] for x in rows),last_reason=reason,elapsed=time.monotonic()-start))
   print(json.dumps(dict(case=case['index'],reason=reason,steps=r.steps,excluded=excluded)),flush=True)
   if excluded:status='service_blocked';break
   if reason=='interrupted':break
 if status!='service_blocked':status='completed' if len(rows)==a.episodes and all(x['reason']!='interrupted' for x in rows) else 'interrupted'
except Exception as exc:
 save(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
finally:
 eligible=[x for x in rows if x['exclusion_reason'] is None]
 save(a.output/'result.json',dict(status=status,provider=a.provider,development=a.development,requested=a.episodes,recorded=len(rows),
  raw_success=sum(x['success'] for x in rows),eligible=len(eligible),eligible_success=sum(x['success'] for x in eligible),
  matched_exclusions_pending=True,elapsed=time.monotonic()-start))
 if r:r.close()
