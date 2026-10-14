"""Resumable bounded evaluation, with an owned VLM service and per-job watchdog.

This is a single local process, not a scheduled automation. No policy tuning.
"""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--protocol',required=True);p.add_argument('--model',required=True);a=p.parse_args()
dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=True)
protocol=json.loads(Path(a.protocol).read_text())
critical=['evaluate_visual_agent.py','visual_agent.py','student_features.py','symbol_scene.py','vlm_server.py','rollout_video.py','run_large_eval.py']
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():return {name:digest(ROOT/'scripts/agentic'/name) for name in critical}
def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def health():
 try:
  with urllib.request.urlopen('http://127.0.0.1:18741',timeout=2) as response:return json.load(response)
 except Exception:return None
freeze_path=dest/'freeze.json'
if freeze_path.exists():
 freeze=json.loads(freeze_path.read_text())
 if freeze['protocol']!=protocol or freeze['source_sha256']!=sources() or freeze['vlm_model']!=str(Path(a.model).resolve()):raise RuntimeError('Resume rejected: frozen protocol, code, or model path differs')
else:
 freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':protocol,'source_sha256':sources(),'vlm_model':str(Path(a.model).resolve()),'model_config_sha256':digest(Path(a.model)/'config.json'),'student_sha256':digest(ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'),'adapter_sha256':digest(ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'),'symbol_assets_sha256':{x.name:digest(x) for x in (ROOT/'assets/usd/semantic_symbols').glob('*.usda')}}
 write(freeze_path,freeze)
server=None;child=None;server_log=None

def stop(proc):
 if proc is not None and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,interrupted)
state={'status':'starting','pid':os.getpid(),'total_jobs':len(protocol['seeds'])*len(protocol['modes']),'planned_episodes':len(protocol['seeds'])*len(protocol['modes'])*6,'completed_jobs':0}
def progress(**fields):
 state.update(fields);state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(dest/'progress.json',state)
def aggregate():
 by_mode={m:[] for m in protocol['modes']}
 for report in sorted(dest.glob('seed*/report.json')):
  for row in json.loads(report.read_text())['episodes']:by_mode[row['mode']].append(row)
 result={m:{'episodes':len(rows),'success':sum(r['success'] for r in rows),'drop':sum(r['drop'] for r in rows),'grounding_correct':sum(r['grounding_correct'] for r in rows),'calls':sum(r['calls'] for r in rows)} for m,rows in by_mode.items()}
 write(dest/'running_counts.json',result)
 return result
try:
 progress(status='loading_vlm')
 response=health()
 if response is not None:raise RuntimeError('Port 18741 already has a service; refusing to reuse an unverified process')
 env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
 dependencies=[str(ROOT/'outputs/vlm_dependencies'),str(ROOT/'outputs/video_dependencies')]
 env['PYTHONPATH']=os.pathsep.join(dependencies+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
 server_log=(dest/'vlm_server.log').open('a')
 server=subprocess.Popen([sys.executable,str(ROOT/'scripts/agentic/vlm_server.py'),'--model',a.model,'--quantize'],cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
 limit=time.monotonic()+120
 while health() is None:
  if server.poll() is not None:raise RuntimeError('VLM exited during load')
  if time.monotonic()>limit:raise TimeoutError('VLM readiness deadline exceeded')
  time.sleep(1)
 completed=0;durations=[]
 for seed in protocol['seeds']:
  for mode in protocol['modes']:
   if sources()!=freeze['source_sha256']:raise RuntimeError('Source changed during frozen evaluation')
   name=f'seed{seed}_{mode}';out=dest/name;report_path=out/'report.json'
   if report_path.exists():
    report=json.loads(report_path.read_text());meta=report['metadata']
    if len(report['episodes'])!=6 or meta['seeds']!=[seed] or meta['modes']!=[mode] or meta['call_budget']!=protocol['call_budget'] or meta['max_steps']!=protocol['max_steps']:raise RuntimeError(f'Invalid resume report: {name}')
    if any(meta['source_sha256'][key]!=freeze['source_sha256'][key] for key in meta['source_sha256']):raise RuntimeError('Completed report source does not match freeze')
    completed+=1;continue
   command=[sys.executable,str(ROOT/'scripts/agentic/evaluate_visual_agent.py'),'--headless','--output',str(out),'--seeds',str(seed),'--modes',mode,'--max-steps',str(protocol['max_steps']),'--budget',str(protocol['call_budget']),'--grounding','inventory']
   if seed in protocol['record_video_seeds']:command.append('--record-video')
   progress(status='running',current_job=name,completed_jobs=completed)
   print(json.dumps({'starting':name,'completed_jobs':completed}),flush=True)
   start=time.monotonic()
   with (dest/(name+'.log')).open('w') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
     if time.monotonic()-start>protocol['job_timeout_seconds']:raise TimeoutError(f'{name} exceeded watchdog deadline')
     progress(current_job_elapsed_s=round(time.monotonic()-start,1));time.sleep(5)
    if child.returncode:raise RuntimeError(f'{name} failed with exit {child.returncode}')
   report=json.loads(report_path.read_text())
   if len(report['episodes'])!=6:raise RuntimeError(f'{name} has incomplete report')
   durations.append(time.monotonic()-start);completed+=1
   counts=aggregate();progress(completed_jobs=completed,current_job=None,estimated_remaining_seconds=round(sum(durations)/len(durations)*(state['total_jobs']-completed)))
   print(json.dumps({'finished':name,'completed_jobs':completed,'counts':counts}),flush=True)
 progress(status='completed',completed_jobs=completed,current_job=None,counts=aggregate(),estimated_remaining_seconds=0)
except BaseException as error:
 progress(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
finally:
 stop(child);stop(server)
 if server_log:server_log.close()
