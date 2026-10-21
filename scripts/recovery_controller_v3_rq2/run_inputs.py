"""Finite two-seed direct-input dependency diagnostic, with all arms retained."""
import argparse,fcntl,hashlib,json,os,random,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/latency_counterfactual'))
from run_original import gpu_available
SEEDS=[83001,83002]
METHODS=['slow','full','frozen_visual','shifted_goal','frozen_visual_shifted_goal']
N=12

def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def counts(rows):
 valid=[r for r in rows if r['eligible']]
 return dict(nominal=len(rows),eligible=len(valid),successes=sum(r['success'] for r in valid),
  success_rate=sum(r['success'] for r in valid)/len(valid) if valid else None,
  wait_drops=sum(r['wait_drop'] for r in valid),resume_drops=sum(r['resume_drop'] for r in valid))
def analyze(out):
 arms=[]
 for done in (out/'eval').glob('*/complete.json'):
  folder=done.parent
  if not (folder/'method.json').exists():continue
  meta=json.loads((folder/'method.json').read_text());rows=json.loads((folder/'eval_trials.json').read_text())
  audit=json.loads((folder/'prefix_replay_audit.json').read_text())
  if len(rows)!=N or audit['max_abs_error']!=0 or not audit['eligible_equal']:raise RuntimeError('Invalid completed arm')
  arms.append((meta,rows))
 groups={(m['seed'],m['delay'],m['repeat']) for m,_ in arms if m['delay']>0}
 balanced={g for g in groups if {m['method'] for m,_ in arms if (m['seed'],m['delay'],m['repeat'])==g}==set(METHODS)}
 pairs=[(m,r) for m,rows in arms for r in rows if (m['seed'],m['delay'],m['repeat']) in balanced]
 report=dict(completed_arms=len(arms),completed_trials=sum(len(rows) for _,rows in arms),balanced_groups=[list(g) for g in sorted(balanced)],
  methods={method:counts([r for m,r in pairs if m['method']==method]) for method in METHODS},
  by_group={str(g):{method:counts([r for m,r in pairs if (m['seed'],m['delay'],m['repeat'])==g and m['method']==method]) for method in METHODS} for g in sorted(balanced)},
  zero_controls=[dict(**m,**counts(rows)) for m,rows in arms if m['delay']==0],
  limitations=['Direct V3 inputs only; live correct RGB-D/language still reach student and can reach V3 indirectly through proposals and proprioception.',
   'Test-time freezing/mismatching can produce distribution shift; not a retrained no-vision/no-language ablation.',
   'Two newly registered evaluation seeds, at most 24 distinct request states; repeated trials are not independent states.',
   'Fixed-state probes measure output sensitivity, not physical counterfactual success; renderer variability remains.',
   'Known canonical goal and simulated delay; no live VLM or new semantic grounding evidence.'])
 save(out/'summary.json',report);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for sub in ['logs','tmp','eval']:(out/sub).mkdir()
 lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not gpu_available(0)['available']:raise RuntimeError('GPU occupied')
 ck=ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt'
 assert sha(ck)=='cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042'
 paths=[f for base in ['scripts/agentic','scripts/recovery_controller_v3_rq2','source','configs','assets'] for f in (ROOT/base).rglob('*') if f.is_file() and f.suffix in ['.py','.json','.yaml','.yml','.usd','.usda','.usdc']]
 paths += [ck,ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt',ROOT/'tests/test_recovery_input_interventions.py']
 frozen={str(f.relative_to(ROOT)):sha(f) for f in paths}
 rng=random.Random(181032);groups=[(s,d) for s in SEEDS for d in [6.,20.]];rng.shuffle(groups)
 schedule=[]
 for seed,delay in groups:
  order=METHODS.copy();rng.shuffle(order)
  for rep,methods in [(1,order),(2,list(reversed(order)))]:schedule.append(dict(seed=seed,delay=delay,repeat=rep,methods=methods))
 protocol=dict(version='v3_rq2_direct_input_diagnostic',seeds=SEEDS,delays=[6.,20.],repeats=2,methods=METHODS,schedule=schedule,
  planned_eval_trials=528,nominal_distinct_request_states=24,planned_videos=16,trained_checkpoint=str(ck),source_asset_weights_sha256=frozen,
  frozen_student_inputs='Correct live RGB-D, proprioception and canonical embedding for all conditions',
  visual_intervention='Hold only V3 four-frame visual history at request',goal_intervention='V3 explicit embedding shifted to next goal; student and scoring unchanged',
  interpretation='Direct-input sensitivity and closed-loop effect; does not establish that the complete system needs each modality or that semantics have been grounded by a VLM',
  success='orientation <= .16 rad for six consecutive post-response control steps with no prior drop, within 300-step recovery',created=time.time())
 save(out/'protocol.json',protocol)
 with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
  for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
 python='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
 env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',PAIRED_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
 env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
 state=dict(pid=os.getpid(),status='starting',completed_stages=0,planned_stages=46,planned_eval_trials=528);child=None;cache_hashes={}
 def progress(**kwargs):state.update(kwargs,updated=time.time());save(out/'progress.json',state)
 def interrupt(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 def run(name,folder,args,timeout=1200):
  nonlocal child
  if folder.exists():raise RuntimeError('Existing output requires review')
  if any(sha(ROOT/name)!=digest for name,digest in frozen.items()):raise RuntimeError('Frozen source or model changed')
  if any(sha(Path(name))!=digest for name,digest in cache_hashes.items()):raise RuntimeError('Frozen request cache changed')
  cmd=[python,str(ROOT/'scripts/agentic/evaluate_recovery_inputs.py'),'--headless','--device','cuda:0','--output',str(folder),'--num-envs',str(N),'--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']+args
  save(out/'logs'/f'{name}_command.json',cmd)
  with (out/'logs'/f'{name}.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.monotonic();progress(status=name,current_output=str(folder),child_pid=child.pid)
   while child.poll() is None:
    progress(stage_wall_seconds=round(time.monotonic()-start,1))
    if time.monotonic()-start>timeout:raise TimeoutError(name)
    time.sleep(5)
   if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}')
  if not (folder/'complete.json').exists():raise RuntimeError('Missing completion')
  progress(completed_stages=state['completed_stages']+1,child_pid=None)
 def evaluate(seed,delay,rep,method):
  name=f'seed_{seed}_delay_{delay:g}_r{rep}_{method}';folder=out/'eval'/name
  mode=method if method in ['continue','slow'] else 'learned'
  args=['--mode',mode,'--seed',str(seed),'--delay',str(delay),'--request-cache',str(out/f'cache_{seed}/request_state.pt')]
  if mode=='learned':args+=['--checkpoint',str(ck),'--controller-input',method]
  if delay>0 and rep==1 and method in ['full','slow']:args+=['--record-slots','0','5']
  run(name,folder,args)
  save(folder/'method.json',dict(method=method,seed=seed,delay=delay,repeat=rep))
  report=analyze(out);progress(completed_evaluation_trials=report['completed_trials'])
 try:
  for seed in SEEDS:
   folder=out/f'cache_{seed}';run(folder.name,folder,['--mode','cache','--seed',str(seed)])
   cache_hashes[str(folder/'request_state.pt')]=sha(folder/'request_state.pt');save(out/'cache_hashes.json',cache_hashes)
  for seed in SEEDS:
   for rep in [1,2]:evaluate(seed,0.,rep,'continue')
   if sum(v['successes'] for v in analyze(out)['zero_controls'] if v['seed']==seed)==0:
    raise RuntimeError('All zero-delay controls failed for registered seed; preserve it and stop')
  for g in schedule:
   for method in g['methods']:evaluate(g['seed'],g['delay'],g['repeat'],method)
  assert analyze(out)['completed_trials']==528
  progress(status='completed_diagnostic',child_pid=None,current_output=None)
 except BaseException as exc:
  progress(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  lock.close()

if __name__=='__main__':main()
