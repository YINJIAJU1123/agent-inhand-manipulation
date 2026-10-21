"""Train V3 once and compare against strong controls with paired repetitions."""
import argparse,fcntl,hashlib,json,os,random,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/latency_counterfactual'))
from run_original import gpu_available
SEEDS=[82001,82002];METHODS=['slow','hold_smooth','v2_learned','v3_learned'];N=12

def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def counts(rows):
 valid=[r for r in rows if r['eligible']]
 return dict(nominal=len(rows),eligible=len(valid),successes=sum(r['success'] for r in valid),
  success_rate=sum(r['success'] for r in valid)/len(valid) if valid else None,
  wait_drops=sum(r['wait_drop'] for r in valid),resume_drops=sum(r['resume_drop'] for r in valid))
def analyze(out):
 arms=[]
 for done in (out/'eval').glob('*/complete.json'):
  folder=done.parent;meta=json.loads((folder/'method.json').read_text());rows=json.loads((folder/'eval_trials.json').read_text())
  audit=json.loads((folder/'prefix_replay_audit.json').read_text())
  if len(rows)!=N or audit['max_abs_error']!=0 or not audit['eligible_equal']:raise RuntimeError('Bad completed arm')
  arms.append((meta,rows))
 groups={(m['seed'],m['delay'],m['repeat']) for m,_ in arms if m['delay']>0}
 balanced={g for g in groups if {m['method'] for m,_ in arms if (m['seed'],m['delay'],m['repeat'])==g}==set(METHODS)}
 positive=[(m,r) for m,rows in arms for r in rows if (m['seed'],m['delay'],m['repeat']) in balanced]
 zero=[dict(**m,**counts(rows)) for m,rows in arms if m['delay']==0]
 summary=dict(completed_arms=len(arms),completed_trials=sum(len(rows) for _,rows in arms),
  balanced_groups=[list(g) for g in sorted(balanced)],
  positive_delay={method:counts([r for m,r in positive if m['method']==method]) for method in METHODS},
  by_seed_delay_repeat={str(g):{method:counts([r for m,r in positive if (m['seed'],m['delay'],m['repeat'])==g and m['method']==method]) for method in METHODS} for g in sorted(balanced)},
  zero_controls=zero,
  limitations=['Two previously examined development seeds; 23 eligible cached states, not 432 independent initial states.',
   'Rendering remains stochastic; retain null-control and repeat spread alongside comparisons.',
   'Known-goal simulated delays, no live VLM/Astra; V3 combines optimizer, predictor and action-initialization changes.'])
 save(out/'summary.json',summary);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for name in ['tmp','logs','eval']: (out/name).mkdir()
 lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not gpu_available(0)['available']:raise RuntimeError('GPU occupied')
 paths=[f for base in ['scripts/agentic','scripts/recovery_controller_v3','source','configs','assets'] for f in (ROOT/base).rglob('*') if f.is_file() and f.suffix in ['.py','.json','.yaml','.yml','.usd','.usda','.usdc']]
 paths += [ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt']
 old=ROOT/'outputs/recovery_controller_v2/pilot_20261018'
 paths += [old/'train_completion/controller_final.pt']+[old/f'cache_{s}/request_state.pt' for s in SEEDS]
 frozen={str(p.relative_to(ROOT)):sha(p) for p in paths}
 rng=random.Random(181028);groups=[(s,d) for s in SEEDS for d in [6.,20.]];rng.shuffle(groups)
 schedule=[]
 for seed,delay in groups:
  order=METHODS.copy();rng.shuffle(order)
  for rep,methods in [(1,order),(2,list(reversed(order)))]:schedule.append(dict(seed=seed,delay=delay,repeat=rep,methods=methods))
 protocol=dict(version='recovery_v3_optimization_pilot',seeds=SEEDS,training_seeds=list(range(920000,920032)),training_delays=[2.,6.,12.],training_iterations=32,
  evaluation_delays=[0.,6.,20.],repetitions=2,methods=METHODS,schedule=schedule,planned_eval_trials=432,nominal_request_states=24,eligible_request_states=23,
  source_asset_weights_caches_sha256=frozen,visual_frontend='unchanged original renderer and RGB-D features; Gaussian diagnostic NOT applied',
  success='six consecutive post-response steps at <=0.16 rad, no prior drop, 300-step recovery',
  changes=['independent optimizers/encoders','value loss normalized','exact KL<=.02 with rollback','outcome replay and whole-seed holdout','baseline-preserving action initialization, smaller residual exploration'],
  scope='development comparison; no agent API or new-object claims; final checkpoint fixed by 32-update budget',created=time.time())
 save(out/'protocol.json',protocol)
 with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
  for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
 python='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
 env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',PAIRED_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
 env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
 state=dict(pid=os.getpid(),status='starting',completed_stages=0,planned_eval_trials=432);child=None
 def progress(**kwargs):state.update(kwargs,updated=time.time());save(out/'progress.json',state)
 def interrupt(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 def run(name,script,folder,args,timeout=1200):
  nonlocal child
  if folder.exists():raise RuntimeError('Existing result needs explicit review')
  if any(sha(ROOT/name)!=value for name,value in frozen.items()):raise RuntimeError('Frozen input changed')
  cmd=[python,str(ROOT/script),'--headless','--device','cuda:0','--output',str(folder),'--num-envs',str(N),'--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']+args
  save(out/'logs'/f'{name}_command.json',cmd)
  with (out/'logs'/f'{name}.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   begin=time.monotonic();progress(status=name,current_output=str(folder),child_pid=child.pid)
   while child.poll() is None:
    progress(stage_wall_seconds=round(time.monotonic()-begin,1))
    if time.monotonic()-begin>timeout:raise TimeoutError(name)
    time.sleep(5)
   if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}')
  if not (folder/'complete.json').exists():raise RuntimeError('Missing completion')
  progress(completed_stages=state['completed_stages']+1,child_pid=None)
 def evaluate(seed,delay,rep,method):
  name=f'seed_{seed}_delay_{delay:g}_r{rep}_{method}';folder=out/'eval'/name
  script='scripts/agentic/train_recovery_controller_v2.py' if method=='v2_learned' else 'scripts/agentic/train_recovery_controller_v3.py'
  mode='learned' if method in ['v2_learned','v3_learned'] else method
  args=['--seed',str(seed),'--delay',str(delay),'--mode',mode,'--request-cache',str(old/f'cache_{seed}/request_state.pt')]
  if method in ['v2_learned','v3_learned']:
   checkpoint=old/'train_completion/controller_final.pt' if method=='v2_learned' else out/'train/controller_final.pt'
   if method=='v3_learned' and sha(checkpoint)!=json.loads((out/'checkpoint.json').read_text())['sha256']:raise RuntimeError('V3 checkpoint changed')
   args+=['--checkpoint',str(checkpoint)]
  if delay>0 and rep==1 and method!='v2_learned':args+=['--record-slots','0','5']
  run(name,script,folder,args)
  save(folder/'method.json',dict(method=method,seed=seed,delay=delay,repeat=rep))
  summary=analyze(out);progress(completed_evaluation_trials=summary['completed_trials'])
 try:
  for seed in SEEDS:
   for rep in [1,2]:evaluate(seed,0.,rep,'continue')
   zero=[v for v in analyze(out)['zero_controls'] if v['seed']==seed]
   if sum(v['successes'] for v in zero)==0:raise RuntimeError('All zero-delay controls failed; retain seed and stop')
  run('train_v3','scripts/agentic/train_recovery_controller_v3.py',out/'train',['--mode','train','--iterations','32','--seed','920000','--objective','completion'],timeout=7200)
  save(out/'checkpoint.json',dict(sha256=sha(out/'train/controller_final.pt')))
  for g in schedule:
   for method in g['methods']:evaluate(g['seed'],g['delay'],g['repeat'],method)
  summary=analyze(out)
  assert summary['completed_trials']==432
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
