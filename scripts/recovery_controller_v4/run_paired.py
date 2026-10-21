"""Finite paired-goal training and two-seed validation on a free local GPU."""
import argparse,fcntl,hashlib,json,os,random,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/latency_counterfactual'),str(ROOT/'scripts/agentic')]
from run_original import gpu_available
TRAIN_SEEDS=[930100,930101,930102,930103]
EVAL_SEEDS=[84001,84002]
METHODS=['slow','v3','v4']
PYTHON='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
V3=ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt'
N=12

def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def counts(rows):
 good=[r for r in rows if r['eligible']]
 return dict(nominal=len(rows),eligible=len(good),successes=sum(r['success'] for r in good),
  success_rate=sum(r['success'] for r in good)/len(good) if good else None,
  wait_drops=sum(r['wait_drop'] for r in good),recovery_drops=sum(r['resume_drop'] for r in good))

def analyze(out):
 arms=[]
 for f in (out/'eval').glob('*/method.json'):
  folder=f.parent
  if not (folder/'complete.json').exists():continue
  meta=read(f);rows=read(folder/'eval_trials.json');audit=read(folder/'goal_switch_audit.json');prefix=read(folder/'prefix_replay_audit.json')
  if len(rows)!=N or audit['max_physical_change'] or prefix['max_abs_error'] or not prefix['eligible_equal']:raise RuntimeError('Invalid evaluation arm')
  if audit['goals']!=[r['goal'] for r in rows]:raise RuntimeError('Scored goal mismatch')
  arms.append((meta,rows))
 group=lambda m:(m['seed'],m['branch'],m['delay'],m['repeat'])
 groups={group(m) for m,_ in arms if m['delay']>0}
 balanced={g for g in groups if {m['method'] for m,_ in arms if group(m)==g}==set(METHODS)}
 pairs=[(m,r) for m,rr in arms if group(m) in balanced for r in rr]
 report=dict(completed_arms=len(arms),completed_trials=sum(len(rr) for _,rr in arms),balanced_groups=[list(g) for g in sorted(balanced)],
  methods={method:counts([r for m,r in pairs if m['method']==method]) for method in METHODS},
  by_group={str(g):{method:counts([r for m,r in pairs if group(m)==g and m['method']==method]) for method in METHODS} for g in sorted(balanced)},
  zero_controls=[dict(**m,**counts(rr)) for m,rr in arms if m['delay']==0],
  distinct_eligible_states=len({(m['seed'],r['slot']) for m,r in pairs if r['eligible']}),
  interpretation='Known correct goal changes at request. No live VLM. Opposing goals share physical states, not independent scenes. Two seed clusters. No test-driven checkpoint selection.')
 save(out/'summary.json',report);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for name in ['logs','tmp','train','cache','eval']:(out/name).mkdir()
 lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 available=gpu_available(0)
 if not available['available']:raise RuntimeError('GPU occupied: '+str(available))
 if sha(V3)!='cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042':raise RuntimeError('V3 checkpoint changed')
 if set(TRAIN_SEEDS)&set(EVAL_SEEDS):raise RuntimeError('Training/evaluation leakage')
 files=[f for base in ['scripts/agentic','scripts/recovery_controller_v4','source','configs','assets'] for f in (ROOT/base).rglob('*') if f.is_file() and f.suffix in ['.py','.json','.yaml','.yml','.usd','.usda','.usdc']]
 files += [V3,ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt',ROOT/'tests/test_recovery_goal_pairing.py']
 frozen={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}
 rng=random.Random(184018);schedule=[]
 for seed in EVAL_SEEDS:
  for branch in [0,1]:
   for delay in [6.,20.]:
    methods=METHODS.copy();rng.shuffle(methods)
    for rep,order in [(1,methods),(2,list(reversed(methods)))]:schedule.append(dict(seed=seed,branch=branch,delay=delay,repeat=rep,methods=order))
 protocol=dict(version='v4_correct_goal_pairing',created=time.time(),train_seeds=TRAIN_SEEDS,evaluation_seeds=EVAL_SEEDS,
  train_branches=list(range(6)),evaluation_branches=[0,1],branch_1='opposing face (base xor 1)',
  updates=8,nominal_training_slots=576,nominal_evaluation_slots=672,evaluation_schedule=schedule,
  train_delays=[2.,6.,12.,6.,12.,6.,2.,12.],evaluation_delays=[6.,20.],repeats=2,
  training='Warm start frozen v3. Collect all six goals with one fixed policy, then one grouped PPO update. No reward for goal-induced action differences.',
  hard_states='First four groups cover each training cache. Remaining groups sample training seeds with half uniform and half failure-weighted probability. Previously hard physical slots have multiplicity two for all goals. Fresh on-policy collection only. Test outcomes are never used.',
  success='Correct selected full orientation <=.16 rad for six consecutive post-response control steps, no earlier drop, within 300 steps.',
  matching='Fresh simulator per branch. Exact prefix command replay, mechanical state hash, request-history hash, target change without simulation advance. Student/controller/scorer all receive same correct goal.',
  frozen_student='RGB-D, action scale 1, same checkpoint and language adapter',
  source_asset_weights_sha256=frozen,initial_checkpoint=str(V3),gpu=available,
  limitations=['Known goal supplied before wait; no live agent or unknown future goal.',
   'Two new evaluation seeds and two opposing goal branches. Repeats/delays/goals share physical states.',
   'Architecture remains v3. Goal pairing plus failure curriculum are combined in this first pilot, not isolated causal contributions.',
   'Renderer variability persists. All registered arms and failed outcomes are retained.',
   'No silent retries, no seed exclusion, no evaluation-based checkpoint selection.'])
 save(out/'protocol.json',protocol)
 with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
  for f in frozen:tar.add(ROOT/f,arcname=f,recursive=False)
 env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PAIRED_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
 env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
 state=dict(pid=os.getpid(),status='starting',completed_stages=0,planned_stages=118,completed_updates=0,planned_updates=8,
  completed_training_trials=0,planned_training_trials=576,completed_eval_trials=0,planned_eval_trials=672)
 child=None;cache_hashes={};state_hashes={};history=[];last_training={}
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def stop(*unused):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def run(name,folder,args,update=False,timeout=1500):
  nonlocal child
  if folder.exists():raise RuntimeError('Existing stage requires review')
  if any(sha(ROOT/f)!=h for f,h in frozen.items()):raise RuntimeError('Frozen source/weights changed')
  if any(sha(Path(f))!=h for f,h in cache_hashes.items()):raise RuntimeError('Cache changed')
  if update:
   cmd=[PYTHON,str(ROOT/'scripts/recovery_controller_v4/update_pair.py'),'--output',str(folder)]+args
  else:
   cmd=[PYTHON,str(ROOT/'scripts/agentic/collect_paired_recovery.py'),'--headless','--device','cuda:0','--num-envs',str(N),'--output',str(folder),'--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']+args
  save(out/'logs'/f'{name}_command.json',cmd)
  with (out/'logs'/f'{name}.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.monotonic();progress(status=name,current_output=str(folder),child_pid=child.pid)
   while child.poll() is None:
    progress(stage_wall_seconds=round(time.monotonic()-start,1))
    if time.monotonic()-start>timeout:raise TimeoutError(name)
    time.sleep(5)
   if child.returncode:raise RuntimeError(f'{name} exit {child.returncode}')
  if not (folder/('update.json' if update else 'complete.json')).exists():raise RuntimeError('Missing completion')
  if (folder/'goal_switch_audit.json').exists():
   aud=read(folder/'goal_switch_audit.json');key=aud['request_cache_sha256']
   pair=(aud['physical_sha256'],aud['request_observation_sha256'])
   if key in state_hashes and state_hashes[key]!=pair:raise RuntimeError('Physical or observation mismatch across branches')
   state_hashes[key]=pair;save(out/'state_hashes.json',state_hashes)
  progress(completed_stages=state['completed_stages']+1,child_pid=None)
 def cache(seed):
  folder=out/'cache'/f'seed_{seed}';run(folder.name,folder,['--mode','cache','--seed',str(seed)])
  cache_hashes[str(folder/'request_state.pt')]=sha(folder/'request_state.pt');save(out/'cache_hashes.json',cache_hashes)
  return folder/'request_state.pt'
 try:
  train_cache={seed:cache(seed) for seed in TRAIN_SEEDS};checkpoint=V3
  for iteration,delay in enumerate(protocol['train_delays'],1):
   # Sampling probabilities and slot weights depend only on preceding TRAIN groups.
   if iteration<=4:seed=TRAIN_SEEDS[iteration-1];probabilities=None
   else:
    scores=[1.-counts(last_training[s])['success_rate']+.1 if counts(last_training[s])['success_rate'] is not None else .1 for s in TRAIN_SEEDS]
    probabilities=[.5/len(TRAIN_SEEDS)+.5*v/sum(scores) for v in scores]
    seed=rng.choices(TRAIN_SEEDS,weights=probabilities,k=1)[0]
   prior=last_training.get(seed)
   mult=[1]*N
   if prior:
    for slot in range(N):
     rr=[r for r in prior if r['slot']==slot]
     if sum(r['eligible'] and not r['success'] for r in rr)>=2 or any(r['eligible'] and (r['wait_drop'] or r['resume_drop']) for r in rr):mult[slot]=2
   group=out/'train'/f'group_{iteration:02d}';group.mkdir();branches=list(range(6));rng.shuffle(branches)
   selection=dict(iteration=iteration,seed=seed,delay=delay,probabilities=probabilities,slot_multiplicity=mult,branch_order=branches,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
   save(group/'selection.json',selection);collections=[];all_rows=[]
   for branch in branches:
    folder=group/f'branch_{branch}';export=group/f'branch_{branch}.pt'
    run(f'train_{iteration:02d}_b{branch}',folder,['--mode','collect','--seed',str(seed),'--delay',str(delay),'--goal-branch',str(branch),'--noise-seed',str(184000+iteration),'--checkpoint',str(checkpoint),'--request-cache',str(train_cache[seed]),'--collection',str(export)])
    collections.append(str(export));all_rows+=read(folder/'eval_trials.json')
    progress(completed_training_trials=state['completed_training_trials']+N)
   run(f'update_{iteration:02d}',group/'update',['--checkpoint',str(checkpoint),'--collections',*collections,'--multiplicity',*[str(x) for x in mult],'--iteration',str(iteration)],update=True)
   checkpoint=group/'update/controller.pt';last_training[seed]=all_rows
   row=dict(**selection,**counts(all_rows),optimization=read(group/'update/update.json'));history.append(row);save(out/'training.json',history)
   progress(completed_updates=iteration,last_training=row)
  shutil.copyfile(checkpoint,out/'controller_final.pt');checkpoint=out/'controller_final.pt'
  save(out/'training_complete.json',dict(status='completed',checkpoint=str(checkpoint),sha256=sha(checkpoint),updates=8,nominal_trials=576,completed=time.time()))
  eval_cache={seed:cache(seed) for seed in EVAL_SEEDS}
  def evaluate(seed,branch,delay,rep,method):
   name=f's{seed}_b{branch}_d{delay:g}_r{rep}_{method}';folder=out/'eval'/name
   args=['--mode','continue' if delay==0 else 'slow' if method=='slow' else 'learned','--seed',str(seed),'--goal-branch',str(branch),'--delay',str(delay),'--request-cache',str(eval_cache[seed])]
   if delay>0 and method!='slow':args+=['--checkpoint',str(V3 if method=='v3' else checkpoint)]
   if delay>0 and rep==1 and method in ['slow','v4']:
    # Select eligible slots by prefix only, before observing evaluation outcomes.
    import torch
    eligible=torch.load(eval_cache[seed],map_location='cpu',weights_only=False)['eligible'].nonzero().flatten().tolist()[:2]
    args+=['--record-slots',*[str(i) for i in eligible]]
   run(name,folder,args);save(folder/'method.json',dict(seed=seed,branch=branch,delay=delay,repeat=rep,method=method))
   progress(completed_eval_trials=analyze(out)['completed_trials'])
  for seed in EVAL_SEEDS:
   for branch in [0,1]:
    for rep in [1,2]:evaluate(seed,branch,0.,rep,'continue')
  for g in schedule:
   for method in g['methods']:evaluate(g['seed'],g['branch'],g['delay'],g['repeat'],method)
  assert analyze(out)['completed_trials']==672
  progress(status='completed_paired_training_and_validation',child_pid=None,current_output=None)
 except BaseException as exc:
  progress(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  lock.close()
if __name__=='__main__':main()
