"""Finite 2x2 controlled training ablation followed by two-seed validation."""
import argparse,fcntl,hashlib,json,os,random,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/latency_counterfactual'),str(ROOT/'scripts/agentic')]
from run_original import gpu_available
from design import ARMS,SHARED_PREFIX,branch_for,weights_for
from factorial_analysis import METHODS,analyze,counts,read,save
PYTHON='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
PRIOR=ROOT/'outputs/recovery_controller_v4/paired_20261018'
V3=ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt'
LEGACY=PRIOR/'controller_final.pt'
EVAL_SEEDS=[85001,85002]
TRAIN_SEEDS=[930100,930101,930102,930103]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def registered_training():
 result=[]
 for i in range(1,9):
  path=PRIOR/f'train/group_{i:02d}/selection.json';previous=read(path)
  if previous['seed'] not in TRAIN_SEEDS:raise RuntimeError('Unexpected prior training seed')
  result.append(dict(iteration=i,seed=previous['seed'],delay=previous['delay'],draw_order=previous['branch_order'],
   failure_weights=previous['slot_multiplicity'],source=str(path.relative_to(ROOT)),source_sha256=sha(path)))
 if any(w!=1 for g in result[:4] for w in g['failure_weights']):raise RuntimeError('Shared-prefix weights must be uniform')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for name in ['logs','tmp','cache','train','models','eval']:(out/name).mkdir()
 lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 available=gpu_available(0)
 if not available['available']:raise RuntimeError('GPU occupied: '+str(available))
 if sha(V3)!='cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042':raise RuntimeError('V3 changed')
 if sha(LEGACY)!='d7d4b10e40fa9dabd674f56b5cd6d632c04590f37f4641f0ac8b6496563c45b0':raise RuntimeError('Legacy V4 changed')
 schedule=registered_training();rng=random.Random(185019)
 if set(TRAIN_SEEDS)&set(EVAL_SEEDS):raise RuntimeError('Train/test leakage')
 evaluation=[]
 groups=[(s,b,d) for s in EVAL_SEEDS for b in [0,1] for d in [6.,20.]];rng.shuffle(groups)
 for seed,branch,delay in groups:
  order=METHODS.copy();rng.shuffle(order)
  for rep,methods in [(1,order),(2,list(reversed(order)))]:evaluation.append(dict(seed=seed,branch=branch,delay=delay,repeat=rep,methods=methods))
 files=[f for base in ['scripts/agentic','scripts/recovery_factorial','scripts/latency_counterfactual','source','configs','assets'] for f in (ROOT/base).rglob('*') if f.is_file() and f.suffix in ['.py','.json','.yaml','.yml','.usd','.usda','.usdc']]
 files += [V3,LEGACY,ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt',ROOT/'tests/test_recovery_factorial.py']
 files += [ROOT/g['source'] for g in schedule]
 frozen={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}
 protocol=dict(version='factorial_recovery_v1',created=time.time(),arms=ARMS,train_seeds=TRAIN_SEEDS,evaluation_seeds=EVAL_SEEDS,
  training_schedule=schedule,evaluation_schedule=evaluation,methods=METHODS,gpu=available,
  training_budget=dict(logical_updates_per_arm=8,nominal_slots_per_arm=576,total_logical_slots=2304,shared_prefix_updates=4,actual_fresh_rollout_slots=1728,actual_optimizer_updates=24),
  evaluation_budget=dict(nominal_slots=1440,positive_delay_slots=1344,zero_delay_slots=96,branches=[0,1],delays=[6.,20.],repeats=2),
  shared_prefix='Within each pairing level, weight-off/on share identical checkpoints, optimizer/replay state and rollouts for groups 1-4, when all weights are one. New trajectories are collected after branching at group 5.',
  pairing_off='Six independent stochastic draws of original goal from the same request state. Draw ids and true goal branches are separate.',
  pairing_on='One independent draw for each of six correct goals from the same request state. Noise seed matched by draw id across factor cells.',
  failure_weighting='Frozen per-slot weights from prior training-only selection records. Same across pairing levels. Apply to PPO/value objectives without duplicating samples or extra epochs.',
  common_schedule='Use the preceding V4 training-only state/delay order for all four cells. No new adaptive state selection in any cell. This controls, but does not separately estimate, the effect of V4 adaptive seed selection.',
  optimizer='Same V3 architecture, loss coefficients, initialization, 3 epochs, batch size 256, optimizer rates and .02 KL guard. Actual valid counts and accepted steps can differ and are recorded.',
  legacy_reference='Original V4 is not the both-on factorial cell: its adaptive sampling and data-duplication optimizer implementation differ. No claim of uniquely explaining every part of its regression.',
  recovery_diagnostics='Response and handoff orientation errors, surviving counts, per-phase rate, drops, completion and goal branches. Ground truth logged/scored only, never added to actor.',
  success='Selected correct full orientation <= .16 rad for six consecutive post-response control steps, no prior drop, within 300 steps. Control through wait and first 60 recovery steps.',
  evidence_limits=['One training realization per factorial branch and two evaluation seeds. Repeated goals/delays are correlated.',
   'Known canonical goals and simulated response delays, no live VLM/Astra.',
   'Final registered eighth update evaluated. No test-driven model selection or replacement of frozen V3.',
   'All failed goals and seeds retained. No silent stage retries.'],source_asset_weights_sha256=frozen)
 save(out/'protocol.json',protocol)
 with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
  for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
 env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PAIRED_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
 env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
 state=dict(pid=os.getpid(),status='starting',completed_stages=0,planned_stages=290,completed_physical_updates=0,planned_physical_updates=24,
  completed_logical_updates=0,planned_logical_updates=32,completed_training_trials=0,planned_training_trials=1728,
  logical_training_trials=0,planned_logical_training_trials=2304,completed_eval_trials=0,planned_eval_trials=1440)
 child=None;cache_hashes={};model_hashes={};state_hashes={};histories={arm:[] for arm in ARMS}
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def stop(*unused):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def run(name,folder,args,is_update=False):
  nonlocal child
  if folder.exists():raise RuntimeError('Existing stage needs review')
  if any(sha(ROOT/f)!=h for f,h in frozen.items()):raise RuntimeError('Frozen source/weights changed')
  if any(sha(Path(f))!=h for f,h in {**cache_hashes,**model_hashes}.items()):raise RuntimeError('Cache or model changed')
  if is_update:cmd=[PYTHON,str(ROOT/'scripts/recovery_factorial/update.py'),'--output',str(folder)]+args
  else:cmd=[PYTHON,str(ROOT/'scripts/agentic/collect_recovery_factorial.py'),'--headless','--device','cuda:0','--num-envs','12','--output',str(folder),'--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']+args
  save(out/'logs'/f'{name}_command.json',cmd)
  with (out/'logs'/f'{name}.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.monotonic();progress(status=name,current_output=str(folder),child_pid=child.pid)
   while child.poll() is None:
    progress(stage_wall_seconds=round(time.monotonic()-start,1))
    if time.monotonic()-start>1800:raise TimeoutError(name)
    time.sleep(5)
   if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}')
  if not (folder/('update.json' if is_update else 'complete.json')).exists():raise RuntimeError('Missing completion')
  if (folder/'goal_switch_audit.json').exists():
   audit=read(folder/'goal_switch_audit.json');key=audit['request_cache_sha256'];value=(audit['physical_sha256'],audit['request_observation_sha256'])
   if key in state_hashes and state_hashes[key]!=value:raise RuntimeError('Mismatched physical/history state')
   state_hashes[key]=value;save(out/'state_hashes.json',state_hashes)
  progress(completed_stages=state['completed_stages']+1,child_pid=None)
 try:
  caches={}
  for seed in TRAIN_SEEDS:
   source=PRIOR/f'cache/seed_{seed}/request_state.pt';dest=out/'cache'/f'train_{seed}.pt';shutil.copyfile(source,dest)
   if sha(source)!=sha(dest):raise RuntimeError('Training cache copy mismatch')
   caches[seed]=dest;cache_hashes[str(dest)]=sha(dest)
  save(out/'cache_hashes.json',cache_hashes)
  checkpoints={arm:V3 for arm in ARMS}
  for g in schedule:
   iteration=g['iteration'];active=['uniform_original','paired_only'] if iteration<=4 else list(ARMS)
   if iteration%2==0:active.reverse()
   for arm in active:
    group=out/'train'/arm/f'group_{iteration:02d}';group.mkdir(parents=True)
    ck=checkpoints[arm];weights=weights_for(arm,g['failure_weights'])
    save(group/'selection.json',dict(**g,arm=arm,factors=ARMS[arm],weights=weights,checkpoint=str(ck),checkpoint_sha256=sha(ck)))
    collections=[];all_rows=[]
    for draw in g['draw_order']:
     branch=branch_for(arm,draw);folder=group/f'draw_{draw}';export=group/f'draw_{draw}.pt'
     args=['--mode','collect','--seed',str(g['seed']),'--delay',str(g['delay']),'--goal-branch',str(branch),'--draw-id',str(draw),'--training-arm',arm,
      '--noise-seed',str(185000+iteration*10+draw),'--checkpoint',str(ck),'--request-cache',str(caches[g['seed']]),'--collection',str(export)]
     run(f'train_{iteration:02d}_{arm}_d{draw}',folder,args)
     collections.append(str(export));all_rows+=read(folder/'eval_trials.json');progress(completed_training_trials=state['completed_training_trials']+12)
    run(f'update_{iteration:02d}_{arm}',group/'update',['--checkpoint',str(ck),'--collections',*collections,'--slot-weights',*[str(v) for v in weights],'--arm',arm,'--iteration',str(iteration)],is_update=True)
    checkpoints[arm]=group/'update/controller.pt';model_hashes[str(checkpoints[arm])]=sha(checkpoints[arm]);save(out/'model_hashes.json',model_hashes)
    row=dict(iteration=iteration,arm=arm,seed=g['seed'],delay=g['delay'],weights=weights,shared=False,**counts(all_rows),optimization=read(group/'update/update.json'),checkpoint=str(checkpoints[arm]))
    histories[arm].append(row)
    logical_multiplier=2 if iteration<=4 else 1
    if iteration<=4:
     target=next(a for a,source in SHARED_PREFIX.items() if source==arm)
     checkpoints[target]=checkpoints[arm]
     shared=dict(row,arm=target,shared=True,shared_from=arm,shared_group=str(group),explanation='Same training when weights are all one, shared exactly to remove nuisance differences.')
     histories[target].append(shared);alias=out/'train'/target/f'group_{iteration:02d}';alias.mkdir(parents=True);save(alias/'shared.json',shared)
    save(out/'training.json',histories)
    progress(completed_physical_updates=state['completed_physical_updates']+1,completed_logical_updates=state['completed_logical_updates']+logical_multiplier,
     logical_training_trials=state['logical_training_trials']+72*logical_multiplier,last_training=row)
  finals={}
  for arm,ck in checkpoints.items():
   target=out/'models'/f'{arm}.pt';shutil.copyfile(ck,target);finals[arm]=target;model_hashes[str(target)]=sha(target)
  save(out/'model_hashes.json',model_hashes)
  save(out/'training_complete.json',dict(status='completed',completed=time.time(),checkpoints={a:dict(path=str(f),sha256=sha(f)) for a,f in finals.items()},physical_updates=24,logical_updates=32,actual_rollout_slots=1728,logical_rollout_slots=2304))
  for seed in EVAL_SEEDS:
   folder=out/'cache'/f'eval_{seed}';run(folder.name,folder,['--mode','cache','--seed',str(seed)])
   caches[seed]=folder/'request_state.pt';cache_hashes[str(caches[seed])]=sha(caches[seed]);save(out/'cache_hashes.json',cache_hashes)
  def evaluate(seed,branch,delay,rep,method):
   name=f's{seed}_b{branch}_d{delay:g}_r{rep}_{method}';folder=out/'eval'/name
   args=['--mode','continue' if delay==0 else 'slow' if method=='slow' else 'learned','--seed',str(seed),'--goal-branch',str(branch),'--delay',str(delay),'--request-cache',str(caches[seed])]
   if delay>0 and method!='slow':args+=['--checkpoint',str(V3 if method=='v3' else LEGACY if method=='legacy_v4' else finals[method])]
   if delay>0 and rep==1 and method in ['v3',*ARMS]:
    import torch
    eligible=torch.load(caches[seed],map_location='cpu',weights_only=False)['eligible'].nonzero().flatten().tolist()[:1]
    args+=['--record-slots',*[str(i) for i in eligible]]
   run(name,folder,args);save(folder/'method.json',dict(seed=seed,branch=branch,delay=delay,repeat=rep,method=method))
   progress(completed_eval_trials=analyze(out)['completed_trials'])
  for seed in EVAL_SEEDS:
   for branch in [0,1]:
    for rep in [1,2]:evaluate(seed,branch,0.,rep,'continue')
  for g in evaluation:
   for method in g['methods']:evaluate(g['seed'],g['branch'],g['delay'],g['repeat'],method)
  result=analyze(out)
  assert result['completed_trials']==1440 and state['completed_stages']==290
  progress(status='completed_factorial_training_and_validation',current_output=None,child_pid=None)
 except BaseException as exc:
  progress(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  lock.close()
if __name__=='__main__':main()
