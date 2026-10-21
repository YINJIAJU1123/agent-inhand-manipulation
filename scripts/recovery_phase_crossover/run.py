"""Finite two-seed test of frozen waiting and recovery actor combinations."""
import argparse,fcntl,hashlib,os,random,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/latency_counterfactual'),str(ROOT/'scripts/agentic')]
from run_original import gpu_available
from phase_analysis import METHODS,read,save,analyze
PYTHON='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
PRIOR=ROOT/'outputs/recovery_factorial/ablation_20261019'
CHECKPOINTS={'v3':ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt','new':PRIOR/'models/uniform_original.pt'}
EXPECTED={'v3':'cccc5515e67ffc3960f6d98db57760493b2a6aaecea96ec0fd44e2a734e72042','new':'531839eaa17dc347b4b58446a103c84c21f5048f70f8c307d8ae7b9c1d4dd738'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n in ['logs','eval','tmp']:(out/n).mkdir()
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=gpu_available(0)
    if not gpu['available']:raise RuntimeError('GPU occupied: '+str(gpu))
    for name,path in CHECKPOINTS.items():
        if sha(path)!=EXPECTED[name]:raise RuntimeError('Checkpoint changed: '+name)
    caches={s:PRIOR/f'cache/eval_{s}/request_state.pt' for s in [85001,85002]}
    frozen=read(PRIOR/'protocol.json')['source_asset_weights_sha256'].copy()
    for path in [*CHECKPOINTS.values(),*caches.values(),*(ROOT/'scripts/recovery_phase_crossover').glob('*.py')]:frozen[str(path.relative_to(ROOT))]=sha(path)
    if any(sha(ROOT/f)!=h for f,h in frozen.items()):raise RuntimeError('Prior sources or weights changed')
    rng=random.Random(185020);groups=[(s,b,d) for s in caches for b in [0,1] for d in [6.,20.]];rng.shuffle(groups);schedule=[]
    for seed,branch,delay in groups:
        order=list(METHODS);rng.shuffle(order)
        for rep,methods in [(1,order),(2,list(reversed(order)))]:
            for method in methods:schedule.append(dict(seed=seed,branch=branch,delay=delay,repeat=rep,method=method,wait=METHODS[method][0],recovery=METHODS[method][1]))
    protocol=dict(version='phase_crossover_v1',created=time.time(),methods=METHODS,checkpoints={k:dict(path=str(p),sha256=EXPECTED[k]) for k,p in CHECKPOINTS.items()},
        schedule=schedule,nominal_trials=768,seeds=list(caches),gpu=gpu,source_cache_weight_sha256=frozen,
        intervention='Change only which frozen actor is selected before versus after the simulated response. Keep RGB-D history, proprioception, correct goal, student, action mapping and two-second bridge unchanged.',
        selection='New means the unpaired, unweighted continued-training model. Chosen after the completed factorial diagnostic to isolate its extra recovery drops.',
        evidence='Exploratory follow-up on the already observed 85001/85002 states. No held-out performance claim. No fitting or automatic baseline replacement.',
        controls='Both same-checkpoint combinations rerun fresh. Prior zero-delay results remain separate. Identical-checkpoint action distribution equality and response-boundary selection checked on real frozen observations before launch.',
        completion='16 state/goal/delay/repeat groups, four methods each, 12 slots per method. All cases retained. No silent retries.')
    save(out/'protocol.json',protocol)
    with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
        for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PAIRED_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
    env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
    state=dict(pid=os.getpid(),status='starting',completed_arms=0,planned_arms=64,completed_trials=0,planned_trials=768);child=None
    known_states=read(PRIOR/'state_hashes.json')
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def stop(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for m in schedule:
            if any(sha(ROOT/f)!=h for f,h in frozen.items()):raise RuntimeError('Frozen experiment dependency changed')
            name=f"s{m['seed']}_b{m['branch']}_d{m['delay']:g}_r{m['repeat']}_{m['method']}";folder=out/'eval'/name
            if folder.exists():raise RuntimeError('Existing arm requires review')
            cmd=[PYTHON,str(ROOT/'scripts/recovery_phase_crossover/evaluate.py'),'--mode','learned','--headless','--device','cuda:0','--num-envs','12','--output',str(folder),
                '--seed',str(m['seed']),'--goal-branch',str(m['branch']),'--delay',str(m['delay']),'--request-cache',str(caches[m['seed']]),
                '--checkpoint',str(CHECKPOINTS[m['wait']]),'--recovery-checkpoint',str(CHECKPOINTS[m['recovery']]),
                '--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
            if m['repeat']==1:cmd+=['--record-slots','0']
            save(out/'logs'/f'{name}_command.json',cmd)
            with (out/'logs'/f'{name}.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
                progress(status='evaluating',current_method=m,current_output=str(folder),child_pid=child.pid)
                while child.poll() is None:
                    progress(stage_wall_seconds=round(time.monotonic()-start,1))
                    if time.monotonic()-start>1800:raise TimeoutError(name)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}')
            end=read(folder/'complete.json');rows=read(folder/'eval_trials.json');audit=read(folder/'goal_switch_audit.json');prefix=read(folder/'prefix_replay_audit.json');manifest=read(folder/'manifest.json')
            if end['trials']!=12 or len(rows)!=12 or {r['slot'] for r in rows}!=set(range(12)):raise RuntimeError('Incomplete arm')
            if audit['max_physical_change']!=0 or prefix['max_abs_error']!=0 or not prefix['eligible_equal']:raise RuntimeError('Physical prefix mismatch')
            if [audit['physical_sha256'],audit['request_observation_sha256']]!=known_states[audit['request_cache_sha256']]:raise RuntimeError('Request differs from original factorial run')
            if [r['goal'] for r in rows]!=audit['goals']:raise RuntimeError('Scorer goal differs')
            if manifest['wait_checkpoint_sha256']!=EXPECTED[m['wait']] or manifest['recovery_checkpoint_sha256']!=EXPECTED[m['recovery']]:raise RuntimeError('Wrong phase checkpoint')
            for d in read(folder/'decisions.json'):
                if d['policy_phase']!=('wait' if d['step']<m['delay']*30 else 'recovery'):raise RuntimeError('Wrong phase label')
            save(folder/'method.json',m);result=analyze(out);progress(completed_arms=result['completed_arms'],completed_trials=result['completed_trials'],child_pid=None)
        assert result['completed_trials']==768 and result['balanced_groups']==16
        progress(status='completed_phase_crossover',current_output=None,current_method=None,child_pid=None)
    except BaseException as exc:
        progress(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        lock.close()
if __name__=='__main__':main()
