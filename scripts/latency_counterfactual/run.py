"""Finite paired evaluation: one fresh simulator process for every arm."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
from analyze import analyze
ROOT = Path(__file__).resolve().parents[2]
MODES = ['continue', 'slow', 'hold', 'learned']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)

def gpu_rows():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [dict(zip(['index','uuid','name','memory','total','util'], [v.strip() for v in line.split(',')])) for line in output.splitlines()]
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    busy = {line.split(',')[0].strip() for line in output.splitlines()}
    for row in rows: row['compute_process_present'] = row['uuid'] in busy
    return rows

def collect(dest, delays, num_envs, complete=False):
    """Require matching state, settings, all arms and full zero-delay traces."""
    rows, audits, settings = [], [], []
    cache_sha = sha(dest/'cache/request_state.pt')
    for delay in delays:
        for mode in MODES:
            branch = dest/f'delay_{delay:g}_{mode}'
            end = json.loads((branch/'complete.json').read_text())
            batch = json.loads((branch/'eval_trials.json').read_text())
            audit = json.loads((branch/'state_audit.json').read_text())
            if end['trials'] != num_envs or len(batch) != num_envs or len(audit) != 1:
                raise RuntimeError(f'Incomplete arm: {branch}')
            if {r['slot'] for r in batch} != set(range(num_envs)):
                raise RuntimeError('Missing or duplicate environment slot')
            if any(r['request_cache_sha256'] != cache_sha or r['mode'] != mode or r['delay_s'] != delay for r in batch):
                raise RuntimeError(f'Mismatched cache or arm labels: {branch}')
            rows.extend(batch); audits.extend(audit)
            settings.append(json.loads((branch/'render_settings.json').read_text()))
    if any(s != settings[0] for s in settings): raise RuntimeError('Renderer settings changed across arms')
    zero = [r for r in rows if r['delay_s'] == 0]
    traces = {r['trajectory_sha256'] for r in audits if r['delay_s'] == 0}
    consistent = len(traces) == 1 and len(zero) == num_envs*4
    for slot in range(num_envs):
        if len({r['eligible'] for r in rows if r['slot'] == slot}) != 1:
            raise RuntimeError('Unpaired eligibility')
        outcomes = {(r['success'],r['wait_drop'],r['resume_drop'],r['resume_steps_to_success']) for r in zero if r['slot'] == slot}
        consistent &= len(outcomes) == 1
    save(dest/'negative_control.json', dict(consistent=consistent,full_trajectory_sha256=sorted(traces)))
    if not consistent: raise RuntimeError(f'Zero-delay full trajectory audit failed: {dest}')
    if complete:
        save(dest/'eval_trials.json', rows); save(dest/'state_audit.json', audits)
        save(dest/'complete.json', dict(status='completed',trials=len(rows),zero_delay_consistent=True,
            eligible_request_states=sum(r['eligible'] for r in zero if r['mode'] == MODES[0]),
            zero_delay_trajectory_sha256=next(iter(traces)),simulator_process='fresh for every branch'))
    return rows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',type=int); parser.add_argument('--wait-hours',type=float,default=12)
    parser.add_argument('--smoke-only',action='store_true'); args=parser.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    owner=(out/'runner.lock').open('a'); fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),status='preparing',planned_formal_trials=3456,completed_formal_trials=0,completed_formal_arms=0)
    def update(**fields):
        state.update(fields,updated=time.time()); save(out/'progress.json',state)
    sources=[p for base in ['scripts/latency_counterfactual','scripts/agentic','source'] for p in (ROOT/base).rglob('*.py')]
    weights=[ROOT/'outputs/latency_controller_v1/train_v2_20261015/controller_v1_final.pt',
             ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',
             ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt']
    configuration=[p for base in ['assets','configs'] for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ['.py','.usd','.usda','.usdc','.json','.yaml','.yml']]
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sources+weights+configuration}
    protocol=dict(version='paired_waiting_v3',seeds=list(range(81001,81013)),num_envs=12,
        delays=[0,.5,2,6,12,20],modes=MODES,resume_steps=300,total_nominal_trials=3456,
        smoke_seed=80999,smoke_excluded=True,process_isolation='fresh per cache and arm; exact prefix action replay preserves contact history',rendering='deterministic ambient-only; all arms identical, separate from live v7 renderer',
        zero_control='all four full zero-delay trajectories must have identical hashes',source_weight_sha256=frozen)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol: raise RuntimeError('Protocol changed')
    else: save(out/'protocol.json',protocol)
    child=None; gpu_lock=None
    def interrupted(*unused): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        stable={}; deadline=time.monotonic()+args.wait_hours*3600
        while gpu_lock is None:
            rows=gpu_rows()
            for row in rows:
                gpu=int(row['index'])
                idle=(not row['compute_process_present'] and int(row['memory'])<4096 and int(row['total'])>=20000
                      and (args.gpu is not None or int(row['util'])<35) and (args.gpu is None or gpu==args.gpu))
                stable[gpu]=stable.get(gpu,0)+1 if idle else 0
                if stable[gpu]<3: continue
                handle=Path('/tmp',f'agent_inhand_paired_gpu_{gpu}.lock').open('a')
                try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: handle.close(); continue
                gpu_lock=handle; break
            update(status='waiting_for_idle_gpu',gpu_snapshot=rows)
            if gpu_lock is not None: break
            if time.monotonic()>=deadline: raise TimeoutError('No idle GPU within finite queue timeout')
            time.sleep(10)
        (out/'tmp').mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_VISIBLE_DEVICES=str(gpu),PAIRED_RENDER_GPU=str(gpu),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
        env['PYTHONPATH']=':'.join(str(ROOT/p) for p in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
        update(status='starting_smoke',gpu=gpu)
        def run_arm(dest,seed,num_envs,resume_steps,extra):
            nonlocal child
            if (dest/'complete.json').exists(): return
            if dest.exists(): raise RuntimeError(f'Incomplete arm needs review: {dest}')
            if any(sha(ROOT/k)!=v for k,v in frozen.items()): raise RuntimeError('Frozen source or weights changed')
            cmd=[sys.executable,str(ROOT/'scripts/latency_counterfactual/evaluate.py'),'--headless','--device','cuda:0',
                '--output',str(dest),'--seed',str(seed),'--num-envs',str(num_envs),'--resume-steps',str(resume_steps),
                '--analytic-render','--no-dither','--fixed-render-clock','--ambient-only','--replay-prefix','--kit_args',
                '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']+extra
            save(dest.parent/(dest.name+'_command.json'),cmd)
            with (dest.parent/(dest.name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                begin=time.monotonic()
                while child.poll() is None:
                    update(child_pid=child.pid,current_arm_wall_s=round(time.monotonic()-begin,1))
                    if time.monotonic()-begin>1200: raise TimeoutError(f'Arm exceeded 20-minute watchdog: {dest}')
                    time.sleep(5)
                if child.returncode: raise RuntimeError(f'{dest.name} exited {child.returncode}; see log')
            if not (dest/'complete.json').exists(): raise RuntimeError(f'Missing completion record: {dest}')
        def run_seed(seed,smoke=False):
            dest=out/('smoke' if smoke else f'seed_{seed}')
            if (dest/'complete.json').exists(): return
            dest.mkdir(exist_ok=True); n=6 if smoke else 12; resume=90 if smoke else 300
            delays=[0,6] if smoke else protocol['delays']
            update(status='smoke_running' if smoke else 'evaluating',current_seed=seed,current_mode=None,current_delay_s=None)
            run_arm(dest/'cache',seed,n,resume,['--cache-only'])
            eligible=json.loads((dest/'cache/request.json').read_text())['eligible']
            if smoke and sum(eligible)<3: raise RuntimeError('Too few smoke request survivors')
            for index,delay in enumerate(delays):
                shift=(seed+index)%4
                for mode in MODES[shift:]+MODES[:shift]:
                    update(current_mode=mode,current_delay_s=delay)
                    extra=['--request-cache',str(dest/'cache/request_state.pt'),'--delays',str(delay),'--modes',mode]
                    if delay==0: extra+=['--audit-steps','4']
                    if (smoke or seed==81001) and delay in [6,20]: extra+=['--record-slots','0','5','--record-delays',str(delay)]
                    run_arm(dest/f'delay_{delay:g}_{mode}',seed,n,resume,extra)
                    if not smoke:
                        arms=list(out.glob('seed_*/delay_*/complete.json'))
                        update(completed_formal_arms=len(arms),completed_formal_trials=len(arms)*12)
                if delay==0: collect(dest,[0],n)
            collect(dest,delays,n,complete=True)
        run_seed(80999,smoke=True)
        if not args.smoke_only:
            for seed in protocol['seeds']:
                run_seed(seed); summary=analyze(out)
                update(completed_formal_trials=summary['completed_trials'],fully_validated_seeds=summary['completed_seeds'])
        update(status='completed',current_seed=None,child_pid=None)
    except BaseException as error:
        update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error)); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        if gpu_lock: gpu_lock.close()
if __name__=='__main__': main()
