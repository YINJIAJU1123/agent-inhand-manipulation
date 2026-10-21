"""Finite original-render diagnostic on exactly two seeds; no live API calls."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, tarfile, time
from pathlib import Path
from analyze import analyze, stats

ROOT=Path(__file__).resolve().parents[2]
SEEDS=[81001,81002]
MODES=['continue','slow','hold','learned']
DELAYS=[0.,.5,2.,6.,12.,20.]
N=12
MIN_ELIGIBLE=6
MIN_BASELINE_RATE=.5
MAX_ZERO_SPREAD=.25

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def baseline_gate(rows):
    valid=[r for r in rows if r['eligible']]
    passed=len(valid)>=MIN_ELIGIBLE and sum(r['success'] for r in valid)>=MIN_BASELINE_RATE*len(valid)
    result=dict(passed=passed,minimum_eligible=MIN_ELIGIBLE,minimum_success_rate=MIN_BASELINE_RATE,**stats(rows))
    return result

def gpu_available(index):
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
    processes=[dict(zip(['uuid','pid','name'],[s.strip() for s in x.split(',',2)])) for x in lines if x.strip()]
    values=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    row=next(dict(zip(['index','uuid','name','memory','total'],[s.strip() for s in x.split(',')])) for x in values if int(x.split(',')[0])==index)
    # ToDesk is the user's existing desktop session, not a training job. Never terminate it.
    row['processes']=[x for x in processes if x['uuid']==row['uuid']]
    row['foreign_compute']=[x for x in row['processes'] if x['name']!='/opt/todesk/bin/ToDesk_Session']
    row['available']=not row['foreign_compute'] and int(row['memory'])<4096 and int(row['total'])>=20000
    return row

def collect(dest,delays,complete=False):
    rows=[];audits=[];settings=[];cache_sha=sha(dest/'cache/request_state.pt')
    for delay in delays:
        for mode in MODES:
            arm=dest/f'delay_{delay:g}_{mode}'
            end=json.loads((arm/'complete.json').read_text());batch=json.loads((arm/'eval_trials.json').read_text())
            audit=json.loads((arm/'state_audit.json').read_text());replay=json.loads((arm/'prefix_replay_audit.json').read_text())
            if end['trials']!=N or len(batch)!=N or len(audit)!=1 or {r['slot'] for r in batch}!=set(range(N)):
                raise RuntimeError(f'Incomplete arm {arm}')
            if any(r['mode']!=mode or r['delay_s']!=delay or r['request_cache_sha256']!=cache_sha or r['request_restore_max_abs_error']!=0. for r in batch):
                raise RuntimeError(f'Mismatched request or arm {arm}')
            if replay['max_abs_error']!=0. or audit[0]['max_abs_error']!=0.:raise RuntimeError(f'Physical replay failed: {arm}')
            rows+=batch;audits+=audit;settings.append(json.loads((arm/'render_settings.json').read_text()))
    if any(s!=settings[0] for s in settings):raise RuntimeError('Render settings differ between arms')
    for slot in range(N):
        if len({r['eligible'] for r in rows if r['slot']==slot})!=1:raise RuntimeError('Eligibility differs between arms')
    zero=[r for r in rows if r['delay_s']==0.]
    gates={mode:baseline_gate([r for r in zero if r['mode']==mode]) for mode in MODES}
    rates=[g['success_rate'] for g in gates.values()]
    spread=max(rates)-min(rates) if all(r is not None for r in rates) else None
    passed=all(g['passed'] for g in gates.values()) and spread is not None and spread<=MAX_ZERO_SPREAD
    traces=sorted({v['trajectory_sha256'] for v in audits if v['delay_s']==0.})
    negative=dict(passed=passed,arms=gates,success_rate_spread=spread,maximum_allowed_spread=MAX_ZERO_SPREAD,
        distinct_trajectory_hashes=len(traces),full_trajectory_sha256=traces,
        success_disagreement_slots=[slot for slot in range(N) if len({r['success'] for r in zero if r['slot']==slot})>1],
        interpretation='Original-render repeated identical controls; pixel/trajectory equality is not required. Mechanical prefix equality is required.')
    save(dest/'negative_control.json',negative)
    if not passed:raise RuntimeError(f'Zero-delay baseline performance or repeat-variation gate failed: {dest}')
    if complete:
        save(dest/'eval_trials.json',rows);save(dest/'state_audit.json',audits)
        save(dest/'complete.json',dict(status='completed',trials=len(rows),zero_delay_performance_passed=True,
            zero_delay_consistent=len(traces)==1,eligible_request_states=gates['continue']['eligible'],
            simulator_process='fresh per arm, original renderer, exact action-prefix replay'))
    return negative

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--wait-hours',type=float,default=1.);args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    owner=(out/'runner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'protocol.json').exists():raise RuntimeError('Fresh output required; failed runs must be reviewed, not silently retried')
    sources=[f for base in ['scripts/latency_counterfactual','scripts/agentic','source'] for f in (ROOT/base).rglob('*.py')]
    config=[f for base in ['assets','configs'] for f in (ROOT/base).rglob('*') if f.is_file() and f.suffix in ['.py','.usd','.usda','.usdc','.json','.yaml','.yml']]
    weights=[ROOT/'outputs/latency_controller_v1/train_v2_20261015/controller_v1_final.pt',
             ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',
             ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt']
    frozen={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(sources+config+weights))}
    protocol=dict(version='original_waiting_v4',seeds=SEEDS,num_envs=N,delays=DELAYS,modes=MODES,resume_steps=300,
        total_nominal_trials=len(SEEDS)*N*len(DELAYS)*len(MODES),nominal_request_states=len(SEEDS)*N,
        rendering='original default; no lighting, preset, dithering, AA or clock changes',
        matching='fresh process; exact prefix actions and request mechanics; cached request observations; fresh subsequent RGB-D',
        zero_control='all four modes identical at zero delay; record nuisance variation instead of enforcing trajectory hashes',
        baseline_gate=dict(minimum_eligible_per_seed=MIN_ELIGIBLE,minimum_success_rate_per_arm=MIN_BASELINE_RATE,maximum_zero_success_rate_spread=MAX_ZERO_SPREAD,
            failure_action='stop entire batch and retain results; no seed exclusion or silent retry',threshold_role='engineering regression checks, not statistical significance'),
        phases=['both seeds: cache and continue zero-delay performance','both seeds: remaining zero-delay repeats','delayed arms'],
        success='orientation error <= 0.16 rad for six consecutive post-response control steps, no prior drop',
        secondary='post-response single-step reach',source_weight_sha256=frozen,
        seed_selection='first two previously planned seeds; seed81001 was used for renderer diagnosis, so these are not held-out validation seeds',
        inference='known canonical goal and fixed simulated delay; no live Astra or VLM calls',
        evidence='two-seed diagnostic only; do not claim significance or pool with ambient v3 or live v7')
    save(out/'protocol.json',protocol)
    with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
        for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
    state=dict(pid=os.getpid(),status='preparing',planned_formal_trials=protocol['total_nominal_trials'],completed_formal_trials=0,
        completed_formal_arms=0,seeds=SEEDS,gpu=args.gpu,baseline_passed_seeds=[],zero_repeat_passed_seeds=[],fully_validated_seeds=[])
    def update(**fields):
        state.update(fields,updated=time.time());save(out/'progress.json',state)
    child=None;gpu_lock=None
    def interrupt(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        stable=0;deadline=time.monotonic()+args.wait_hours*3600
        while gpu_lock is None:
            snapshot=gpu_available(args.gpu);stable=stable+1 if snapshot['available'] else 0
            update(status='waiting_for_available_gpu',gpu_snapshot=snapshot)
            if stable>=3:
                handle=Path('/tmp',f'agent_inhand_paired_gpu_{args.gpu}.lock').open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_lock=handle
                except BlockingIOError:handle.close()
            if gpu_lock is not None:break
            if time.monotonic()>=deadline:raise TimeoutError('GPU unavailable within finite wait')
            time.sleep(5)
        (out/'tmp').mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(args.gpu),
            PAIRED_RENDER_GPU=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
        env['PYTHONPATH']=':'.join(str(ROOT/x) for x in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
        def run_arm(seed,delay=None,mode=None):
            nonlocal child
            parent=out/f'seed_{seed}';parent.mkdir(exist_ok=True);dest=parent/('cache' if mode is None else f'delay_{delay:g}_{mode}')
            if (dest/'complete.json').exists():return
            if dest.exists():raise RuntimeError(f'Incomplete arm requires review: {dest}')
            if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Frozen source, asset or weights changed')
            cmd=[sys.executable,str(ROOT/'scripts/latency_counterfactual/evaluate.py'),'--protocol-name','original_waiting_v4',
                 '--headless','--device','cuda:0','--output',str(dest),'--seed',str(seed),'--num-envs',str(N),'--resume-steps','300',
                 '--replay-prefix','--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
            if mode is None:cmd+=['--cache-only']
            else:
                cmd+=['--request-cache',str(parent/'cache/request_state.pt'),'--delays',str(delay),'--modes',mode]
                if seed==SEEDS[0] and delay in [0.,6.,20.]:cmd+=['--record-slots','0','5','--record-delays',str(delay)]
            save(parent/(dest.name+'_command.json'),cmd)
            with (parent/(dest.name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                begin=time.monotonic();update(current_seed=seed,current_mode=mode,current_delay_s=delay,child_pid=child.pid)
                while child.poll() is None:
                    update(current_arm_wall_s=round(time.monotonic()-begin,1))
                    if time.monotonic()-begin>1200:raise TimeoutError(f'20-minute watchdog: {dest}')
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{dest.name} exited {child.returncode}; inspect log')
            if not (dest/'complete.json').exists():raise RuntimeError(f'Missing completion: {dest}')
            manifest=json.loads((dest/'manifest.json').read_text())
            if manifest['render_profile']!='original default renderer':raise RuntimeError('Wrong render profile')
            if mode is not None:
                audit=json.loads((dest/'prefix_replay_audit.json').read_text())
                if audit['max_abs_error']!=0.:raise RuntimeError('Prefix physical replay differs')
            arms=list(out.glob('seed_*/delay_*/complete.json'))
            update(completed_formal_arms=len(arms),completed_formal_trials=len(arms)*N,child_pid=None)
        update(status='checking_baselines')
        for seed in SEEDS:
            run_arm(seed);run_arm(seed,0.,'continue')
            rows=json.loads((out/f'seed_{seed}'/'delay_0_continue/eval_trials.json').read_text())
            gate=baseline_gate(rows);save(out/f'seed_{seed}'/'baseline_gate.json',gate)
            if not gate['passed']:raise RuntimeError(f'Zero-delay baseline performance failed on seed {seed}; no delayed arms started')
            update(baseline_passed_seeds=state['baseline_passed_seeds']+[seed])
        save(out/'baseline_checks.json',{str(s):json.loads((out/f'seed_{s}'/'baseline_gate.json').read_text()) for s in SEEDS})
        update(status='checking_zero_delay_repeats')
        for seed in SEEDS:
            for mode in MODES[1:]:run_arm(seed,0.,mode)
            collect(out/f'seed_{seed}',[0.]);update(zero_repeat_passed_seeds=state['zero_repeat_passed_seeds']+[seed])
        save(out/'launch_validation.json',dict(baseline_checks='passed',zero_repeat_checks='passed',seeds=SEEDS,updated=time.time()))
        update(status='evaluating_delays')
        for seed in SEEDS:
            for i,delay in enumerate(DELAYS[1:]):
                shift=(seed+i)%len(MODES)
                for mode in MODES[shift:]+MODES[:shift]:run_arm(seed,delay,mode)
            collect(out/f'seed_{seed}',DELAYS,complete=True);summary=analyze(out)
            update(fully_validated_seeds=summary['completed_seeds'])
        update(status='completed',current_seed=None,current_mode=None,current_delay_s=None,child_pid=None)
    except BaseException as error:
        update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if gpu_lock:gpu_lock.close()

if __name__=='__main__':main()
