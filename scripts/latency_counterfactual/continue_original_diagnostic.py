"""Complete the same two-seed grid after explicitly recording a baseline warning.

No failed trials are retried or excluded. Frozen evaluation code and weights
remain unchanged. The initial engineering gate is reported, never relabelled
as passed; this continuation yields an exploratory diagnostic only.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
from run_original import ROOT,SEEDS,MODES,DELAYS,N,save,sha,gpu_available,collect
from analyze_original_diagnostic import analyze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output).resolve();owner=(out/'runner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=json.loads((out/'protocol.json').read_text());prior=json.loads((out/'progress.json').read_text())
    if prior['status']!='failed' or prior.get('error')!='Zero-delay baseline performance failed on seed 81002; no delayed arms started':
        raise RuntimeError('This bounded continuation applies only to the reviewed baseline warning')
    assert original['seeds']==SEEDS and original['total_nominal_trials']==576
    frozen=original['source_weight_sha256']
    if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Original evaluator, assets or weights changed')
    save(out/'progress_before_diagnostic_continuation.json',prior)
    extra={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/latency_counterfactual/analyze_original_diagnostic.py']}
    amendment=dict(created=time.time(),reason='Seed81002 sustained baseline is 4/10, below initial 50% engineering threshold; nonzero task performance permits exploratory diagnosis.',
        original_protocol_sha256=sha(out/'protocol.json'),original_gate_warning=prior['error'],
        modification='Complete the already planned two-seed diagnostic grid while retaining failed performance/repeat-spread gates as warnings. Do not claim passed validation.',
        unchanged=['two seeds 81001/81002','576 total nominal rollouts','all existing trials retained','no reruns or exclusions','original rendering','evaluation code and weights','six-step primary success'],
        hard_stops=['incomplete or mismatched trials','nonzero physical replay error','changed source/assets/weights/render settings','fewer than six eligible requests per seed','zero successes across all four zero-delay repeats for a seed','runtime error or watchdog'],
        statistical_claims=False,extra_sources=extra)
    if (out/'diagnostic_amendment.json').exists():raise RuntimeError('Continuation already attempted; inspect its status')
    save(out/'diagnostic_amendment.json',amendment)
    with tarfile.open(out/'diagnostic_continuation_sources.tar.gz','w:gz') as tar:
        for name in extra:tar.add(ROOT/name,arcname=name,recursive=False)
    state=dict(prior,pid=os.getpid(),status='starting_diagnostic_continuation',error=None,
        diagnostic_only=True,engineering_baseline_warning_seeds=[81002],completed_diagnostic_seeds=[])
    def update(**fields):state.update(fields,updated=time.time());save(out/'progress.json',state)
    child=None;gpu_lock=None
    def interrupt(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def check_sources():
        if any(sha(ROOT/k)!=v for k,v in {**frozen,**extra}.items()):raise RuntimeError('Frozen source/assets/weights changed')
    try:
        deadline=time.monotonic()+3600;stable=0;gpu=prior['gpu']
        while gpu_lock is None:
            snapshot=gpu_available(gpu);stable=stable+1 if snapshot['available'] else 0
            update(status='waiting_for_available_gpu',gpu_snapshot=snapshot)
            if stable>=3:
                handle=Path('/tmp',f'agent_inhand_paired_gpu_{gpu}.lock').open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_lock=handle
                except BlockingIOError:handle.close()
            if gpu_lock:break
            if time.monotonic()>=deadline:raise TimeoutError('GPU unavailable within finite wait')
            time.sleep(5)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
            PAIRED_RENDER_GPU=str(gpu),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
        env['PYTHONPATH']=':'.join(str(ROOT/x) for x in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
        def run_pending(seed,delay,mode):
            nonlocal child
            check_sources();parent=out/f'seed_{seed}';dest=parent/f'delay_{delay:g}_{mode}'
            if (dest/'complete.json').exists():return
            if dest.exists():raise RuntimeError(f'Incomplete arm requires review; no retry: {dest}')
            cmd=[sys.executable,str(ROOT/'scripts/latency_counterfactual/evaluate.py'),'--protocol-name','original_waiting_v4',
                 '--headless','--device','cuda:0','--output',str(dest),'--seed',str(seed),'--num-envs',str(N),'--resume-steps','300',
                 '--replay-prefix','--request-cache',str(parent/'cache/request_state.pt'),'--delays',str(delay),'--modes',mode,
                 '--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
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
            if not (dest/'complete.json').exists():raise RuntimeError('Missing completion record')
            if json.loads((dest/'manifest.json').read_text())['render_profile']!='original default renderer':raise RuntimeError('Wrong renderer')
            if json.loads((dest/'prefix_replay_audit.json').read_text())['max_abs_error']!=0.:raise RuntimeError('Physical replay differs')
            arms=list(out.glob('seed_*/delay_*/complete.json'));update(completed_formal_arms=len(arms),completed_formal_trials=len(arms)*N,child_pid=None)
        def audited_collect(seed,delays,complete=False):
            dest=out/f'seed_{seed}'
            try:negative=collect(dest,delays,complete=False)
            except RuntimeError as error:
                if not str(error).startswith('Zero-delay baseline performance or repeat-variation gate failed:'):raise
                negative=json.loads((dest/'negative_control.json').read_text())
            if any(g['eligible']<6 for g in negative['arms'].values()):raise RuntimeError('Too few eligible requests')
            if sum(g['successes'] for g in negative['arms'].values())==0:raise RuntimeError('All zero-delay repetitions fail; stop diagnostic')
            save(dest/'diagnostic_gate.json',dict(physical_pairing_passed=True,zero_delay_audited=True,
                original_engineering_gate_passed=negative['passed'],diagnostic_only=True))
            if complete:
                rows=[];audits=[]
                for delay in delays:
                    for mode in MODES:
                        arm=dest/f'delay_{delay:g}_{mode}'
                        rows+=json.loads((arm/'eval_trials.json').read_text());audits+=json.loads((arm/'state_audit.json').read_text())
                save(dest/'eval_trials.json',rows);save(dest/'state_audit.json',audits)
                save(dest/'complete.json',dict(status='completed_diagnostic',trials=len(rows),zero_delay_audited=True,
                    zero_delay_performance_passed=negative['passed'],zero_delay_consistent=negative['distinct_trajectory_hashes']==1,
                    eligible_request_states=negative['arms']['continue']['eligible']))
            return negative
        update(status='diagnostic_zero_repeats')
        for seed in SEEDS:
            for mode in MODES:run_pending(seed,0.,mode)
            audited_collect(seed,[0.])
        save(out/'launch_validation.json',dict(status='diagnostic_continuation',mechanical_pairing='passed',original_engineering_gate='see retained per-seed warnings',
            seeds=SEEDS,updated=time.time(),diagnostic_only=True))
        update(status='diagnostic_delays')
        for seed in SEEDS:
            for index,delay in enumerate(DELAYS[1:]):
                shift=(seed+index)%len(MODES)
                for mode in MODES[shift:]+MODES[:shift]:run_pending(seed,delay,mode)
            audited_collect(seed,DELAYS,complete=True);summary=analyze(out);update(completed_diagnostic_seeds=summary['completed_seeds'])
        update(status='completed_diagnostic',current_seed=None,current_mode=None,current_delay_s=None,child_pid=None)
    except BaseException as error:
        update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if gpu_lock:gpu_lock.close()

if __name__=='__main__':main()
