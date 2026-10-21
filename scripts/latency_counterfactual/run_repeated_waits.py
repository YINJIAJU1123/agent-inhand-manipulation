"""Two additional matched-state repetitions at zero, 6 and 20 seconds.

This is a finite exploratory evaluation with original rendering, frozen
weights, counterbalanced execution order and retained failures. Repeated
runs measure variability; they do not create new independent request states.
"""
import argparse,fcntl,json,os,random,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
from run_original import ROOT,SEEDS,MODES,N,save,sha,gpu_available
from analyze import stats

DELAYS=[0.,6.,20.]
REPEATS=[1,2]
SCHEDULE_SEED=181026

def make_schedule():
    rng=random.Random(SCHEDULE_SEED);groups=[]
    for delay in DELAYS:
        for seed in SEEDS:
            order=MODES.copy();rng.shuffle(order)
            for repeat in REPEATS:
                modes=order if repeat==1 else list(reversed(order))
                groups.append(dict(seed=seed,repeat=repeat,delay_s=delay,modes=modes))
    # Zero-delay repeats precede delays. Within each phase shuffle group order;
    # opposite repeat orders balance each method's serial position per state/delay.
    zero=[g for g in groups if g['delay_s']==0.];positive=[g for g in groups if g['delay_s']>0.]
    rng.shuffle(zero);rng.shuffle(positive)
    return zero+positive

def arm_path(out,group,mode):
    return out/f"seed_{group['seed']}"/f"repeat_{group['repeat']}"/f"delay_{group['delay_s']:g}_{mode}"

def detailed_stats(rows):
    result=stats(rows);valid=[r for r in rows if r['eligible']]
    survivors=sum(not r['wait_drop'] for r in valid)
    result['recovery_drop_given_wait_survival']=result['recovery_drops']/survivors if survivors else None
    return result

def summarize(out,protocol):
    rows=[];completed=[];zero=[]
    for group in protocol['schedule']:
        paths=[arm_path(out,group,m) for m in MODES]
        if not all((p/'complete.json').exists() for p in paths):continue
        block=[]
        for mode,p in zip(MODES,paths):
            values=json.loads((p/'eval_trials.json').read_text())
            block.extend([dict(r,repeat=group['repeat']) for r in values])
        rows+=block;completed.append(group)
        if group['delay_s']==0.:
            counts={m:detailed_stats([r for r in block if r['mode']==m]) for m in MODES}
            rates=[v['success_rate'] for v in counts.values()]
            zero.append(dict(seed=group['seed'],repeat=group['repeat'],arms=counts,
                success_rate_spread=max(rates)-min(rates),
                interpretation='Identical control law; method labels are repetition labels at zero delay.'))
    positive=[r for r in rows if r['delay_s']>0.]
    reference=Path(protocol['reference_output']);old=[]
    for seed in SEEDS:
        old.extend(r for r in json.loads((reference/f'seed_{seed}'/'eval_trials.json').read_text()) if r['delay_s'] in DELAYS)
    result=dict(status='completed_diagnostic' if len(completed)==len(protocol['schedule']) else 'partial_diagnostic',
        completed_balanced_groups=len(completed),planned_balanced_groups=len(protocol['schedule']),
        completed_balanced_trials=len(rows),completed_groups=completed,
        nominal_unique_request_states=24,eligible_unique_request_states=22,
        fresh_positive_delay_results={m:detailed_stats([r for r in positive if r['mode']==m]) for m in MODES},
        fresh_by_delay={str(d):{m:detailed_stats([r for r in rows if r['mode']==m and r['delay_s']==d]) for m in MODES} for d in DELAYS},
        by_seed_repeat_delay=[dict(seed=g['seed'],repeat=g['repeat'],delay_s=g['delay_s'],
            arms={m:detailed_stats([r for r in rows if r['seed']==g['seed'] and r['repeat']==g['repeat'] and r['delay_s']==g['delay_s'] and r['mode']==m]) for m in MODES}) for g in completed],
        zero_delay_repeat_audits=zero,
        previous_same_delay_results={str(d):{m:detailed_stats([r for r in old if r['mode']==m and r['delay_s']==d]) for m in MODES} for d in DELAYS},
        limitations=[
            'Exploratory repetition of the same two seeds and 22 eligible cached request states; repeats are not independent initial states.',
            'Focus on 6s and 20s waiting; do not extrapolate to the old mixture of short and long delays.',
            'Original-render repeated zero-delay controls showed large variation previously; report current repeat spread without attributing label differences to a policy.',
            'Fresh results and previous results remain separate; do not select favorable runs or pool an unbalanced delay mixture.',
            'Known canonical goal, fixed simulated delay and no live Astra/VLM calls.',
            'Primary success is six consecutive post-response steps within 0.16rad without prior drop, with 300 recovery steps.',
            'No significance test or bootstrap interval is reported for two reset seeds.'])
    save(out/'summary.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reference',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    out=Path(a.output).resolve();reference=Path(a.reference).resolve();out.mkdir(parents=True,exist_ok=True)
    owner=(out/'runner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'protocol.json').exists():raise RuntimeError('Fresh output required; no automatic retries or seed exclusions')
    base=json.loads((reference/'protocol.json').read_text());assert base['seeds']==SEEDS
    frozen=base['source_weight_sha256']
    if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Reference frozen source/assets/weights changed')
    frozen=dict(frozen,**{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    # Pin the previously observed request states so a new prefix cannot select an easier population.
    cache_meta={};expected_render={}
    for seed in SEEDS:
        dest=out/'caches'/f'seed_{seed}';dest.mkdir(parents=True)
        original=reference/f'seed_{seed}'/'cache'
        for name in ['request_state.pt','request.json','manifest.json','render_settings.json']:
            shutil.copy2(original/name,dest/name)
        request=json.loads((dest/'request.json').read_text())
        cache_meta[str(seed)]=dict(path=str(dest/'request_state.pt'),sha256=sha(dest/'request_state.pt'),eligible=request['eligible'])
        assert cache_meta[str(seed)]['sha256']==request['cache_sha256']
        expected_render[str(seed)]=json.loads((original/'render_settings.json').read_text())
    schedule=make_schedule()
    protocol=dict(version='original_waiting_repeats_v5',reference_output=str(reference),reference_protocol_sha256=sha(reference/'protocol.json'),
        seeds=SEEDS,repeats=REPEATS,delays=DELAYS,modes=MODES,num_envs=N,total_nominal_trials=len(schedule)*4*N,
        independent_request_states=24,eligible_request_states=22,cache_meta=cache_meta,schedule=schedule,schedule_seed=SCHEDULE_SEED,
        rendering='original default; no overrides',source_weight_sha256=frozen,expected_render_settings=expected_render,
        order='Randomized groups within zero/positive phases; repeat 2 reverses repeat 1 mode order for each seed/delay.',
        selection='Same two seeds and exact cached physical request states as the completed diagnostic; no new seeds or cherry-picked retries.',
        primary_metrics=['wait_drop','resume_drop','six-step post-response success'],secondary_metrics=['single-step reach','recovery time','mode use','handoff state'],
        interpretation='Frozen-policy exploratory replication, no training and no live Astra/VLM calls.',
        hard_stops=['changed frozen files or caches','nonzero mechanical prefix error','wrong renderer','incomplete or mislabelled trials','runtime failure or watchdog','all zero-delay repeats fail for a seed'],
        warnings='Outcome spread and low success rates are retained diagnostic findings; no post-hoc 50% gate or relabelled pass.',
        created=time.time())
    save(out/'protocol.json',protocol)
    with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
        for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
    state=dict(pid=os.getpid(),status='preparing',planned_trials=protocol['total_nominal_trials'],completed_trials=0,completed_arms=0,
        seeds=SEEDS,repeats=REPEATS,delays=DELAYS,gpu=a.gpu,diagnostic_only=True)
    def update(**fields):state.update(fields,updated=time.time());save(out/'progress.json',state)
    child=None;gpu_lock=None
    def interrupted(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        stable=0;deadline=time.monotonic()+3600
        while gpu_lock is None:
            snapshot=gpu_available(a.gpu);stable=stable+1 if snapshot['available'] else 0;update(status='waiting_for_available_gpu',gpu_snapshot=snapshot)
            if stable>=3:
                handle=Path('/tmp',f'agent_inhand_paired_gpu_{a.gpu}.lock').open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_lock=handle
                except BlockingIOError:handle.close()
            if gpu_lock:break
            if time.monotonic()>=deadline:raise TimeoutError('GPU unavailable within finite wait')
            time.sleep(5)
        (out/'tmp').mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu),
            PAIRED_RENDER_GPU=str(a.gpu),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
        env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
        zero_checked=False
        for group in schedule:
            if group['delay_s']>0. and not zero_checked:
                summary=summarize(out,protocol)
                for seed in SEEDS:
                    audits=[v for v in summary['zero_delay_repeat_audits'] if v['seed']==seed]
                    assert len(audits)==2
                    if sum(x['successes'] for v in audits for x in v['arms'].values())==0:raise RuntimeError(f'All zero-delay repeats fail for {seed}')
                save(out/'zero_delay_review.json',dict(status='audited_for_diagnostic',repeats=summary['zero_delay_repeat_audits'],statistical_validation=False))
                zero_checked=True
            seed=group['seed'];delay=group['delay_s'];cache=cache_meta[str(seed)]
            update(status='zero_delay_repeats' if delay==0. else 'waiting_policy_repeats',current_seed=seed,current_repeat=group['repeat'],current_delay_s=delay)
            for position,mode in enumerate(group['modes']):
                if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Frozen code/assets/weights changed')
                if sha(Path(cache['path']))!=cache['sha256']:raise RuntimeError('Request cache changed')
                dest=arm_path(out,group,mode)
                if dest.exists():raise RuntimeError(f'Existing arm requires review; no automatic retry: {dest}')
                dest.parent.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,str(ROOT/'scripts/latency_counterfactual/evaluate.py'),'--protocol-name','original_waiting_v4',
                    '--headless','--device','cuda:0','--output',str(dest),'--seed',str(seed),'--num-envs',str(N),'--resume-steps','300',
                    '--replay-prefix','--request-cache',cache['path'],'--delays',str(delay),'--modes',mode,
                    '--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
                if group['repeat']==1 and delay>0.:cmd+=['--record-slots','0','5','--record-delays',str(delay)]
                save(dest.parent/(dest.name+'_command.json'),cmd)
                with (dest.parent/(dest.name+'.log')).open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    begin=time.monotonic();update(current_mode=mode,mode_order_position=position,child_pid=child.pid)
                    while child.poll() is None:
                        update(current_arm_wall_s=round(time.monotonic()-begin,1))
                        if time.monotonic()-begin>1200:raise TimeoutError(f'20-minute watchdog: {dest}')
                        time.sleep(5)
                    if child.returncode:raise RuntimeError(f'{dest} exited {child.returncode}; inspect retained log')
                complete=json.loads((dest/'complete.json').read_text());rows=json.loads((dest/'eval_trials.json').read_text())
                manifest=json.loads((dest/'manifest.json').read_text());replay=json.loads((dest/'prefix_replay_audit.json').read_text())
                audit=json.loads((dest/'state_audit.json').read_text())
                if complete['trials']!=N or len(rows)!=N or {r['slot'] for r in rows}!=set(range(N)):raise RuntimeError('Incomplete result')
                if manifest['render_profile']!='original default renderer' or json.loads((dest/'render_settings.json').read_text())!=expected_render[str(seed)]:raise RuntimeError('Render configuration changed')
                if replay['max_abs_error']!=0. or len(audit)!=1 or audit[0]['max_abs_error']!=0.:raise RuntimeError('Physical replay differs')
                for r in rows:
                    if r['seed']!=seed or r['mode']!=mode or r['delay_s']!=delay or r['request_cache_sha256']!=cache['sha256'] or r['request_restore_max_abs_error']!=0. or r['eligible']!=cache['eligible'][r['slot']]:
                        raise RuntimeError('Mismatched trial or request identity')
                update(completed_arms=state['completed_arms']+1,completed_trials=state['completed_trials']+N,child_pid=None)
            summary=summarize(out,protocol);update(completed_balanced_groups=summary['completed_balanced_groups'])
        summary=summarize(out,protocol);assert summary['completed_balanced_trials']==576
        update(status='completed_diagnostic',current_seed=None,current_repeat=None,current_delay_s=None,current_mode=None,child_pid=None)
    except BaseException as error:
        update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if gpu_lock:gpu_lock.close()

if __name__=='__main__':main()
