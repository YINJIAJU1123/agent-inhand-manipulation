"""Finite fresh-seed live Astra pilot. Waits for GPU; never replaces baselines."""
import argparse,fcntl,hashlib,json,os,random,secrets,signal,socket,subprocess,sys,tarfile,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/latency_counterfactual'),str(ROOT/'scripts/agentic')]
from run_original import gpu_available
PYTHON='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
V3=ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt'
NEW=ROOT/'outputs/astra_learning_v1/train_20261019/controller_final.pt'
PRIOR=ROOT/'outputs/astra_learning_v1/live_20261019'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text())
def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def analyze(out):
    arms={a:[] for a in ['continue','slow','v3','learned']};initial={};comparisons=0;maxerr=0.
    import numpy as np
    for folder in sorted(out.glob('eval_*')):
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json');m=report['metadata'];arms[m['waiting']]+=report['episodes']
        key=(m['seed'],m['template_set']);value=read(folder/'initial.json')
        if key in initial:
            for field in value:maxerr=max(maxerr,float(np.max(np.abs(np.asarray(value[field])-np.asarray(initial[key][field])))))
            comparisons+=1
        else:initial[key]=value
    if maxerr>1e-6:raise RuntimeError('Matched initial states differ')
    summary=dict(completed_episodes=sum(map(len,arms.values())),planned_episodes=96,initial_comparisons=comparisons,initial_max_abs_error=maxerr,smoke_excluded=True,
        evidence='Two new seeds, descriptive pilot. Actual Astra responses/latencies differ across arms. No significance claim.',arms={})
    for name,rows in arms.items():
        summary['arms'][name]=dict(episodes=len(rows),success=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows) if rows else None,drop=sum(r['drop'] for r in rows),wait_drop=sum(r['wait_drop'] for r in rows),false_stop=sum(r['false_stop'] for r in rows),calls=sum(r['calls'] for r in rows))
    paired={}
    for ref in ['continue','slow','v3']:
        rm={(r['seed'],r['template_set'],r['case']):r for r in arms[ref]};wins=losses=n=0
        for row in arms['learned']:
            key=(row['seed'],row['template_set'],row['case'])
            if key in rm:
                n+=1;wins+=int(row['success'] and not rm[key]['success']);losses+=int(rm[key]['success'] and not row['success'])
        paired[ref]=dict(matched=n,wins=wins,losses=losses)
    summary['paired_learned_vs']=paired
    save(out/'summary.json',summary);return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for d in ['logs','tmp','requests']:(out/d).mkdir()
    state=dict(pid=os.getpid(),status='waiting_gpu',planned_episodes=96,completed_episodes=0)
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');child=service=None;keyfile=out/'bridge.key'
    frozen=read(ROOT/'outputs/recovery_factorial/ablation_20261019/protocol.json')['source_asset_weights_sha256'].copy()
    for p in [*Path(__file__).parent.glob('*.py'),*list((ROOT/'scripts/astra_learning_v1').glob('*.py')),V3,NEW,*list((ROOT/'scripts/agentic').glob('*.py'))]:frozen[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Dependencies changed before freeze')
    prior=read(PRIOR/'protocol.json')
    for k,v in prior['source_asset_weight_sha256'].items():
        if sha(ROOT/k)!=v:raise RuntimeError('Original experiment dependency changed')
    old_calls=[read(p) for p in (PRIOR/'requests').glob('request_*.json')]
    remaining_calls=prior['max_api_attempts']-len(old_calls)
    used_tokens=sum((x.get('usage') or {}).get('total_tokens',0) for x in old_calls)
    remaining_tokens=prior['token_stop_threshold']-used_tokens
    if remaining_calls<=0 or remaining_tokens<=0:raise RuntimeError('Original experiment budget exhausted')
    accepted=[];excluded=[]
    for folder in sorted(PRIOR.glob('eval_*')):
        if not (folder/'report.json').exists():continue
        r=read(folder/'report.json');events=[e for row in r['episodes'] for e in row['async_events']]
        if len(r['episodes'])!=6 or any(e['status']=='error' for e in events):
            excluded.append(dict(path=str(folder),reason='Active inference transport error',replacement='Fresh matched batch retained alongside original failure, not a policy-based retry'))
            continue
        (out/folder.name).symlink_to(folder,target_is_directory=True);accepted.append(str(folder))
    if len(accepted)!=5 or len(excluded)!=1:raise RuntimeError('Unexpected prior completion state, requires review')
    (out/'prior_smoke_validation.json').write_text((PRIOR/'smoke_validation.json').read_text())
    rng=random.Random(191020);schedule=[]
    for seed in [74101,74102]:
        for template in ['canonical','paraphrase']:
            arms=['continue','slow','v3','learned'];rng.shuffle(arms)
            for arm in arms:schedule.append(dict(seed=seed,template=template,arm=arm))
    protocol=dict(version='live_astra_representation_pilot_v1',seeds=[74101,74102],templates=['canonical','paraphrase'],arms=['continue','slow','v3','learned'],
        episodes_per_arm=24,total_formal_episodes=96,smoke_episodes=6,smoke_seed=74099,max_steps=3600,calls_per_episode=6,max_api_attempts=612,token_stop_threshold=1500000,
        token_caveat='Reported-token stop threshold checked before request; concurrent in-flight calls can exceed it. Failed request usage may be unknown.',
        schedule=schedule,source_asset_weight_sha256=frozen,checkpoint_sha256={'v3':sha(V3),'learned':sha(NEW)},
        historical_freeze_sha256=sha(ROOT/'outputs/astra_learning_v1/baseline_freeze_20261019/manifest.json'),
        physics='No pause for inference. Actual asynchronous Astra calls and latency-aligned response delivery.',
        comparison='Continue and slow preserve the live baseline control/scoring logic. V3/new share response-aware runtime and two-second recovery window. New outcomes are diagnostics only, not sent to Astra.',
        evidence='Development pilot with two fresh seeds, not a large held-out benchmark. No automated baseline promotion. No silent retries.',
        old_controller_diagnostics='Known-goal phase crossover remains separate; it does not enter this live success rate.')
    protocol.update(resumes=str(PRIOR),inherited_valid_batches=accepted,excluded_infrastructure_batches=excluded,
        original_protocol_sha256=sha(PRIOR/'protocol.json'),original_max_api_attempts=prior['max_api_attempts'],original_token_stop_threshold=prior['token_stop_threshold'],
        previous_api_attempts=len(old_calls),previous_reported_tokens=used_tokens,max_api_attempts=remaining_calls,token_stop_threshold=remaining_tokens,
        resumption='Transport recovered after TLS EOF. Preserve five completed batches, replace one documented interrupted batch, then finish original schedule. No new seeds or learning changes.',
        smoke='Reuse successful original simulator smoke because sources, weights and simulator environment are unchanged. Fresh bridge health checked.')
    save(out/'protocol.json',protocol);progress(completed_episodes=30)
    analyze(out)
    with tarfile.open(out/'frozen_dependencies.tar.gz','w:gz') as t:
        for path in frozen:t.add(ROOT/path,arcname=path,recursive=False)
    try:
        deadline=time.monotonic()+14400
        while True:
            locked=False
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
            except BlockingIOError:pass
            if locked:
                gpu=gpu_available(0)
                if gpu['available']:break
                fcntl.flock(lock,fcntl.LOCK_UN)
            progress(status='waiting_gpu',reason='Existing experiment owns GPU lock or GPU is occupied')
            if time.monotonic()>deadline:raise TimeoutError('GPU queue expired after four hours')
            time.sleep(15)
        progress(status='starting_astra_bridge',gpu=gpu,reason=None)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',ASTRA_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
        env['PYTHONPATH']=':'.join(str(ROOT/x) for x in ['outputs/vlm_dependencies','outputs/learning_dependencies','outputs/video_dependencies','scripts/agentic'])
        port=18791
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        key=secrets.token_urlsafe(32)
        with os.fdopen(os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(key)
        env.update(ASTRA_BRIDGE_URL=f'http://127.0.0.1:{port}',ASTRA_BRIDGE_KEY_FILE=str(keyfile))
        command=[PYTHON,str(ROOT/'scripts/agentic/astra_bridge_v6.py'),'--port',str(port),'--key-file',str(keyfile),'--audit-dir',str(out/'requests'),'--workers','6','--max-calls',str(remaining_calls),'--max-tokens',str(remaining_tokens),'--max-sessions','110','--transport-attempts','1']
        with (out/'logs/bridge.log').open('w') as log:service=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+180
        while True:
            if service.poll() is not None:raise RuntimeError('Astra bridge exited; see logs/bridge.log')
            try:
                req=urllib.request.Request(env['ASTRA_BRIDGE_URL']+'/health',headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=3) as res:health=json.load(res)
                save(out/'bridge_health.json',health);break
            except Exception:
                if time.monotonic()>deadline:raise TimeoutError('Astra bridge health timeout')
                time.sleep(2)
        jobs=schedule
        for job in jobs:
            if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Frozen dependency changed')
            smoke=job.get('smoke',False);name=f"{'smoke' if smoke else 'eval'}_{job['seed']}_{job['template']}_{job['arm']}";dest=out/name
            if (dest/'report.json').exists():continue
            if service.poll() is not None:raise RuntimeError('Astra bridge stopped')
            command=[PYTHON,str(ROOT/'scripts/astra_learning_v1/evaluate.py'),'--headless','--device','cuda:0','--output',str(dest),'--seed',str(job['seed']),'--mode','astra','--waiting',job['arm'],'--template-set',job['template'],'--max-steps','3600','--budget','6','--controller',str(V3 if job['arm']=='v3' else NEW),
                '--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
            if smoke or job['seed']==74101:command+=['--record-video']
            save(out/'logs'/f'{name}_command.json',command)
            with (out/'logs'/f'{name}.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);started=time.monotonic()
                while child.poll() is None:
                    progress(status='smoke' if smoke else 'evaluating',current_job=name,child_pid=child.pid,job_wall_seconds=round(time.monotonic()-started,1))
                    if time.monotonic()-started>3600:raise TimeoutError(name)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}; see log')
            report=read(dest/'report.json');rows=report['episodes'];events=[e for r in rows for e in r['async_events']]
            if len(rows)!=6 or any(r['calls']>6 for r in rows):raise RuntimeError('Episode count/call cap mismatch')
            if any(e['status']=='error' for e in events):raise RuntimeError('Active inference error; batch retained, no retries')
            delivered=[e for e in events if e['status']=='delivered']
            if smoke and not delivered:raise RuntimeError('Smoke had no delivered Astra response')
            if any(e['physics_steps']<=0 for e in delivered):raise RuntimeError('Physics failed to advance during inference')
            if job['arm'] in ['v3','learned'] and report['metadata']['controller_sha256']!=protocol['checkpoint_sha256'][job['arm']]:raise RuntimeError('Wrong checkpoint')
            if smoke:
                trace=read(dest/'gate_trace.json')
                if not any(any(x['controller_active']) for x in trace):raise RuntimeError('Smoke never exercised learned control')
                save(out/'smoke_validation.json',dict(passed=True,episodes=6,delivered_responses=len(delivered),excluded_from_formal=True))
            summary=analyze(out);progress(completed_episodes=summary['completed_episodes'],child_pid=None,last_completed=name)
        progress(status='completed',current_job=None,child_pid=None)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        stop(child);stop(service);keyfile.unlink(missing_ok=True);lock.close()
if __name__=='__main__':main()
