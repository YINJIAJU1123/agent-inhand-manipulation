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
from analysis import analyze_learning as analyze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for d in ['logs','tmp','requests']:(out/d).mkdir()
    state=dict(pid=os.getpid(),status='waiting_gpu',planned_episodes=192,completed_episodes=0)
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');child=service=None;keyfile=out/'bridge.key'
    frozen=read(ROOT/'outputs/recovery_factorial/ablation_20261019/protocol.json')['source_asset_weights_sha256'].copy()
    for p in [*Path(__file__).parent.glob('*.py'),*list((ROOT/'scripts/astra_runtime_v2').glob('*.py')),*list((ROOT/'scripts/astra_transport_v2').glob('*.py')),*list((ROOT/'scripts/astra_learning_v1').glob('*.py')),V3,NEW,*list((ROOT/'scripts/agentic').glob('*.py'))]:frozen[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Dependencies changed before freeze')
    remaining_calls=1152;remaining_tokens=3000000
    rng=random.Random(191021);schedule=[]
    for repeat in [1,2]:
        for seed in [74101,74102]:
            for template in ['canonical','paraphrase']:
                arms=['continue','slow','v3','learned'];rng.shuffle(arms)
                for arm in arms:schedule.append(dict(repeat=repeat,seed=seed,template=template,arm=arm))
    protocol=dict(version='astra_repeated_eval_v1',seeds=[74101,74102],repeats=2,templates=['canonical','paraphrase'],arms=['continue','slow','v3','learned'],
        total_formal_episodes=192,episodes_per_arm=48,max_steps=3600,calls_per_episode=6,max_api_attempts=remaining_calls,token_stop_threshold=remaining_tokens,max_transport_attempts=2,
        schedule=schedule,source_asset_weight_sha256=frozen,checkpoint_sha256={'v3':sha(V3),'learned':sha(NEW)},
        evidence='User authorized more Astra evaluation. Two repeated rounds on the existing two seeds, not new independent scenes. All four arms use the same repaired runtime and transport settings. Original mixed-protocol pilot stays separate.',
        physics='Actual asynchronous calls; physics continues with measured-latency-aligned response delivery.',
        failures='Keep every completed episode, including API errors. No silent retries of episodes, no success-based stopping. Stop on a runtime failure, authorization/budget error, or three API-error episodes within one batch.',
        smoke='Prior live pilot completed all 96 cases under frozen weights; runtime fix passed CPU and simulator replay. No new smoke requested.',
        token_caveat='Reported-token stop threshold, not an exact bill or hard limit on already in-flight calls.')
    save(out/'protocol.json',protocol);progress();analyze(out)
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
        port=18797
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        key=secrets.token_urlsafe(32)
        with os.fdopen(os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(key)
        env.update(ASTRA_BRIDGE_URL=f'http://127.0.0.1:{port}',ASTRA_BRIDGE_KEY_FILE=str(keyfile))
        command=[PYTHON,str(ROOT/'scripts/astra_transport_v2/bridge.py'),'--port',str(port),'--key-file',str(keyfile),'--audit-dir',str(out/'requests'),'--workers','6','--max-calls',str(remaining_calls),'--max-tokens',str(remaining_tokens),'--max-sessions','200','--transport-attempts','2']
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
            smoke=job.get('smoke',False);name=f"eval_r{job['repeat']}_{job['seed']}_{job['template']}_{job['arm']}";dest=out/name
            if (dest/'report.json').exists():continue
            if service.poll() is not None:raise RuntimeError('Astra bridge stopped')
            command=[PYTHON,str(ROOT/'scripts/astra_runtime_v2/evaluate_learning.py'),'--headless','--device','cuda:0','--output',str(dest),'--seed',str(job['seed']),'--mode','astra','--waiting',job['arm'],'--template-set',job['template'],'--max-steps','3600','--budget','6','--controller',str(V3 if job['arm']=='v3' else NEW),
                '--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
            if job['repeat']==1 and job['seed']==74101:command+=['--record-video']
            save(out/'logs'/f'{name}_command.json',command)
            with (out/'logs'/f'{name}.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);started=time.monotonic()
                while child.poll() is None:
                    progress(status='smoke' if smoke else 'evaluating',current_job=name,child_pid=child.pid,job_wall_seconds=round(time.monotonic()-started,1))
                    if (dest/'runtime_failure.json').exists():raise RuntimeError(f'{name}: recorded runtime exception; see runtime_failure.json')
                    heartbeat=dest/'progress.json'
                    if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>300:raise TimeoutError(f'{name}: no physics progress for five minutes')
                    if time.monotonic()-started>3600:raise TimeoutError(name)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}; see log')
            report=read(dest/'report.json');rows=report['episodes'];events=[e for r in rows for e in r['async_events']]
            if len(rows)!=6 or any(r['calls']>6 for r in rows):raise RuntimeError('Episode count/call cap mismatch')
            failures=[e for e in events if e['status']=='error']
            allowed=('SSLError','ProxyError','ConnectionError','ReadTimeout','ConnectTimeout','Timeout','ChunkedEncodingError')
            if any(not (str(e.get('error','')).endswith(allowed) or 'http_520_api_error' in str(e.get('error',''))) for e in failures):raise RuntimeError('Unexpected inference error; completed report retained')
            if failures:save(dest/'infrastructure_errors.json',dict(events=failures,retained_in_total_denominator=True))
            if sum(any(e['status']=='error' for e in r['async_events']) for r in rows)>=3:raise RuntimeError('Network circuit breaker: at least three cases failed transport in one batch')
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
        try:progress(completed_episodes=analyze(out)['completed_episodes'])
        except Exception:pass
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        stop(child);stop(service);keyfile.unlink(missing_ok=True);lock.close()
if __name__=='__main__':main()
