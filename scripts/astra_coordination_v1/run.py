"""Finite Astra response-coordination diagnostic, queued behind existing work."""
import argparse,fcntl,hashlib,json,os,random,secrets,signal,socket,subprocess,sys,tarfile,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/latency_counterfactual'),str(ROOT/'scripts/agentic')]
from run_original import gpu_available
PYTHON='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
V3=ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt'
NEW=ROOT/'outputs/astra_learning_v1/train_20261019/controller_final.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text())
def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
from analyze import analyze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for d in ['logs','tmp','requests']:(out/d).mkdir()
    state=dict(pid=os.getpid(),status='waiting_gpu',planned_episodes=48,completed_episodes=0)
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');child=service=None;keyfile=out/'bridge.key'
    frozen=read(ROOT/'outputs/recovery_factorial/ablation_20261019/protocol.json')['source_asset_weights_sha256'].copy()
    for p in [*Path(__file__).parent.glob('*.py'),*list((ROOT/'scripts/astra_learning_v1').glob('*.py')),V3,NEW,*list((ROOT/'scripts/agentic').glob('*.py'))]:frozen[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Dependencies changed before freeze')
    rng=random.Random(191021);schedule=[]
    for seed in [75201,75202]:
        combinations=[(h,d) for h in ['direct','learned'] for d in [0,6]];rng.shuffle(combinations)
        for handoff,delay in combinations:schedule.append(dict(seed=seed,handoff=handoff,delay=delay))
    protocol=dict(version='live_astra_coordination_diagnostic_v1',seeds=[75201,75202],template='canonical',handoff=['direct','learned'],extra_delay_s=[0,6],
        total_formal_episodes=48,episodes_per_cell=12,smoke_episodes=6,smoke_seed=75199,max_steps=3600,calls_per_episode=6,max_api_attempts=324,token_stop_threshold=900000,
        token_caveat='Reported-token stop threshold checked before requests; in-flight calls can exceed it. Failed request usage may be unknown.',
        schedule=schedule,source_asset_weight_sha256=frozen,checkpoint_sha256=sha(NEW),
        hypothesis='Does state change during agent reasoning create a measurable response-handoff problem, and can the existing learned transition help?',
        intervention='Same fixed slow waiting in both arms. Direct frozen student versus learned two-second transition after each delivered reply. Additional six seconds of simulated transport delay on requests after a skill is known. Initial cold start unchanged.',
        evidence='Development diagnostic with two seeds, not a trained agent-level replanner or publication-level result. No automatic instruction rejection, replanning, or baseline promotion.',
        audit='Save sensor state pairs, request/response RGB frames, privileged physics drift for analysis only, and events following response. No privileged audit data enters agent or policy.',
        physics='Actual Astra calls. Physics continues; no waiting on Future. Extra delay is simulated response delivery delay, not claimed real API latency.',
        old_live_run='Existing 96-episode live pilot continues independently. Do not combine success rates.')
    save(out/'protocol.json',protocol);progress()
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
        port=18793
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        key=secrets.token_urlsafe(32)
        with os.fdopen(os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(key)
        env.update(ASTRA_BRIDGE_URL=f'http://127.0.0.1:{port}',ASTRA_BRIDGE_KEY_FILE=str(keyfile))
        command=[PYTHON,str(ROOT/'scripts/agentic/astra_bridge_v6.py'),'--port',str(port),'--key-file',str(keyfile),'--audit-dir',str(out/'requests'),'--workers','6','--max-calls','324','--max-tokens','900000','--max-sessions','60','--transport-attempts','1']
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
        jobs=[dict(seed=75199,handoff='learned',delay=6,smoke=True)]+schedule
        for job in jobs:
            if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Frozen dependency changed')
            smoke=job.get('smoke',False);name=f"{'smoke' if smoke else 'eval'}_{job['seed']}_{job['handoff']}_d{job['delay']}";dest=out/name
            if service.poll() is not None:raise RuntimeError('Astra bridge stopped')
            command=[PYTHON,str(ROOT/'scripts/astra_coordination_v1/evaluate.py'),'--headless','--device','cuda:0','--output',str(dest),'--seed',str(job['seed']),'--mode','astra','--waiting','slow','--handoff',job['handoff'],'--extra-delay-s',str(job['delay']),'--template-set','canonical','--max-steps','3600','--budget','6','--controller',str(NEW),
                '--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
            if smoke or job['seed']==75201:command+=['--record-video']
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
            audit_rows=read(dest/'response_audit.json')
            if len(audit_rows)!=sum(r['calls'] for r in rows):raise RuntimeError('Request audit count mismatch')
            for r in audit_rows:
                if r['status']=='delivered':
                    expected_extra=job['delay'] if r['previous_decision']['decision']['color'] is not None else 0
                    if r['async_event']['injected_delay_s']!=expected_extra:raise RuntimeError('Incorrect extra delay')
                    if r['response_step']<r['async_event']['earliest_delivery_step']:raise RuntimeError('Premature response delivery')
            if job['handoff']=='learned' and report['metadata']['controller_sha256']!=protocol['checkpoint_sha256']:raise RuntimeError('Wrong checkpoint')
            if smoke:
                trace=read(dest/'gate_trace.json')
                if not any(any(x['applied_recovery']) for x in trace):raise RuntimeError('Smoke never exercised learned control')
                audit=read(dest/'response_audit.json')
                if not any(r['status']=='delivered' and r['async_event']['injected_delay_s']==6 for r in audit):raise RuntimeError('Smoke did not exercise injected response delay')
                save(out/'smoke_validation.json',dict(passed=True,episodes=6,delivered_responses=len(delivered),excluded_from_formal=True))
            summary=analyze(out);progress(completed_episodes=summary['completed_episodes'],child_pid=None,last_completed=name)
        progress(status='completed',current_job=None,child_pid=None)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        stop(child);stop(service);keyfile.unlink(missing_ok=True);lock.close()
if __name__=='__main__':main()
