"""Finite two-GPU evaluation, gated by real async smoke tests; no automation."""
import argparse,concurrent.futures,hashlib,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--astra-gpu',type=int,default=4);p.add_argument('--latent-gpu',type=int,default=5)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--validated-smoke');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    available=[int(row.split(',')[0]) for row in inventory.splitlines() if int(row.split(',')[1])<100 and int(row.split(',')[2])<10]
    preference=[4,5,3,6,7,0,1,2];available=[g for g in preference if g in available]
    if len(available)<2:raise RuntimeError('Two idle simulation GPUs required')
    a.astra_gpu,a.latent_gpu=available[:2]
    save(out/'gpu_assignment.json',dict(astra=a.astra_gpu,latent=a.latent_gpu,inventory=inventory))
    import fcntl
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'scripts/agentic').glob('*.py')}
    protocol=dict(episodes_per_arm=120,total_episodes=720,seeds=list(range(73001,73011)),templates=['canonical','paraphrase'],arms=['astra/continue','astra/slow','astra/learned','latent/continue','latent/slow','latent/learned'],max_steps=3600,call_budget=6,astra_max_calls=2160,astra_token_stop_threshold=6000000,smoke_excluded=True,source_sha256=sources)
    if (out/'protocol.json').exists():raise RuntimeError('Fresh evaluation directory required')
    save(out/'protocol.json',protocol)
    snap=out/'sources';snap.mkdir()
    for path in sources:(snap/Path(path).name).write_bytes((ROOT/path).read_bytes())
    stopped=threading.Event();children={};mutex=threading.Lock();progress={}
    signal.signal(signal.SIGTERM,lambda sig,frame:stopped.set())
    signal.signal(signal.SIGINT,lambda sig,frame:stopped.set())
    def update(mode,**fields):
        with mutex:
            progress.setdefault(mode,{}).update(fields);save(out/'progress.json',dict(updated=time.time(),workers=progress))
    def one(mode,waiting,seed,template,smoke=False):
        if stopped.is_set():raise RuntimeError('Peer worker failed')
        if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=value for path,value in sources.items()):raise RuntimeError('Source changed during frozen evaluation')
        gpu=a.astra_gpu if mode=='astra' else a.latent_gpu
        name=f'{"smoke" if smoke else "eval"}_{seed}_{template}_{mode}_{waiting}'
        dest=out/name
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),ASTRA_RENDER_GPU=str(gpu),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',TMPDIR='/home/brainco/tmp_latency_v1')
        cmd=[sys.executable,str(ROOT/'scripts/agentic/evaluate_async_v7.py'),'--headless','--device','cuda:0','--output',str(dest),'--seed',str(seed),'--mode',mode,'--waiting',waiting,'--template-set',template,'--max-steps','3600','--budget','6']
        if smoke or seed==73001:cmd.append('--record-video')
        update(mode,status='running',job=name)
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with mutex:children[mode]=child
            start=time.monotonic()
            while child.poll() is None:
                if stopped.is_set() or time.monotonic()-start>2400:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('peer failure or simulator watchdog: '+name)
                time.sleep(2)
            if child.returncode:raise RuntimeError('Simulator failed: '+name)
        report=json.loads((dest/'report.json').read_text());episodes=report['episodes']
        assert len(episodes)==6 and all(e['calls']<=6 for e in episodes)
        events=[event for e in episodes for event in e['async_events']]
        if any(e.get('error') for e in events):raise RuntimeError('Inference error: '+name)
        if smoke and not any(e['status']=='delivered' and e['physics_steps']>0 for e in events):raise RuntimeError('No verified asynchronous delivery: '+name)
        assert all(e['physics_steps']>0 for e in events if e['status']=='delivered')
        update(mode,last_completed=name,last_success=sum(e['success'] for e in episodes),last_drop=sum(e['drop'] for e in episodes))
        return dest
    def worker(mode):
        try:
            one(mode,'learned',72999,'canonical',True)
            update(mode,status='smoke_passed')
        except BaseException:
            stopped.set();update(mode,status='failed');raise
    if a.validated_smoke:
        critical=['evaluate_async_v7.py','async_agent_v7.py','latency_controller_v1.py','latent_service_v7.py','latent_goal_v4.py','latent_supervisor_v4.py','astra_agent_v4.py','astra_bridge_v6.py','student_features.py','symbol_scene.py']
        for mode in ['astra','latent']:
            report=json.loads((Path(a.validated_smoke)/f'smoke_72999_canonical_{mode}_learned/report.json').read_text())
            assert len(report['episodes'])==6
            assert all(report['metadata']['source_sha256'][name]==sources['scripts/agentic/'+name] for name in critical)
            events=[event for e in report['episodes'] for event in e['async_events']]
            assert not any(e.get('error') for e in events)
            assert any(e['status']=='delivered' and e['physics_steps']>0 for e in events)
            update(mode,status='prior_smoke_verified',smoke_source=a.validated_smoke)
    else:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(worker,mode) for mode in ['astra','latent']]
            for f in fs:f.result()
    # Both real services must pass before any large-budget runs.
    if not a.smoke_only:
        def large(mode):
            try:
                done=0
                for seed in protocol['seeds']:
                    for template in protocol['templates']:
                        # Rotate waiting-policy order to reduce chronology confounding.
                        modes=['continue','slow','learned'];shift=(seed-73001)%3;modes=modes[shift:]+modes[:shift]
                        for waiting in modes:
                            one(mode,waiting,seed,template);done+=6
                            update(mode,completed_episodes=done)
                            from analyze_async_v7 import analyze
                            with mutex:analyze(out)
                update(mode,status='completed')
            except BaseException as e:
                stopped.set();update(mode,status='failed',error=str(e));raise
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(large,m) for m in ['astra','latent']]
            for f in fs:f.result()
    save(out/'completed.json',dict(smoke_only=a.smoke_only,completed=True,wall_time=time.time()))
if __name__=='__main__':main()
