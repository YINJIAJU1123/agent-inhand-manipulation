"""Finite, resumable local-only evaluation; independent backend failure handling.

One simulator at a time on the local GPU. Latent inference is a separate local
process on that GPU. Physics never waits for model replies. No SSH or scheduler.
"""
import argparse,fcntl,hashlib,json,os,secrets,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CONTROLLER=ROOT/'outputs/latency_controller_v1/train_v2_20261015/controller_v1_final.pt'
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Preserve the previously announced total experiment allowance across restarts.
    previous=[]
    for name in ['smoke_20261016a','large_20261016']:
        previous += [json.loads(x.read_text()) for x in (ROOT/'outputs/async_v7'/name/'requests').glob('request_*.json')]
    used_tokens=sum((x.get('usage') or {}).get('total_tokens',0) for x in previous)
    paths=list((ROOT/'scripts/agentic').glob('*.py'))
    frozen={str(x.relative_to(ROOT)):sha(x) for x in paths}
    weights=[CONTROLLER,ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt',ROOT/'outputs/agentic_v4/checkpoints/latent_v2/goal_head.pt',ROOT/'outputs/agentic_v4/checkpoints/latent_v2/lora/adapter_model.safetensors']
    weight_hashes={str(x.relative_to(ROOT)):sha(x) for x in weights}
    protocol=dict(benchmark='live_async_v7_local',total_episodes=720,episodes_per_arm=120,seeds=list(range(73001,73011)),templates=['canonical','paraphrase'],waiting=['continue','slow','learned'],backends=['latent','astra'],max_steps=3600,call_budget=6,device='local RTX 3090 GPU 0; sequential simulators; latent inference shares GPU in separate process',source_sha256=frozen,weight_sha256=weight_hashes,max_transport_attempts=2,max_batch_attempts=2,prior_requests=len(previous),prior_reported_tokens=used_tokens,max_new_api_attempts=max(0,2160-len(previous)),new_token_stop_threshold=max(0,6000000-used_tokens),unknown_failed_request_tokens='not observable; reported usage may undercount interrupted requests',smoke_excluded=True)
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Frozen protocol/source/weights changed; use a new output')
    else:
        save(protocol_path,protocol);(out/'sources').mkdir(exist_ok=True)
        for path in paths:(out/'sources'/path.name).write_bytes(path.read_bytes())
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',ASTRA_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
    (out/'tmp').mkdir(exist_ok=True)
    env['PYTHONPATH']=':'.join(str(ROOT/'outputs'/x) for x in ['vlm_dependencies','learning_dependencies','video_dependencies'])+':'+str(ROOT/'scripts/agentic')
    keyfile=out/'bridge.key';key=secrets.token_urlsafe(32)
    fd=os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:f.write(key)
    env['ASTRA_BRIDGE_URL']='http://127.0.0.1:18781';env['ASTRA_BRIDGE_KEY_FILE']=str(keyfile)
    state=dict(pid=os.getpid(),status='starting',planned_episodes=720,completed_episodes=0,backends={m:dict(status='starting') for m in ['latent','astra']})
    services={};child=None;disabled=set()
    def progress(**fields):
        state.update(fields);state['updated']=time.time();save(out/'progress.json',state)
    def interrupt(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def health(mode):
        port=18781 if mode=='astra' else 18782
        request=urllib.request.Request(f'http://127.0.0.1:{port}/health',headers={'Authorization':'Bearer '+key} if mode=='astra' else {})
        with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
    def service(mode):
        port=18781 if mode=='astra' else 18782
        with socket.socket() as probe:probe.bind(('127.0.0.1',port))
        for attempt in range(2):
            if mode=='astra':
                cmd=[sys.executable,str(ROOT/'scripts/agentic/astra_bridge_v6.py'),'--port',str(port),'--workers','6','--max-calls',str(protocol['max_new_api_attempts']),'--max-tokens',str(protocol['new_token_stop_threshold']),'--max-sessions','800','--transport-attempts','2','--key-file',str(keyfile),'--audit-dir',str(out/'requests')]
            else:
                cmd=[sys.executable,str(ROOT/'scripts/agentic/latent_service_v7.py'),'--port',str(port),'--checkpoint',str(ROOT/'outputs/agentic_v4/checkpoints/latent_v2')]
            with (out/f'{mode}_service_attempt{attempt+1}.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            services[mode]=proc;deadline=time.monotonic()+240
            while proc.poll() is None and time.monotonic()<deadline:
                try:
                    info=health(mode)
                    if mode=='latent':assert info['checkpoint_sha256']==weight_hashes['outputs/agentic_v4/checkpoints/latent_v2/goal_head.pt']
                    state['backends'][mode].update(status='ready',service_pid=proc.pid);progress();return
                except Exception:time.sleep(2)
            stop(proc)
        raise RuntimeError(f'{mode} service failed startup; see service logs')
    def valid(dest,mode,smoke=False):
        r=json.loads((dest/'report.json').read_text());es=r['episodes'];m=r['metadata']
        if len(es)!=6 or any(e['calls']>6 for e in es):raise RuntimeError('Invalid episode count or budget')
        events=[x for e in es for x in e['async_events']]
        errors=[x.get('error') for x in events if x['status']=='error']
        if errors:raise RuntimeError('Active inference error: '+str(errors))
        if smoke and not any(x['status']=='delivered' and x['physics_steps']>0 for x in events):raise RuntimeError('No asynchronous delivery')
        if any(x['physics_steps']<=0 for x in events if x['status']=='delivered'):raise RuntimeError('Physics did not advance')
        if m['mode']!=mode or any(m['source_sha256'][Path(k).name]!=v for k,v in frozen.items()):raise RuntimeError('Source mismatch')
        return r
    def batch(mode,waiting,seed,template,smoke=False):
        nonlocal child
        name=f'{"smoke" if smoke else "eval"}_{seed}_{template}_{mode}_{waiting}';dest=out/name
        if (dest/'report.json').exists():valid(dest,mode,smoke);return
        if any(sha(ROOT/k)!=v for k,v in {**frozen,**weight_hashes}.items()):raise RuntimeError('Frozen source or weights changed')
        for attempt in range(1,3):
            attempt_dir=out/f'attempt_{name}_{attempt}'
            if (attempt_dir/'failure.json').exists():continue
            if attempt_dir.exists():raise RuntimeError('Interrupted attempt requires review: '+str(attempt_dir))
            state['backends'][mode].update(status='running',job=name,attempt=attempt)
            progress(current_job=name)
            cmd=[sys.executable,str(ROOT/'scripts/agentic/evaluate_async_v7.py'),'--headless','--device','cuda:0','--output',str(attempt_dir),'--seed',str(seed),'--mode',mode,'--waiting',waiting,'--template-set',template,'--max-steps','3600','--budget','6','--controller',str(CONTROLLER),'--latent-url','http://127.0.0.1:18782']
            if smoke or seed==73001:cmd.append('--record-video')
            try:
                if services[mode].poll() is not None:raise RuntimeError('Backend service exited')
                with (out/f'{name}_attempt{attempt}.log').open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    begin=time.monotonic()
                    while child.poll() is None:
                        if time.monotonic()-begin>3600:raise RuntimeError('Local simulator watchdog expired')
                        progress(current_job=name,job_wall_s=round(time.monotonic()-begin,1),simulator_pid=child.pid);time.sleep(5)
                    if child.returncode:raise RuntimeError(f'Simulator exit {child.returncode}')
                report=valid(attempt_dir,mode,smoke)
                attempt_dir.rename(dest)
                state['backends'][mode].update(last_completed=name,last_success=sum(e['success'] for e in report['episodes']),status='ready')
                progress();return
            except Exception as error:
                stop(child);attempt_dir.mkdir(exist_ok=True)
                save(attempt_dir/'failure.json',dict(batch=name,attempt=attempt,error=str(error),excluded_from_success_rate=True,time=time.time()))
                progress(last_failure=dict(batch=name,error=str(error)))
        raise RuntimeError('Both batch attempts failed: '+name)
    def disable(mode,error):
        disabled.add(mode);state['backends'][mode].update(status='blocked',error=str(error));stop(services.get(mode));progress()
    try:
        for mode in ['latent','astra']:
            try:service(mode)
            except Exception as error:disable(mode,error)
        # New GPU/runtime and same-GPU inference require new local smoke checks.
        for mode in ['latent','astra']:
            if mode in disabled:continue
            try:batch(mode,'learned',72999,'canonical',True)
            except Exception as error:disable(mode,error)
        if not a.smoke_only:
            from analyze_async_v7 import analyze
            for seed in protocol['seeds']:
                for template in protocol['templates']:
                    waits=['continue','slow','learned'];shift=(seed-73001)%3;waits=waits[shift:]+waits[:shift]
                    for waiting in waits:
                        for mode in ['latent','astra']:
                            if mode in disabled:continue
                            try:batch(mode,waiting,seed,template)
                            except Exception as error:disable(mode,error);continue
                            summary=analyze(out)
                            progress(status='running',completed_episodes=summary['completed_episodes'])
        progress(status='incomplete' if disabled else 'completed',current_job=None,disabled_backends=sorted(disabled))
    except BaseException as error:
        progress(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        stop(child)
        for proc in services.values():stop(proc)
        keyfile.unlink(missing_ok=True)
if __name__=='__main__':main()
