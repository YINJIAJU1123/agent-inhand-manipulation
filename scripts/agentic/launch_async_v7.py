"""Local owner of finite async benchmark, OAuth bridge and separate latent GPU."""
import argparse,json,os,secrets,shlex,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HOST='brainco@10.3.100.21';REMOTE='/home/brainco/agent_inhand_astra_20261015'
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--smoke-only',action='store_true');p.add_argument('--validated-smoke');a=p.parse_args()
    out=ROOT/'outputs/async_v7'/a.name;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'launcher_status.json').exists():raise RuntimeError('Fresh run required')
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
    env['PYTHONPATH']=':'.join(str(ROOT/'outputs'/x) for x in ['vlm_dependencies','learning_dependencies','video_dependencies'])+':'+str(ROOT/'scripts/agentic')
    key=secrets.token_urlsafe(32);keyfile=out/'bridge.key';fd=os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(key)
    remote_key='/home/brainco/.config/agent-inhand-astra/async_v7_'+a.name+'.key'
    subprocess.run(['ssh','-o','BatchMode=yes',HOST,'umask 077; cat > '+shlex.quote(remote_key)],input=key.encode(),check=True)
    remote_out=REMOTE+'/outputs/async_v7/'+a.name
    processes=[]
    def status(**fields):save(out/'launcher_status.json',dict(pid=os.getpid(),updated=time.time(),**fields))
    def launch(cmd,name):
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);return proc
    def health(url,key=None):
        r=urllib.request.Request(url,headers={'Authorization':'Bearer '+key} if key else {})
        with urllib.request.urlopen(r,timeout=2) as f:return json.load(f)
    def interrupt(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    try:
        status(status='starting_services')
        bridge=launch([sys.executable,str(ROOT/'scripts/agentic/astra_bridge_v6.py'),'--port','18771','--workers','6','--max-calls','2160' if a.validated_smoke else '2196','--max-tokens','6000000','--max-sessions','366','--key-file',str(keyfile),'--audit-dir',str(out/'requests')],'bridge')
        latent=launch([sys.executable,str(ROOT/'scripts/agentic/latent_service_v7.py'),'--port','18772','--checkpoint',str(ROOT/'outputs/agentic_v4/checkpoints/latent_v2')],'latent_service')
        deadline=time.monotonic()+240
        while True:
            if any(p.poll() is not None for p in processes):raise RuntimeError('Service startup failed')
            try:health('http://127.0.0.1:18771/health',key);health('http://127.0.0.1:18772/health');break
            except Exception:
                if time.monotonic()>deadline:raise RuntimeError('Service startup timeout')
                time.sleep(2)
        cmd='cd '+shlex.quote(REMOTE)+' && env ASTRA_BRIDGE_URL=http://127.0.0.1:18771 ASTRA_BRIDGE_KEY_FILE='+shlex.quote(remote_key)+' /home/brainco/isaaclab5090/env/bin/python scripts/agentic/run_async_v7.py --output '+shlex.quote(remote_out)+(' --smoke-only' if a.smoke_only else '')+(' --validated-smoke '+shlex.quote(a.validated_smoke) if a.validated_smoke else '')
        remote=launch(['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-R','127.0.0.1:18771:127.0.0.1:18771','-R','127.0.0.1:18772:127.0.0.1:18772',HOST,cmd],'remote')
        status(status='running',ssh_pid=remote.pid,bridge_pid=bridge.pid,latent_pid=latent.pid,planned_episodes=12 if a.smoke_only else 720 if a.validated_smoke else 732)
        while remote.poll() is None:
            if bridge.poll() is not None or latent.poll() is not None:raise RuntimeError('Inference service exited')
            subprocess.run(['rsync','-a','--exclude=*.mp4','--exclude=*.log',HOST+':'+remote_out+'/',str(out)+'/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)
            status(status='running',ssh_pid=remote.pid,bridge_pid=bridge.pid,latent_pid=latent.pid,planned_episodes=12 if a.smoke_only else 720 if a.validated_smoke else 732)
            time.sleep(15)
        subprocess.run(['rsync','-a',HOST+':'+remote_out+'/',str(out)+'/'],check=True,timeout=180)
        if remote.returncode:raise RuntimeError('Remote evaluation failed; inspect logs')
        status(status='completed')
    except BaseException as e:status(status='failed',error=str(e));raise
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        keyfile.unlink(missing_ok=True)
        subprocess.run(['ssh','-o','BatchMode=yes',HOST,'rm -f '+shlex.quote(remote_key)],timeout=15,check=False)
if __name__=='__main__':main()
