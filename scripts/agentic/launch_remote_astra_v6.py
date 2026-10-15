"""Own the local OAuth bridge, SSH reverse tunnel, and finite remote rollout."""
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from astra_subscription_v6 import save

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'outputs/agentic_v4/astra_eval_20261015'
HOST='brainco@10.3.100.21'
REMOTE='/home/brainco/agent_inhand_astra_20261015'
REMOTE_OUT=REMOTE+'/outputs/astra_eval_20261015'
KEY_LOCAL=Path.home()/'.config/agent-inhand-siwc/bridge_20261015.key'
KEY_REMOTE='/home/brainco/.config/agent-inhand-astra/bridge_20261015.key'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(DEST/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key=secrets.token_urlsafe(32)
    fd=os.open(KEY_LOCAL,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:f.write(key)
    subprocess.run(['ssh','-o','BatchMode=yes',HOST,'umask 077; mkdir -p /home/brainco/.config/agent-inhand-astra; cat > '+KEY_REMOTE],input=key.encode(),check=True)
    env=dict(os.environ)
    env['PYTHONPATH']=':'.join(str(ROOT/'outputs'/x) for x in ['vlm_dependencies','learning_dependencies','video_dependencies'])+':'+str(ROOT/'scripts/agentic')
    bridge=remote=None
    def progress(**v):save(DEST/'launcher_status.json',dict(pid=os.getpid(),updated_at=time.time(),**v))
    def stop(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:
        with (DEST/'bridge.log').open('a') as log:
            bridge=subprocess.Popen([sys.executable,str(ROOT/'scripts/agentic/astra_bridge_v6.py'),'--key-file',str(KEY_LOCAL),'--audit-dir',str(DEST/'requests')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+60
        while True:
            if bridge.poll() is not None:raise RuntimeError('bridge_start_failed')
            try:
                r=urllib.request.Request('http://127.0.0.1:18761/health',headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(r,timeout=2) as f:json.load(f)
                break
            except Exception:
                if time.monotonic()>deadline:raise RuntimeError('bridge_start_timeout')
                time.sleep(1)
        command='cd '+shlex.quote(REMOTE)+' && mkdir -p tmp && env TMPDIR='+shlex.quote(REMOTE+'/tmp')+' CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 ASTRA_RENDER_GPU=7 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 ASTRA_BRIDGE_URL=http://127.0.0.1:18761 ASTRA_BRIDGE_KEY_FILE='+shlex.quote(KEY_REMOTE)+' /home/brainco/isaaclab5090/env/bin/python scripts/agentic/run_astra_v6.py --device cuda:0 --output '+shlex.quote(REMOTE_OUT)
        with (DEST/'remote_launcher.log').open('a') as log:
            remote=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-R','127.0.0.1:18761:127.0.0.1:18761',HOST,command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        progress(status='running',host=HOST,device='cuda:0',physical_gpu=7,bridge_pid=bridge.pid,ssh_pid=remote.pid,planned_episodes=60)
        while remote.poll() is None:
            if bridge.poll() is not None:raise RuntimeError('bridge_exited')
            subprocess.run(['rsync','-a','--exclude=*.png','--exclude=*.mp4',HOST+':'+REMOTE_OUT+'/',str(DEST)+'/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)
            time.sleep(20)
        subprocess.run(['rsync','-a',HOST+':'+REMOTE_OUT+'/',str(DEST)+'/'],check=True,timeout=180)
        if remote.returncode:raise RuntimeError('remote_runner_failed')
        progress(status='completed',planned_episodes=60)
    except BaseException as e:
        progress(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:
        for proc in [remote,bridge]:
            if proc and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        # Short-lived experiment keys are removed; OAuth record is unaffected.
        KEY_LOCAL.unlink(missing_ok=True)
        subprocess.run(['ssh','-o','BatchMode=yes',HOST,'rm -f '+KEY_REMOTE],timeout=15,check=False)

if __name__=='__main__':main()
