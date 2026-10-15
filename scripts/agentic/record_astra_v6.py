"""Two fresh Astra video batches; separate from the completed benchmark."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/agentic_v4/astra_video_20261015'

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'protocol.json').exists():raise RuntimeError('Use a new output directory for a fresh recording')
    paths=list((ROOT/'scripts/agentic').glob('*.py'))
    paths+=list((ROOT/'source/BrainCo_DexHand').rglob('*.py'))
    paths+=list((ROOT/'assets/usd/semantic_symbols').glob('*.usda'))
    paths+=[ROOT/'configs/visual_student_freeze.json']
    paths+=[ROOT/'outputs/visual_student_v3/checkpoints'/n for n in ['dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt','siglip2_text_classifier_v2.pt']]
    hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen=hashes()
    save(OUT/'protocol.json',dict(seeds=[35002,35003],template='canonical',purpose='fresh demonstration reruns, excluded from original 60-episode benchmark',model='gpt-6-astra',call_budget=6,max_steps=600,api_workers=2,max_api_calls=72,max_api_tokens=400000,device='local RTX3090 / cuda:0',source_hashes=frozen))
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',ASTRA_RENDER_GPU='0',TMPDIR=str(OUT/'tmp'))
    (OUT/'tmp').mkdir(exist_ok=True)
    env.pop('ASTRA_BRIDGE_URL',None);env.pop('ASTRA_BRIDGE_KEY_FILE',None)
    env['PYTHONPATH']=':'.join(str(ROOT/'outputs'/n) for n in ['vlm_dependencies','learning_dependencies','video_dependencies'])
    state=dict(pid=os.getpid(),status='starting',planned_episodes=12,completed_episodes=0)
    def progress(**fields):
        state.update(fields);state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(OUT/'progress.json',state)
    def interrupt(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    child=None;episodes=[]
    try:
        for seed in [35002,35003]:
            if hashes()!=frozen:raise RuntimeError('Sources changed during recording')
            name=f'seed{seed}_offset0_canonical_astra';dest=OUT/name
            cmd=[sys.executable,str(ROOT/'scripts/agentic/evaluate_astra_v6.py'),'--headless','--device','cuda:0','--output',str(dest),'--seeds',str(seed),'--modes','astra','--template-set','canonical','--budget','6','--max-steps','600','--api-workers','2','--max-api-calls','72','--max-api-tokens','400000','--request-audit-dir',str(OUT/'requests'),'--record-video']
            progress(status='running',current_job=name)
            with (OUT/(name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic()
                while child.poll() is None:
                    if time.monotonic()-start>1800:raise RuntimeError('Recording watchdog expired')
                    progress(job_elapsed_s=round(time.monotonic()-start,1));time.sleep(5)
                if child.returncode:raise RuntimeError(f'Simulator exit {child.returncode}')
            report=json.loads((dest/'report.json').read_text())
            assert report['metadata']['seeds']==[seed] and len(report['episodes'])==6
            for e in report['episodes']:
                assert e['calls']<=6 and (dest/e['video']['path']).stat().st_size>0
            episodes.extend(report['episodes'])
            summary=dict(purpose='demonstration reruns; do not merge with original benchmark',episodes=len(episodes),success=sum(e['success'] for e in episodes),drop=sum(e['drop'] for e in episodes),false_stop=sum(e['false_stop'] for e in episodes),calls=sum(e['calls'] for e in episodes))
            save(OUT/'summary.json',summary);progress(completed_episodes=len(episodes))
        progress(status='completed',current_job=None)
    except BaseException as e:
        progress(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

if __name__=='__main__':main()
