"""Finite, frozen 60-episode Astra rollout. No scheduler or paid-key fallback."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def summarize(dest):
    episodes=[];jobs=0
    for p in sorted(dest.glob('seed*/report.json')):
        r=json.loads(p.read_text());jobs+=1
        episodes.extend(dict(x,template_set=r['metadata']['template_set']) for x in r['episodes'])
    keys=[(e['seed'],e['case'],e['template_set']) for e in episodes]
    assert len(keys)==len(set(keys))
    traces=[t for e in episodes for t in e['trace']]
    n=len(episodes)
    summary={'completed_jobs':jobs,'planned_jobs':10,'episodes':n,'planned_episodes':60,
             'status':'completed' if n==60 else 'partial','model':'gpt-6-astra','auth_method':'sign_in_with_chatgpt',
             'success':sum(e['success'] for e in episodes),'success_rate':sum(e['success'] for e in episodes)/n if n else None,
             'drop':sum(e['drop'] for e in episodes),'grounding_correct':sum(e['grounding_correct'] for e in episodes),
             'stop_claimed':sum(e['stop_claimed'] for e in episodes),'false_stop':sum(e['false_stop'] for e in episodes),
             'calls':sum(e['calls'] for e in episodes),
             'usage':{k:sum((t.get('usage') or {}).get(k,0) for t in traces) for k in ['input_tokens','output_tokens','total_tokens']},
             'cached_input_tokens':sum((t.get('usage') or {}).get('input_tokens_details',{}).get('cached_tokens',0) for t in traces),
             'reasoning_tokens':sum((t.get('usage') or {}).get('output_tokens_details',{}).get('reasoning_tokens',0) for t in traces),
             'limits':'Five seed clusters, two phrasings share physical scenes. Primary reach metric uses evaluator stopping; not a test of autonomous successful stopping. Compare matched subsets only; different hardware requires initial-state verification.',
             'by_symbol':{s:{'n':sum(e['target_symbol']==s for e in episodes),'success':sum(e['success'] for e in episodes if e['target_symbol']==s)} for s in sorted({e['target_symbol'] for e in episodes})}}
    write(dest/'analysis.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:7');a=p.parse_args()
    dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=list((ROOT/'scripts/agentic').glob('*.py'))+list((ROOT/'source/BrainCo_DexHand').rglob('*.py'))+list((ROOT/'assets').rglob('*'))
    files=[x for x in files if x.is_file() and '__pycache__' not in str(x)]
    files += [ROOT/'configs/visual_student_freeze.json']
    for name in ['dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt','siglip2_text_classifier_v2.pt']:
        files.append(ROOT/'outputs/visual_student_v3/checkpoints'/name)
    sources=lambda:{str(x.relative_to(ROOT)):sha(x) for x in files}
    freeze={'hashes':sources(),'protocol':{'seeds':list(range(35001,35006)),'templates':['canonical','paraphrase'],'episodes':60,'call_budget':6,'max_steps':600,'api_workers':2,'model':'gpt-6-astra','reasoning':'low'},
            'runtime':{x:importlib.metadata.version(x) for x in ['isaacsim','isaaclab','torch']},'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':a.device,'renderer_physical_gpu':os.environ.get('ASTRA_RENDER_GPU')}
    if (dest/'freeze.json').exists():
        if json.loads((dest/'freeze.json').read_text())!=freeze:raise RuntimeError('frozen_experiment_changed')
    else:write(dest/'freeze.json',freeze)
    state={'pid':os.getpid(),'status':'starting','planned_episodes':60,'completed_jobs':0}
    def progress(**kwargs):
        state.update(kwargs);state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(dest/'progress.json',state)
    def stop(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    child=None
    try:
        completed=0
        for seed in range(35001,35006):
            for template in ['canonical','paraphrase']:
                name=f'seed{seed}_offset0_{template}_astra';out=dest/name
                if sources()!=freeze['hashes']:raise RuntimeError('frozen_sources_or_weights_changed')
                if not (out/'report.json').exists():
                    command=[sys.executable,str(ROOT/'scripts/agentic/evaluate_astra_v6.py'),'--headless','--output',str(out),'--seeds',str(seed),'--modes','astra',
                             '--template-set',template,'--budget','6','--max-steps','600','--api-workers','2','--device',a.device,'--request-audit-dir',str(dest/'request_audit')]
                    if seed==35001:command.append('--record-video')
                    progress(status='running',current_job=name,completed_jobs=completed)
                    with (dest/(name+'.log')).open('w') as log:
                        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        started=time.monotonic()
                        while child.poll() is None:
                            if time.monotonic()-started>1800:raise RuntimeError('job_watchdog_expired')
                            progress(current_job_elapsed_s=round(time.monotonic()-started,1));time.sleep(5)
                        if child.returncode:raise RuntimeError(f'job_failed_exit_{child.returncode}: {name}')
                r=json.loads((out/'report.json').read_text());m=r['metadata'];eps=r['episodes']
                assert m['seeds']==[seed] and m['template_set']==template and m['call_budget']==6 and m['max_steps']==600
                assert len(eps)==6 and {e['case'] for e in eps}==set(range(6))
                assert all(e['calls']<=6 for e in eps)
                for k,v in m['source_sha256'].items():assert freeze['hashes']['scripts/agentic/'+k]==v
                completed+=1;s=summarize(dest)
                progress(completed_jobs=completed,completed_episodes=s['episodes'],success=s['success'],usage=s['usage'])
        progress(status='completed',current_job=None)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        lock.close()

if __name__=='__main__':main()
