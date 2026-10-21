"""Finish local optimization and two-seed shadow validation. No cloud transport."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from assets import ROOT,PYTHON,save,sha
from collect_run import env_variables,KIT

def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--old-cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    collection=Path(a.collection).resolve();old=Path(a.old_cache).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n in ['logs','tmp']:(out/n).mkdir()
    state=dict(pid=os.getpid(),status='waiting_collection',cloud_calls=0);child=None;lock=None
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    frozen={str(f):sha(f) for f in Path(__file__).parent.glob('*.py')}
    save(out/'protocol.json',dict(source_sha256=frozen,collection=str(collection),additional_training_cache=str(old),
         held_out_seeds=[96311,96312],formal_seeds_untouched=[86101,86102],model_kind='mlp',
         purpose='Learned completion/progress given Astra semantic binding. Local shadow evaluation only; not a full semantic agent result.',cloud_calls=0))
    env=env_variables(out)
    def run(cmd,name,timeout=1800):
        nonlocal child
        for f,h in frozen.items():
            if sha(f)!=h:raise RuntimeError('Frozen optimization code changed')
        with (out/'logs'/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
            while child.poll() is None:
                progress(status=name,child_pid=child.pid,wall_s=time.monotonic()-start)
                if time.monotonic()-start>timeout:raise TimeoutError(name)
                time.sleep(5)
            if child.returncode:raise RuntimeError(name+' exited '+str(child.returncode))
    try:
        progress();deadline=time.monotonic()+3600
        while True:
            source=json.loads((collection/'progress.json').read_text())
            if source['status']=='completed':break
            if source['status'] in ['failed','interrupted']:raise RuntimeError('Data collection failed')
            if time.monotonic()>deadline:raise TimeoutError('Collection wait')
            progress(collection_completed=source.get('completed'),collection_item=source.get('current'));time.sleep(10)
        lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        run([PYTHON,str(ROOT/'scripts/semantic_completion_v2/cache.py'),'--data',str(collection/'data'),'--output',str(out/'cache')],'feature_cache')
        manifest=json.loads((out/'cache/manifest.json').read_text());old_manifest=json.loads((old/'manifest.json').read_text())
        for key in ['student_sha256','adapter_sha256','feature_source_sha256']:
            if manifest[key]!=old_manifest[key]:raise RuntimeError('Feature contract changed between cached splits')
        for row in old_manifest['shards']:
            if row['shard'].startswith('train_'):
                source=old/(row['shard']+'.pt');dest=out/'cache'/source.name
                if sha(source)!=row['cache_sha256']:raise RuntimeError('Previous training cache changed')
                os.link(source,dest);manifest['shards'].append(row)
        manifest['additional_training_manifest_sha256']=sha(old/'manifest.json');save(out/'cache/manifest.json',manifest)
        run([PYTHON,str(ROOT/'scripts/semantic_completion_v2/train.py'),'--cache',str(out/'cache'),'--output',str(out/'train'),'--kind','mlp'],'training')
        result=json.loads((out/'train/result.json').read_text());progress(training_result=result)
        for seed in [96311,96312]:
            run([PYTHON,str(ROOT/'scripts/semantic_completion_v2/shadow_eval.py'),'--checkpoint',str(out/'train/head.pt'),'--output',str(out/f'shadow_{seed}'),
                 '--seed',str(seed),'--headless','--device','cuda:0']+KIT,f'shadow_{seed}')
        reports=[json.loads((out/f'shadow_{s}/report.json').read_text()) for s in [96311,96312]]
        episodes=[r for report in reports for r in report['episodes']];proposals=[r for r in episodes if r['first_stop_proposal'] is not None]
        summary=dict(episodes=len(episodes),first_proposals=len(proposals),correct_first_proposals=sum(r['valid_first_proposal'] for r in proposals),
             reached=sum(r['first_true_reach'] is not None for r in episodes),protocol=reports[0]['protocol'])
        save(out/'shadow_summary.json',summary)
        first_stop_ok=bool(len(proposals)>=3 and summary['correct_first_proposals']/len(proposals)>=.9)
        progress(status='local_candidate_ready' if result['cloud_eligible'] and first_stop_ok else 'needs_improvement',child_pid=None,shadow_summary=summary,frame_gate_passed=result['cloud_eligible'],first_stop_gate_passed=first_stop_ok)
    except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        if lock:lock.close()
if __name__=='__main__':main()
