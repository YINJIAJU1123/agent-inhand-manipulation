"""Finite goal-study pipeline: idle GPU, smoke admission, then registered live trials."""
import argparse,fcntl,json,os,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path

CONDITIONS=('correct_active_goal','retrained_no_explicit_goal','shuffled_controller_goal')
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def verify_smoke(root):
    initial=None;counts={}
    for name in CONDITIONS:
        out=root/'smoke'/name/'182941'
        manifest=json.loads((out/'manifest.json').read_text())
        result=json.loads((out/'result.json').read_text())
        if result['status']!='completed' or result['recorded_episodes']!=2:raise ValueError('Incomplete smoke')
        if manifest['formal_trials_used'] or manifest['upstream']['actual_model_calls']!=0:
            raise ValueError('Smoke must use only public action replay')
        expected_mode='center' if name=='retrained_no_explicit_goal' else 'active'
        expected_shift='cyclic_shift' if name=='shuffled_controller_goal' else 'none'
        actual=manifest['controller_goal_intervention']
        if (actual['mode']!=expected_shift or actual['waiting_goal_input']!=expected_mode
                or actual['execution_goal_input']!=expected_mode):raise ValueError('Wrong loaded goal mode')
        episodes=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
        current=[(row['trial_seed'],row['initial_state']) for row in episodes]
        if initial is None:initial=current
        elif current!=initial:raise ValueError('Smoke initial physical states differ')
        validity=json.loads((out.parent/'182941.validity.json').read_text())
        if not validity['physical_log_valid']:raise ValueError('Invalid smoke physics log')
        count={'pending':0,'execution':0}
        with (out/'steps.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                if row['controller_goal_input']!=expected_mode:raise ValueError('Actor goal mode mismatch')
                expected=None if expected_mode=='center' else (row['active_goal']+(1 if expected_shift=='cyclic_shift' else 0))%6
                if row['controller_goal_index']!=expected:raise ValueError('Actor goal mapping mismatch')
                count['pending' if row['pending'] else 'execution']+=1
        if not all(count.values()):raise ValueError('Smoke did not exercise both waiting and execution')
        counts[name]=count
    save(root/'smoke_admission.json',{'passed':True,'scope':'Runtime correctness only, not task performance admission','counts':counts})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--replication_root',type=Path,required=True);a=p.parse_args()
    lock=(a.root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.root/'queue_status.json').exists():raise FileExistsError('Inspect prior process before resuming')
    state=dict(pid=os.getpid(),created_at=datetime.now(timezone.utc).isoformat(),completed=[],status='initializing')
    def persist(**kw):
        state.update(kw,observed_at=datetime.now(timezone.utc).isoformat());save(a.root/'queue_status.json',state)
    def acquire_gpu():
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            readings=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
            persist(status='waiting_for_idle_gpu',gpu_observations=readings,child_pid=None,gpu=None)
            for gpu,mem,util in readings:
                if mem>=1000 or util>10:continue
                candidate=Path(f'/tmp/agentic_rotation_experiment_gpu_{gpu}.lock').open('a')
                try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close();continue
                return gpu,candidate
            time.sleep(30)
    try:
        for stage in ('smoke','formal'):
            if stage=='formal':
                verify_smoke(a.root)
                # Avoid concurrent upstream calls with the primary replication.
                while True:
                    prior=json.loads((a.replication_root/'queue_status.json').read_text())
                    if prior['status']=='completed':break
                    if prior['status']=='failed':raise RuntimeError('Replication failed; review before additional model calls')
                    try:os.kill(prior['pid'],0)
                    except ProcessLookupError:raise RuntimeError('Replication supervisor missing; inspect results')
                    persist(status='waiting_for_replication',stage=stage,condition=None,predecessor_pid=prior['pid'])
                    time.sleep(30)
            for name in CONDITIONS:
                persist(stage=stage,condition=name)
                gpu,gpu_lock=acquire_gpu()
                try:
                    if stage=='formal':
                        key=(a.runtime/'outputs/live_services_20261029/bridge.key').read_text().strip()
                        req=urllib.request.Request('http://127.0.0.1:18812/health',headers={'Authorization':'Bearer '+key})
                        with urllib.request.urlopen(req,timeout=10) as r:
                            if r.status!=200:raise RuntimeError('Astra bridge unavailable')
                    env=dict(os.environ,RUNTIME_ROOT=str(a.runtime),STUDY_ROOT=str(a.root),CONDITION=name,STAGE=stage,CUDA_VISIBLE_DEVICES=str(gpu))
                    with (a.root/(stage+'_'+name+'.queue.log')).open('x') as log:
                        child=subprocess.Popen(['bash',str(a.root/'run_goal_condition.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
                        persist(status='running',gpu=gpu,child_pid=child.pid)
                        code=child.wait()
                    if code:raise RuntimeError(stage+'/'+name+' exited '+str(code))
                    state['completed'].append(stage+'/'+name);persist(child_pid=None)
                finally:gpu_lock.close()
        persist(status='completed',condition=None,stage=None,gpu=None,child_pid=None)
    except BaseException as e:
        persist(status='failed',error=type(e).__name__+': '+str(e));raise
if __name__=='__main__':main()
