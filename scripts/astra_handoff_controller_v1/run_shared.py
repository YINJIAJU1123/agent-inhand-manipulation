"""Fixed-horizon local training followed by paired reply-state evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,CONTROLLER,STUDENT,ADAPTER,save,sha
from prepare import environment_variables,KIT
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ('logs','tmp'): (out/name).mkdir()
    script=Path(__file__).with_name('evaluate_shared.py')
    sources=[script,Path(__file__),ROOT/'scripts/agentic/recovery_controller_v3.py',ROOT/'scripts/agentic/recovery_controller_v2.py',ROOT/'scripts/agentic/latency_request_cache_v1.py',ROOT/'scripts/agentic/student_features.py',ROOT/'scripts/agentic/symbol_scene.py',ROOT/'scripts/async_stability_v1/placement.py']
    frozen={str(f):sha(f) for f in sources+[CONTROLLER,STUDENT,ADAPTER]}
    save(out/'protocol.json',dict(training_source='outputs/astra_handoff_controller_v1/paired24_20261022/train',wait_action_replay=True,eval_seeds=[123201,123202],eval_delay=6,num_envs=12,
        selection='Fixed final training iteration, no evaluation-seed model selection.',source_sha256=frozen,
        treatment='Only the two-second post-reply bridge is fine-tuned. Pending phase executes original frozen V3 for both arms. A different balanced color is revealed only at delivery.',
        gate='Identical physical/sensor reply state; >=2 more correct six-step reaches and >=2 fewer post-reply drops among 24 nominal trials.',
        outcome='Correct full pose <=.16 rad for six consecutive post-reply steps without earlier drop. Includes all initial failures in nominal denominator.',
        scope='Scripted local-control diagnostic, no real Astra, semantic binding or autonomous-stop score. No automatic live deployment.',cloud_calls=0))
    env=environment_variables(out);child=None;state=dict(pid=os.getpid(),status='starting',completed=[],cloud_calls=0)
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def run(name,args,timeout):
        nonlocal child
        for f,h in frozen.items():
            if sha(f)!=h:raise RuntimeError('Frozen file changed: '+f)
        dest=out/name
        command=[PYTHON,str(script),'--output',str(dest),'--num-envs','12','--headless','--device','cuda:0']+args+KIT
        save(out/'logs'/(name+'_command.json'),command)
        with (out/'logs'/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic()
        while child.poll() is None:
            detail={}
            for file in ('progress.json','rollout_progress.json'):
                q=dest/file
                if q.exists():detail[file]=json.loads(q.read_text())
            progress(status='training' if name=='train' else 'evaluating',current=name,child_pid=child.pid,detail=detail,elapsed_s=time.monotonic()-started)
            if time.monotonic()-started>timeout:raise TimeoutError(name)
            time.sleep(10)
        if child.returncode or not (dest/'complete.json').exists():raise RuntimeError(name+' failed; '+(out/'logs'/(name+'.log')).read_text()[-1600:])
        progress(completed=state['completed']+[name],child_pid=None)
    try:
        candidate=ROOT/'outputs/astra_handoff_controller_v1/paired24_20261022/train/controller_final.pt';frozen[str(candidate)]=sha(candidate)
        summaries={};audits=[]
        for seed in (123201,123202):
            cache=ROOT/f'outputs/astra_handoff_controller_v1/paired24_20261022/cache_{seed}/request_state.pt'
            if not cache.exists():
                cached=out/f'cache_{seed}'
                command=[PYTHON,str(Path(__file__).with_name('train.py')),'--output',str(cached),'--num-envs','12','--seed',str(seed),'--mode','cache','--headless','--device','cuda:0']+KIT
                with (out/'logs'/f'cache_{seed}.log').open('w') as log:
                    proc=subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
                cache=cached/'request_state.pt'
            for arm,checkpoint in [('original',CONTROLLER),('handoff',candidate)]:
                extra=['--wait-source',str(out/f'original_{seed}')] if arm=='handoff' else []
                run(f'{arm}_{seed}',['--seed',str(seed),'--mode','learned','--checkpoint',str(checkpoint),'--request-cache',str(cache),'--delay','6','--record-slots','0','3']+extra,900)
            b=torch.load(out/f'original_{seed}/arrival_state.pt',map_location='cpu',weights_only=False)
            c=torch.load(out/f'handoff_{seed}/arrival_state.pt',map_location='cpu',weights_only=False)
            errors={k:float((b[k].double()-c[k].double()).abs().max()) for k in b if torch.is_tensor(b[k]) and k not in ('torch_rng','cuda_rng')}
            audit=dict(seed=seed,errors=errors,exact_equal=all(x==0 for x in errors.values()))
            audits.append(audit);save(out/'reply_state_audit.json',audits)
            if not audit['exact_equal']:raise RuntimeError('Reply states diverged before treatment')
        for arm in ('original','handoff'):
            rows=sum([json.loads((out/f'{arm}_{seed}/eval_trials.json').read_text()) for seed in (123201,123202)],[])
            summaries[arm]=dict(nominal=len(rows),successes=sum(r['success'] for r in rows),nominal_success_rate=sum(r['success'] for r in rows)/len(rows),
                prefix_failures=sum(not r['eligible'] for r in rows),wait_drops=sum(r['wait_drop'] for r in rows),post_reply_drops=sum(r['resume_drop'] for r in rows))
        b,c=summaries['original'],summaries['handoff']
        result=dict(status='completed',summaries=summaries,reply_states_exact_equal=True,
            local_gate_passed=c['successes']>=b['successes']+2 and c['post_reply_drops']<=b['post_reply_drops']-2,
            checkpoint_sha256=sha(candidate),cloud_calls=0,deployed=False,scope='Local changed-goal recovery; not real Astra task success.')
        save(out/'result.json',result);progress(status='completed',current=None,result=result)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        lock.close()

if __name__=='__main__':main()
