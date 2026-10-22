"""Bounded startup integration, sharing the resumed experiment's aggregate budget."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save,sha
from prepare import environment_variables,KIT
spec=importlib.util.spec_from_file_location('memory_statistics',ROOT/'scripts/astra_semantic_memory_live_v1/analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
statistics=module.statistics
ACCOUNT=ROOT/'outputs/semantic_completion_v1/account_20261020'


def read(p):return json.loads(Path(p).read_text())


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def usage(out):
    rows=[read(p) for p in (out/'requests').glob('request_*.json')]
    return dict(attempts=len(rows),reported_tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in rows),
        usage_unknown=sum(not r.get('usage') for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    out=a.output.resolve();parent=a.parent.resolve();permission=read(a.authorization)
    assert permission['approved'] and permission['user_confirmation'] and permission['endpoint']=='https://api.openai.com/v1/responses'
    assert permission['protocol_sha256']==sha(parent/'protocol.json')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('logs','tmp','requests','source'): (out/name).mkdir()
    frozen=dict(read(parent/'protocol.json')['source_weight_sha256'])
    for f in Path(__file__).parent.glob('*.py'):frozen[str(f.resolve())]=sha(f)
    for path,digest in frozen.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    save(out/'protocol.json',dict(version='fixed_startup_astra_integration_v1',planned_episodes=12,seeds=[110101,110102],
        memory_mode='memory',startup_mode='fixed_green',startup_skill=1,source_sha256=frozen,
        parent=str(parent),authorization_sha256=sha(a.authorization),
        aggregate_attempt_limit=144,aggregate_reported_token_stop_threshold=600000,
        budget='The parent run and this continuation share one allowance. Subtract every parent request and reported token. Unknown failed usage remains disclosed. Inflight requests can overshoot the token stop threshold.',
        scope='Development continuation on the same twelve initial cases. Compare against the already recorded memory/latch arm. Independent service latency and post-initialization images mean this is not a new randomized confirmation.',
        treatment='Before the first delivered Astra skill, execute pilot-selected green through the original pending V3. Local startup does not establish a semantic binding. All semantic choices and stop proposals remain Astra decisions.',
        first_skill_evidence='outputs/astra_startup_value_20261022/fixed_fresh_r1/result.json',
        success='Correct Astra-declared completion with correct semantic binding, full quaternion error <=0.16rad, and no drop. All raw failures retained.',
        service_policy='Raw denominator primary; exclude any active API error or latency >120s from both corresponding arms in a separate paired analysis.',
        max_steps=3600,calls_per_episode=6,candidate_promoted=False))
    state=dict(pid=os.getpid(),status='waiting_for_parent',completed_episodes=0,cloud_started=False)
    child=service=lock=None;keyfile=out/'bridge.key';env=environment_variables(out)
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def check_frozen():
        for path,digest in frozen.items():
            if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    def run(seed,mode,smoke):
        nonlocal child
        check_frozen();name=f'{"smoke" if smoke else "eval"}_{seed}_{mode}';dest=out/name
        command=[PYTHON,str(Path(__file__).with_name('evaluate.py')),'--output',str(dest),'--seed',str(seed),
            '--startup-mode',mode,'--memory-mode','memory','--arm','verify','--initialization','collision_clearance',
            '--evidence-model',str(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'),
            '--headless','--device','cuda:0','--max-steps','220' if smoke else '3600','--budget','6']+KIT
        command+=['--offline-smoke','--offline-retention-probe'] if smoke else ['--record-video']
        save(out/'logs'/(name+'_command.json'),command)
        with (out/'logs'/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic()
        while child.poll() is None:
            progress(status='smoke' if smoke else 'evaluating',current=name,child_pid=child.pid,elapsed_s=time.monotonic()-started)
            if time.monotonic()-started>2400:raise TimeoutError(name)
            pf=dest/'progress.json'
            if pf.exists() and time.time()-pf.stat().st_mtime>300:raise TimeoutError('No simulator progress')
            time.sleep(5)
        if child.returncode or not (dest/'report.json').exists():raise RuntimeError(name+' did not complete; inspect saved log')
        report=read(dest/'report.json');return dest,report
    def summarize():
        base=[];candidate=[];initial_checks=[]
        for seed in (110101,110102):
            d=out/f'eval_{seed}_fixed_green'
            if not (d/'report.json').exists():continue
            b=parent/f'eval_r1_{seed}_canonical_memory'
            same=read(d/'initial_applied.json')==read(b/'initial_applied.json');initial_checks.append(same)
            if not same:raise RuntimeError('Initial states differ from recorded memory/latch comparison')
            base.extend(dict(r,seed=seed) for r in read(b/'report.json')['episodes'])
            candidate.extend(dict(r,seed=seed) for r in read(d/'report.json')['episodes'])
        key=lambda r:(r['seed'],r['case'])
        excluded={key(r) for r in base+candidate if r['service_anomaly']}
        cleanbase=[r for r in base if key(r) not in excluded];cleancandidate=[r for r in candidate if key(r) not in excluded]
        result=dict(completed_episodes=len(candidate),planned_episodes=12,raw=dict(latch=statistics(base),fixed_green=statistics(candidate)),
            normal_service_matched=dict(latch=statistics(cleanbase),fixed_green=statistics(cleancandidate)),
            excluded_matched_keys=sorted(excluded),initial_states_equal=all(initial_checks),parent_usage=usage(parent),continuation_usage=usage(out),
            note='Development reuse of initial cases; no independent 80% confirmation. Previous latch observations are retained; service latency differs. No success-triggered reruns.')
        save(out/'summary.json',result);return result
    try:
        progress();deadline=time.monotonic()+3600
        while True:
            prior=read(parent/'progress.json')
            if prior['status']=='completed':break
            if prior['status'] in ('failed','interrupted'):raise RuntimeError('Parent ended abnormally; continuation withheld')
            if time.monotonic()>deadline:raise TimeoutError('Parent completion timeout')
            time.sleep(10)
        lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        local=read(ROOT/'outputs/astra_startup_value_20261022/fixed_fresh_r1/result.json')
        if not local['fresh_local_gate_passed']:raise RuntimeError('Startup local validation failed')
        smoke=[]
        for mode in ('latch','fixed_green'):
            dest,report=run(119001,mode,True);trace=read(dest/'startup_trace.json')
            before=[r for r in trace if r['step']<180]
            assert before and all(not any(r['semantic_skill_delivered']) for r in before)
            assert all(all(k==(mode=='fixed_green') for k,active in zip(r['control_known'],r['active']) if active) for r in before)
            assert any(r['action_delta_from_latch'][i]>1e-6 for r in before for i,active in enumerate(r['active']) if active) if mode=='fixed_green' else all(r['action_delta_from_latch'][i]==0 for r in before for i,active in enumerate(r['active']) if active)
            smoke.append(read(dest/'initial_applied.json'))
        assert smoke[0]==smoke[1]
        save(out/'smoke_validation.json',dict(status='completed',cloud_calls=0,initial_states_equal=True,delayed_first_reply_steps=180,semantic_binding_not_fabricated=True))
        prior=usage(parent);limits=dict(max_calls=144-prior['attempts'],max_tokens=600000-prior['reported_tokens'])
        save(out/'budget_ledger.json',dict(parent=prior,remaining=limits,aggregate_limit=dict(attempts=144,reported_tokens=600000),authorization_sha256=sha(a.authorization)))
        if not a.execute:progress(status='prepared_no_external_calls',child_pid=None);return
        if limits['max_calls']<36 or limits['max_tokens']<=0:raise RuntimeError('Insufficient aggregate allowance for one bounded six-case batch')
        identity=read(ACCOUNT/'identity.json');key=secrets.token_urlsafe(32)
        with os.fdopen(os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(key)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env.update(ASTRA_BRIDGE_URL=f'http://127.0.0.1:{port}',ASTRA_BRIDGE_KEY_FILE=str(keyfile))
        command=[PYTHON,str(ROOT/'scripts/semantic_completion_v1/bridge.py'),'--auth-dir',identity['auth_dir'],
            '--auth-result',str(ACCOUNT/'result.json'),'--identity-sha256',identity['identity_sha256'],'--port',str(port),
            '--key-file',str(keyfile),'--audit-dir',str(out/'requests'),'--workers','6','--max-calls',str(limits['max_calls']),
            '--max-tokens',str(limits['max_tokens']),'--max-sessions','12','--transport-attempts','1']
        with (out/'logs/bridge.log').open('w') as log:
            service=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+120
        while True:
            if service.poll() is not None:raise RuntimeError('Bridge exited')
            try:
                request=urllib.request.Request(env['ASTRA_BRIDGE_URL']+'/health',headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(request,timeout=3) as response:save(out/'bridge_health.json',json.load(response))
                break
            except Exception:
                if time.monotonic()>deadline:raise TimeoutError('Bridge health timeout')
                time.sleep(2)
        progress(cloud_started=True)
        for seed in (110101,110102):
            used=usage(out)
            if used['attempts']+36>limits['max_calls'] or used['reported_tokens']>=limits['max_tokens']:
                progress(status='stopped_aggregate_budget',child_pid=None);summarize();return
            dest,report=run(seed,'fixed_green',False);result=summarize();progress(completed_episodes=result['completed_episodes'],child_pid=None)
            errors=[e for r in report['episodes'] for e in r['async_events'] if e['status']=='error']
            if any(any(x in str(e.get('error','')).lower() for x in ('401','403','unauthorized','identity','scope','refresh','limit')) for e in errors):raise RuntimeError('Account or budget error')
            if sum(r['reason']=='infrastructure_error' for r in report['episodes'])>=3:raise RuntimeError('Infrastructure circuit breaker')
        progress(status='completed',current=None,child_pid=None)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e),child_pid=None);raise
    finally:
        stop(child);stop(service);keyfile.unlink(missing_ok=True)
        if lock:lock.close()


if __name__=='__main__':main()
