"""Finite frozen four-arm experiment. No scheduled task, old-account fallback or silent retry."""
import argparse,fcntl,json,os,random,secrets,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from common import ROOT,PYTHON,STUDENT,ADAPTER,CONTROLLER,ARMS,save,sha
from prepare import environment_variables,KIT
from analyze import summarize


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True);p.add_argument('--learning',required=True)
    p.add_argument('--account',required=True)
    p.add_argument('--transfer-approval')
    p.add_argument('--offline-only',action='store_true')
    a=p.parse_args()
    if not a.offline_only and not a.transfer_approval:p.error('--transfer-approval is required for cloud evaluation')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    learning=Path(a.learning).resolve();account=Path(a.account).resolve()
    for name in ('logs','tmp','requests'):(out/name).mkdir()
    state=dict(pid=os.getpid(),status='waiting_training',completed_episodes=0,planned_episodes=0 if a.offline_only else 192,planned_smoke_episodes=24)
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    progress();child=service=None;lock=None;keyfile=out/'bridge.key'
    try:
        deadline=time.monotonic()+7200
        while True:
            source=json.loads((learning/'progress.json').read_text())
            if source['status'] in ('failed','interrupted'):raise RuntimeError('Training pipeline did not complete')
            if source['status']=='completed':break
            if time.monotonic()>deadline:raise TimeoutError('Training wait exceeded two hours')
            progress(status='waiting_training',training_stage=source['status'],training_item=source.get('current'))
            time.sleep(10)
        checkpoint=learning/'train/evidence.pt'
        training=json.loads((learning/'train/result.json').read_text())
        if sha(checkpoint)!=training['checkpoint_sha256']:raise RuntimeError('Training checkpoint checksum mismatch')
        rng=random.Random(20261020);schedule=[]
        for repeat in [1,2]:
            for seed in [86101,86102]:
                for template in ['canonical','paraphrase']:
                    arms=list(ARMS);rng.shuffle(arms)
                    schedule.extend(dict(repeat=repeat,seed=seed,template=template,arm=arm) for arm in arms)
        sources=list(Path(__file__).parent.glob('*.py'))
        sources+=list((ROOT/'scripts/agentic').glob('*.py'))
        sources+=list((ROOT/'scripts/astra_runtime_v2').glob('*.py'))
        sources+=list((ROOT/'scripts/astra_transport_v2').glob('*.py'))
        frozen={str(p):sha(p) for p in sources+[STUDENT,ADAPTER,CONTROLLER,checkpoint]}
        protocol=dict(version='semantic_completion_four_arm_v1',seeds=[86101,86102],repeats=2,templates=['canonical','paraphrase'],arms=list(ARMS),schedule=schedule,
            episodes_per_arm=48,total_episodes=192,call_budget=6,max_steps=3600,orientation_threshold_rad=.16,
            max_api_attempts=1152,reported_token_stop_threshold=3000000,transport_attempts=1,
            inference='Actual Astra, physics advances, latency-aligned delivery; local evidence frontend runs in all arms.',
            primary='Autonomous successful stop; correct semantic binding AND full pose error<=.16 AND no drop.',
            secondary='Ever reached under correct binding; false stops, reached-without-completion, budget exhausted, drops, calls.',
            service_policy='Prespecified for this new run: any active API error or request >120s is service abnormal. Preserve raw outcomes; remove same case key from all four groups for matched normal-service comparison. No outcome-based reruns.',
            controller='Frozen V3 for all groups, initial cold hold, frozen student/action scale.',
            source_weight_sha256=frozen,training=training,
            scope='First implementation pilot. Readability/survival action prediction from random rollouts; not yet same-state action-conditioned completion-value learning.',
            language='Astra receives actual language; local helper uses disclosed text-only aliases for six supported symbols. No layout/case-index parser.',
            comparisons='Within this new autonomous-stop protocol only; do not pool with old reach-terminated baseline.')
        save(out/'protocol.json',protocol)
        lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=environment_variables(out)
        def run_job(name,job,smoke=False):
            nonlocal child
            for path,digest in frozen.items():
                if sha(path)!=digest:raise RuntimeError('Frozen source or weight changed')
            dest=out/name
            command=[PYTHON,str(ROOT/'scripts/semantic_completion_v1/evaluate.py'),'--output',str(dest),'--seed',str(job['seed']),
                '--arm',job['arm'],'--template-set',job['template'],'--evidence-model',str(checkpoint),
                '--headless','--device','cuda:0','--max-steps','140' if smoke else '3600','--budget','6']+KIT
            if smoke:command+=['--offline-smoke']
            elif job['repeat']==1 and job['seed']==86101:command+=['--record-video']
            save(out/'logs'/(name+'_command.json'),command)
            with (out/'logs'/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
                while child.poll() is None:
                    progress(status='offline_smoke' if smoke else 'evaluating',current=name,child_pid=child.pid,job_wall_s=time.monotonic()-start)
                    if (dest/'runtime_failure.json').exists():raise RuntimeError(f'{name}: runtime exception')
                    heartbeat=dest/'progress.json'
                    if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>300:raise TimeoutError(f'{name}: no physics progress for five minutes')
                    if time.monotonic()-start>2400:raise TimeoutError(name)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}')
            save(dest/'job.json',job)
            report=json.loads((dest/'report.json').read_text())
            if len(report['episodes'])!=6 or any(r['calls']>6 for r in report['episodes']):raise RuntimeError('Call/episode budget mismatch')
            events=[e for r in report['episodes'] for e in r['async_events']]
            if any(e.get('physics_steps',0)<=0 for e in events if e['status']=='delivered'):raise RuntimeError('Physics did not advance during a request')
            if smoke:return report
            errors=[e for e in events if e['status']=='error']
            if any(any(word in str(e.get('error','')).lower() for word in ['401','403','limit','unauthorized','identity','scope','refresh']) for e in errors):
                raise RuntimeError('Authorization or usage error; raw report retained')
            if sum(r['reason']=='infrastructure_error' for r in report['episodes'])>=3:raise RuntimeError('Infrastructure circuit breaker')
            summary=summarize(out)
            if not summary['initial_states_equal']:raise RuntimeError('Paired initial states differ')
            progress(completed_episodes=summary['completed_episodes'],last_completed=name,child_pid=None)
            return report
        # Mechanics and scoring are exercised without consuming a cloud call.
        for arm in ARMS:
            run_job('smoke_'+arm,dict(repeat=0,seed=86001,template='canonical',arm=arm),True)
        save(out/'smoke_validation.json',dict(status='completed',arms=list(ARMS),cloud_calls=0,excluded_from_formal=True))
        if a.offline_only:
            progress(status='offline_completed',current=None,child_pid=None)
            return
        if not training.get('cloud_eligible',True) or not training['metrics']['gate_calibrated'] or training['metrics']['symbol_accuracy_readable'] < .7:
            raise RuntimeError('Local validation gate failed; improve learned evidence before spending cloud evaluation budget')
        approval=Path(a.transfer_approval)
        while not approval.exists():
            progress(status='awaiting_transfer_approval',current=None,child_pid=None)
            if time.monotonic()>deadline:raise TimeoutError('Awaiting explicit image-transfer approval requested by automatic review')
            time.sleep(10)
        permitted=json.loads(approval.read_text())
        if not permitted.get('approved') or permitted.get('endpoint')!='https://api.openai.com/v1/responses':raise RuntimeError('Missing matching transfer approval')
        identity=json.loads((account/'identity.json').read_text())
        if not identity['different_from_previous']:raise RuntimeError('Expected a newly authorized account')
        key=secrets.token_urlsafe(32)
        with os.fdopen(os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(key)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env.update(ASTRA_BRIDGE_URL=f'http://127.0.0.1:{port}',ASTRA_BRIDGE_KEY_FILE=str(keyfile))
        cmd=[PYTHON,str(ROOT/'scripts/semantic_completion_v1/bridge.py'),'--auth-dir',identity['auth_dir'],'--auth-result',str(account/'result.json'),
             '--identity-sha256',identity['identity_sha256'],'--port',str(port),'--key-file',str(keyfile),'--audit-dir',str(out/'requests'),
             '--workers','6','--max-calls','1152','--max-tokens','3000000','--max-sessions','192','--transport-attempts','1']
        with (out/'logs/bridge.log').open('w') as log:service=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        wait=time.monotonic()+120
        while True:
            if service.poll() is not None:raise RuntimeError('New-account bridge failed')
            try:
                req=urllib.request.Request(env['ASTRA_BRIDGE_URL']+'/health',headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=3) as response:health=json.load(response)
                save(out/'bridge_health.json',health);break
            except Exception:
                if time.monotonic()>wait:raise TimeoutError('Bridge health')
                time.sleep(2)
        for job in schedule:
            if service.poll() is not None:raise RuntimeError('Bridge stopped')
            name=f"eval_r{job['repeat']}_{job['seed']}_{job['template']}_{job['arm']}"
            run_job(name,job)
        progress(status='completed',current=None,child_pid=None)
    except BaseException as e:
        try:state['completed_episodes']=summarize(out)['completed_episodes']
        except Exception:pass
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        stop(child);stop(service);keyfile.unlink(missing_ok=True)
        if lock:lock.close()


if __name__=='__main__':main()
