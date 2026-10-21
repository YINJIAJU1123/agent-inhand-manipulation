"""Finite paired learned-semantic-memory evaluation; preparation is strictly local.

--execute requires a newly authorized bounded experiment after the previous
reported-token cap was exhausted. Existing payload scope stays unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON, STUDENT, ADAPTER, CONTROLLER, save, sha
from prepare import environment_variables, KIT
from analyze import summarize
ARMS = ('none', 'memory')
SEMANTIC_CHECKPOINT=ROOT/'outputs/astra_symbol_trajectory_20261022/learning_dispatch/verified_shadow/model/head.pt'
CHECKPOINT = ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
ACCOUNT = ROOT/'outputs/semantic_completion_v1/account_20261020'


def read(path):
    return json.loads(Path(path).read_text())


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'tmp', 'requests', 'source'):
        (out/name).mkdir()
    result = read(CHECKPOINT.with_name('result.json'))
    if sha(CHECKPOINT) != result['checkpoint_sha256']:
        raise RuntimeError('Candidate checksum mismatch')
    fresh=read(ROOT/'outputs/astra_semantic_memory_20261022/result.json')
    if fresh['status']!='completed' or any(row['checkpoint_sha256']!=sha(SEMANTIC_CHECKPOINT) for row in fresh['results']):
        raise RuntimeError('Frozen reader fresh validation is missing or mismatched')
    save(out/'fresh_identity_validation.json',fresh)
    schedule = []
    rng = random.Random(20261022)
    for repeat in range(1, 2):
        for seed in (110101, 110102):
            for template in ('canonical',):
                arms = list(ARMS)
                rng.shuffle(arms)
                schedule.extend(dict(repeat=repeat, seed=seed, template=template, arm=arm) for arm in arms)
    folders = ('astra_semantic_memory_live_v1','astra_semantic_memory_v1','astra_symbol_verified_v1','astra_symbol_invariant_v1','astra_symbol_regions_v1','astra_retention_live_v2', 'astra_retention_online_v1', 'astra_retention_v1', 'astra_observation_control_v1', 'async_stability_v1', 'astra_completion_v3', 'semantic_completion_v3', 'semantic_completion_v2',
               'semantic_completion_v1', 'astra_runtime_v2', 'astra_transport_v2', 'agentic')
    sources = [p for folder in folders for p in (ROOT/'scripts'/folder).glob('*.py')]
    sources += list((ROOT/'source/BrainCo_DexHand/BrainCo_DexHand/algo/agentic').glob('*.py'))
    sources += list((ROOT/'source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct').glob('*.py'))
    for path in sources:
        target = out/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    frozen = {str(p): sha(p) for p in sources + [STUDENT, ADAPTER, CONTROLLER, CHECKPOINT, SEMANTIC_CHECKPOINT]}
    protocol = dict(version='learned_semantic_memory_live_20261022', total_episodes=24,
        episodes_per_arm=12, arms=list(ARMS), seeds=[110101, 110102], repeats=1,
        templates=['canonical'], schedule=schedule, max_steps=3600,
        call_budget=6, max_api_attempts=144, reported_token_stop_threshold=600000,
        transport_attempts=1, source_weight_sha256=frozen, checkpoint=str(CHECKPOINT),
        threshold=result['threshold'], candidate_offline_gate_passed=result['offline_gate_passed'],
        semantic_checkpoint=str(SEMANTIC_CHECKPOINT),semantic_threshold=.85,
        candidate_status='Frozen RGB reader passed fresh identity/coverage validation. No demonstrated task benefit yet. Existing dense pose verifier stays below its original 90% gate; this is an exploratory comparison, not baseline promotion.',
        primary='Correct agent-declared completion at <=0.16 rad with correct semantic binding and no drop.',
        secondary='Ever correct reach, false stop, reached without completion, cold/wait/total drops, rejected correct/incorrect proposals, calls.',
        controls='Both use frozen student, original V3, dense pose verifier, collision-clearance initialization, cold hold and six calls. Both run the same RGB reader and memory at 3Hz to equalize compute. Only the memory arm exposes learned historical color-symbol observations, confidence and age in Astra feedback. No geometric labels or target-color lookup enter feedback; no new action override or stop authority. Astra still selects all colors and bindings.',
        service_policy='Preserve every raw episode. Any active API error or measured request latency >120s excludes that exact case key from both arms in a separate matched normal-service analysis. No outcome-based reruns.',
        independence='Twelve unique physical initial states across two seeds. This is a bounded development comparison; reaching 80% on it alone cannot establish broad generalization.',
        physics='Advances during real asynchronous inference. Replies are latency aligned and consumed on a fresh camera step.',
        frozen_threshold='0.7 selected previously on calibration data; no retuning on these test outcomes.',
        comparison='Paired autonomous-stop protocol with shared dense verification and initial placement. All outcomes are retained, including early drops and abstentions. Independent closed-loop sensor variation remains possible after initialization.',
        payload=dict(endpoint='https://api.openai.com/v1/responses', model='gpt-6-astra',
            content=['simulation RGB frame and RGB-derived crop', 'semantic instruction', 'measured execution feedback, learned pose probabilities, and RGB-derived historical color-symbol confidence/timestamps', 'conversation/tool history'],
            credentials='Only the previously authorized new-account bridge. Credentials remain outside reports and repository.'),
        recordings='Every formal episode: 24 videos including failures.', preparation='Offline smoke must exercise memory-present and memory-absent transport plus stop handling; fresh identity validation retained separately. Request a new bounded budget after this concrete protocol is ready.')
    save(out/'protocol.json', protocol)
    return protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--transfer-approval')
    p.add_argument('--prepare-wait-pid',type=int)
    p.add_argument('--control-output',type=Path)
    a = p.parse_args()
    if a.prepare_wait_pid and (a.execute or a.control_output is None):
        p.error('--prepare-wait-pid needs --control-output and cannot accompany --execute')
    out = Path(a.output).resolve()
    protocol = read(out/'protocol.json') if a.execute else prepare(out)
    if a.execute:
        if not a.transfer_approval:
            p.error('--execute requires explicit --transfer-approval')
        permission = read(a.transfer_approval)
        if not (permission.get('approved') and permission.get('endpoint') == protocol['payload']['endpoint']
                and permission.get('max_episodes', 0) >= 24
                and permission.get('max_api_attempts', 0) >= 144
                and permission.get('reported_token_stop_threshold', 0) >= 600000
                and permission.get('protocol_sha256') == sha(out/'protocol.json')
                and permission.get('user_confirmation')):
            raise RuntimeError('Missing matching authorization for this new bounded experiment')
        if read(out/'smoke_validation.json')['status'] != 'completed':
            raise RuntimeError('Local mechanics smoke required')
        if list(out.glob('eval_*/report.json')) or list((out/'requests').glob('request_*.json')):
            raise RuntimeError('Existing cloud outcomes: review explicitly rather than silently restart')
    state = dict(pid=os.getpid(), status='preparing', planned_episodes=24, completed_episodes=0,
                 exploratory=True, candidate_promoted=False, cloud_started=False)
    child = service = None
    keyfile = out/'bridge.key'
    lock = None
    def progress(**kwargs):
        state.update(kwargs, updated=time.time())
        save(out/'progress.json', state)
    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    def check_frozen():
        for path, digest in protocol['source_weight_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('Frozen source or weight changed: '+path)
    def run_job(job, smoke=False):
        nonlocal child
        check_frozen()
        name = 'smoke_'+job['arm'] if smoke else f"eval_r{job['repeat']}_{job['seed']}_{job['template']}_{job['arm']}"
        dest = out/name
        command = [PYTHON, str(Path(__file__).with_name('evaluate.py')), '--output', str(dest),
            '--seed', str(job['seed']), '--arm', 'verify', '--initialization', 'collision_clearance', '--memory-mode', job['arm'], '--template-set', job['template'],
            '--evidence-model', str(CHECKPOINT), '--headless', '--device', 'cuda:0',
            '--max-steps', '180' if smoke else '3600', '--budget', '6']+KIT
        if smoke:
            command += ['--offline-smoke']
        else:
            command += ['--record-video']
        save(out/'logs'/(name+'_command.json'), command)
        with (out/'logs'/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            while child.poll() is None:
                progress(status='local_smoke' if smoke else 'evaluating', current=name,
                         child_pid=child.pid, job_elapsed_s=time.monotonic()-start)
                if (dest/'runtime_failure.json').exists():
                    raise RuntimeError(name+': evaluator exception')
                heartbeat = dest/'progress.json'
                if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime > 300:
                    raise TimeoutError(name+': no physics progress for five minutes')
                if time.monotonic()-start > 2400:
                    raise TimeoutError(name)
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(name+': evaluator exit '+str(child.returncode))
        save(dest/'job.json', job)
        report = read(dest/'report.json')
        rows = report['episodes']
        if len(rows) != 6 or any(r['calls'] > 6 for r in rows):
            raise RuntimeError('Episode/call budget mismatch')
        events = [e for row in rows for e in row['async_events']]
        if any(e.get('physics_steps', 0) <= 0 for e in events if e['status'] == 'delivered'):
            raise RuntimeError('Physics did not advance during a request')
        if smoke:
            if not any(e['status'] == 'delivered' for e in events):
                raise RuntimeError('No scripted reply delivered in mechanics smoke')
            if not any(e['event'] == 'astra_stop_proposed' for row in rows for e in row['controller_events']):
                raise RuntimeError('Mechanics smoke did not exercise stop handling')
            contexts=read(dest/'memory_feedback.json')
            if not contexts or any(x['exposed'] != (job['arm']=='memory') for x in contexts):
                raise RuntimeError('Wrong memory exposure on submitted requests')
            if job['arm']=='memory' and any(x['context'] is None for x in contexts):
                raise RuntimeError('Missing learned memory context')
            return report
        summary = summarize(out)
        progress(completed_episodes=summary['completed_episodes'], child_pid=None)
        if not summary['source_initial_states_equal']:
            raise RuntimeError('Paired initial states differ')
        failures = [e for e in events if e['status'] == 'error']
        if any(any(word in str(e.get('error', '')).lower() for word in
                   ('401', '403', 'unauthorized', 'identity', 'scope', 'refresh', 'limit')) for e in failures):
            raise RuntimeError('Account or budget error; raw outcomes retained')
        if sum(row['reason'] == 'infrastructure_error' for row in rows) >= 3:
            raise RuntimeError('Infrastructure circuit breaker; raw outcomes retained')
        return report
    try:
        progress()
        check_frozen()
        if a.prepare_wait_pid:
            deadline=time.monotonic()+7200
            while True:
                control=read(a.control_output/'progress.json')
                if control['status']=='completed':
                    save(out/'control_dependency.json',dict(status='completed',output=str(a.control_output.resolve()),
                        summary_sha256=sha(a.control_output/'summary.json'),note='Completion only; treatment benefit must be reviewed separately before live execution.'))
                    break
                if control['status'] in ('failed','interrupted'):
                    progress(status='not_started',reason='Shared-prefix diagnostic ended without completion');return
                os.kill(a.prepare_wait_pid,0)
                command=Path(f'/proc/{a.prepare_wait_pid}/cmdline').read_bytes()
                if b'astra_reply_retention_v1/run.py' not in command:
                    raise RuntimeError('Dependency PID no longer identifies the expected runner')
                progress(status='waiting_for_control',dependency_pid=a.prepare_wait_pid,dependency_current=control.get('current'))
                if time.monotonic()>deadline:raise TimeoutError('Control dependency observation deadline')
                time.sleep(10)
        lock = Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        deadline=time.monotonic()+60
        while True:
            try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('GPU lock still held after dependency')
                time.sleep(2)

        env = environment_variables(out)
        if not a.execute:
            initials=[]
            for arm in ARMS:
                report=run_job(dict(seed=110001,template='canonical',arm=arm,repeat=0),smoke=True)
                initials.append(read(out/('smoke_'+arm)/'initial_applied.json'))
            if initials[0]!=initials[1]:raise RuntimeError('Smoke applied initial states differ')
            save(out/'smoke_validation.json',dict(status='completed',arms=list(ARMS),episodes=12,
                cloud_calls=0,excluded_from_formal=True,applied_initial_states_equal=True,
                purpose='Scripted asynchronous inspection, RGB-reader inference, present/absent memory feedback, and premature-stop plumbing only. No task-performance evidence.'))
            progress(status='prepared_no_external_calls',child_pid=None,current=None)
            return
        identity = read(ACCOUNT/'identity.json')
        if not identity['different_from_previous']:
            raise RuntimeError('Expected the newly authorized account')
        key = secrets.token_urlsafe(32)
        with os.fdopen(os.open(keyfile, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as file:
            file.write(key)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env.update(ASTRA_BRIDGE_URL=f'http://127.0.0.1:{port}', ASTRA_BRIDGE_KEY_FILE=str(keyfile))
        command = [PYTHON, str(ROOT/'scripts/semantic_completion_v1/bridge.py'),
            '--auth-dir', identity['auth_dir'], '--auth-result', str(ACCOUNT/'result.json'),
            '--identity-sha256', identity['identity_sha256'], '--port', str(port),
            '--key-file', str(keyfile), '--audit-dir', str(out/'requests'), '--workers', '6',
            '--max-calls', '144', '--max-tokens', '600000', '--max-sessions', '24', '--transport-attempts', '1']
        with (out/'logs/bridge.log').open('w') as log:
            service = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+120
        while True:
            if service.poll() is not None:
                raise RuntimeError('New-account bridge exited')
            try:
                request = urllib.request.Request(env['ASTRA_BRIDGE_URL']+'/health', headers={'Authorization': 'Bearer '+key})
                with urllib.request.urlopen(request, timeout=3) as response:
                    save(out/'bridge_health.json', json.load(response))
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Bridge health timeout')
                time.sleep(2)
        progress(cloud_started=True)
        for job in protocol['schedule']:
            if service.poll() is not None:
                raise RuntimeError('Bridge stopped')
            run_job(job)
        progress(status='completed', child_pid=None, current=None)
    except BaseException as error:
        progress(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                 error=str(error), child_pid=None)
        raise
    finally:
        stop(child)
        stop(service)
        keyfile.unlink(missing_ok=True)
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
