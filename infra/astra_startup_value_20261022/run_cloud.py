"""Finite action-consequence pilot. All model and simulation files are frozen."""
import base64
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts/astra_startup_value_v1'), str(ROOT / 'scripts/semantic_completion_v1')]
from common import save
from prepare import KIT
from analyze import analyze, ARMS

out = Path(os.environ['STARTUP_VALUE_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
for name in ('tmp', 'logs', 'cases'):
    (out / name).mkdir()
manifest = json.loads((ROOT / 'package_manifest.json').read_text())
for name, digest in manifest['sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen package mismatch ' + name)
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0',
    TMPDIR=str(out / 'tmp'), PYTHONPATH=':'.join([str(ROOT / 'scripts/agentic'),
        str(ROOT / 'outputs/learning_dependencies'), os.environ.get('PYTHONPATH', '')]))
state = dict(status='starting', completed=[], cloud_calls=0)
child = None
experiment_started = time.monotonic()


def progress(**kw):
    state.update(kw, updated=time.time())
    save(out / 'progress.json', state)
    print('STARTUP_VALUE_PROGRESS ' + json.dumps(state), flush=True)


def run(seed, wait, mode):
    global child
    dest = out / 'cases' / str(seed) / mode
    command = [sys.executable, str(ROOT / 'scripts/astra_startup_value_v1/collect.py'),
        '--mode', mode, '--seed', str(seed), '--wait-steps', str(wait), '--output', str(dest),
        '--headless', '--device', 'cuda:0']
    if mode != 'record':
        command += ['--source', str(dest.parent / 'record')]
    command += KIT
    name = str(seed) + '_' + mode
    save(out / 'logs' / (name + '_command.json'), command)
    with (out / 'logs' / (name + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    start = time.monotonic()
    while child.poll() is None:
        status = dest / 'progress.json'
        progress(status='running', current=name, child_pid=child.pid,
            child_progress=json.loads(status.read_text()) if status.exists() else None,
            elapsed_s=time.monotonic() - start)
        if time.monotonic() - experiment_started > 2400:
            raise TimeoutError('Pilot exceeded forty minutes')
        if time.monotonic() - start > 1200:
            raise TimeoutError(name + ' exceeded 20 minutes')
        time.sleep(15)
    report = dest / 'report.json'
    if child.returncode or not report.exists():
        raise RuntimeError((out / 'logs' / (name + '.log')).read_text()[-8000:])
    progress(completed=state['completed'] + [name], child_pid=None)


try:
    save(out / 'protocol.json', dict(seeds=[111101, 111102], wait_steps=[180, 360],
        modes=['record'] + ARMS + ['pose'], package=manifest,
        opportunity_gate='At least 12 states and oracle completion gain of at least two over the best fixed action.',
        selection='Predeclared development pilot, all initial cases included. Future goal randomized independently.',
        scope='Action-outcome data only. No API calls or automatic deployment.', cloud_calls=0))
    for seed, wait in ((111101, 180), (111102, 360)):
        for mode in ['record'] + ARMS + ['pose']:
            run(seed, wait, mode)
    result = analyze(out / 'cases')
    result['reports'] = {p.parent.parent.name + '_' + p.parent.name: json.loads(p.read_text())
        for p in (out / 'cases').glob('*/*/report.json')}
    save(out / 'result.json', result)
    blob = gzip.compress(json.dumps(result).encode(), mtime=0)
    print('STARTUP_VALUE_RESULT_SHA256 ' + hashlib.sha256(blob).hexdigest(), flush=True)
    print('STARTUP_VALUE_RESULT_BEGIN', flush=True)
    encoded = base64.b64encode(blob).decode()
    for i in range(0, len(encoded), 64):
        print(encoded[i:i+64], flush=True)
    print('STARTUP_VALUE_RESULT_END', flush=True)
    progress(status='completed', current=None, child_pid=None)
except BaseException:
    save(out / 'failure.json', dict(error=traceback.format_exc()))
    progress(status='failed', error=traceback.format_exc())
    raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
