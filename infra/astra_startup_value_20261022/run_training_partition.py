"""Independent startup data, seed-held-out learning, and bounded artifact export.

Requires the completed action pilot's opportunity gate. Three partitions own
disjoint seeds. Rank zero freezes on calibration before opening audit labels.
This script never starts Astra requests or promotes the learned controller.
"""
import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT

rank = int(os.environ['STARTUP_VALUE_RANK'])
if rank not in (0, 1, 2):
    raise ValueError('Expected one of three partitions')
out = Path(os.environ['STARTUP_VALUE_OUTPUT'])
pilot = Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_startup_value_20261022/pilot_r1/result.json')
gate = json.loads(pilot.read_text())
if gate['status'] != 'completed' or not gate['opportunity_gate_passed'] or gate['nominal_states'] < 12:
    raise RuntimeError('Completed pilot opportunity gate required before collecting training data')
manifest = json.loads((ROOT / 'package_manifest.json').read_text())
for name, digest in manifest['sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen training package mismatch ' + name)
local = out / ('partition' + str(rank))
local.mkdir(parents=True, exist_ok=False)
for name in ('tmp', 'logs'):
    (local / name).mkdir()
(out / 'cases').mkdir(exist_ok=True)
splits = json.loads((ROOT / 'startup_splits.json').read_text())
schedule = [(split, seed, 180 if index % 2 == 0 else 360)
            for split in ('train', 'calibration', 'audit')
            for index, seed in enumerate(splits[split])]
mine = [row for index, row in enumerate(schedule) if index % 3 == rank]
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0',
    TMPDIR=str(local / 'tmp'), PYTHONPATH=':'.join([str(ROOT / 'scripts/agentic'),
        str(ROOT / 'outputs/learning_dependencies'), os.environ.get('PYTHONPATH', '')]))
state = dict(status='starting', rank=rank, completed=[], cloud_calls=0)
child = None
started = time.monotonic()
deadline = started + 10200


def progress(**kw):
    state.update(kw, updated=time.time(), total_elapsed_s=time.monotonic() - started)
    save(local / 'progress.json', state)
    print('STARTUP_TRAIN_PROGRESS ' + json.dumps(state), flush=True)


def run(name, command, target, limit=1200):
    global child
    save(local / 'logs' / (name + '_command.json'), command)
    with (local / 'logs' / (name + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    begin = time.monotonic()
    while child.poll() is None:
        status_path = target.parent / 'progress.json'
        progress(status='training' if name == 'train' else 'collecting', current=name,
            child_pid=child.pid, child_progress=json.loads(status_path.read_text()) if status_path.exists() else None,
            child_elapsed_s=time.monotonic() - begin)
        if time.monotonic() > deadline or time.monotonic() - begin > limit:
            raise TimeoutError('Bounded runtime exceeded: ' + name)
        time.sleep(15)
    if child.returncode or not target.exists():
        raise RuntimeError((local / 'logs' / (name + '.log')).read_text()[-8000:])
    progress(completed=state['completed'] + [name], child_pid=None)


try:
    save(local / 'protocol.json', dict(rank=rank, schedule=mine, splits=splits,
        pilot_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest(), package=manifest,
        cloud_calls=0, maximum_minutes=170,
        scope='Independent action-consequence learning only. Future target and latency absent from inputs.'))
    for split, seed, wait in mine:
        for mode in ['record'] + [f'skill{i}' for i in range(6)] + ['latch', 'pose']:
            dest = out / 'cases' / str(seed) / mode
            command = [sys.executable, str(ROOT / 'scripts/astra_startup_value_v1/collect.py'),
                '--mode', mode, '--seed', str(seed), '--wait-steps', str(wait),
                '--output', str(dest), '--headless', '--device', 'cuda:0']
            if mode != 'record':
                command += ['--source', str(dest.parent / 'record')]
            run(str(seed) + '_' + mode, command + KIT, dest / 'report.json')
    save(local / 'data_completed.json', dict(status='completed', rank=rank, schedule=mine,
        package_sha256=hashlib.sha256((ROOT / 'package_manifest.json').read_bytes()).hexdigest()))
    if rank == 0:
        while True:
            for part in range(3):
                failure = out / ('partition' + str(part)) / 'failure.json'
                if failure.exists():
                    raise RuntimeError('A data partition failed: ' + failure.read_text())
            completed = [out / ('partition' + str(part)) / 'data_completed.json' for part in range(3)]
            if all(path.exists() for path in completed):
                expected = hashlib.sha256((ROOT / 'package_manifest.json').read_bytes()).hexdigest()
                if any(json.loads(path.read_text())['package_sha256'] != expected for path in completed):
                    raise RuntimeError('Mixed packages across partitions')
                break
            progress(status='waiting_for_partitions', current=None, child_pid=None,
                finished_partitions=[i for i, path in enumerate(completed) if path.exists()])
            if time.monotonic() > deadline:
                raise TimeoutError('Partition wait exceeded total time bound')
            time.sleep(30)
        run('train', [sys.executable, str(ROOT / 'scripts/astra_startup_value_v1/train.py'),
            '--data', str(out / 'cases'), '--splits', str(ROOT / 'startup_splits.json'),
            '--output', str(out / 'train')], out / 'train/result.json')
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name in ('head.pt', 'result.json', 'protocol.json', 'history.json', 'frozen_selection.json'):
                tar.add(out / 'train' / name, arcname=name)
        blob = stream.getvalue()
        (out / 'learner_bundle.tar.gz').write_bytes(blob)
        print('STARTUP_LEARNER_SHA256 ' + hashlib.sha256(blob).hexdigest(), flush=True)
        print('STARTUP_LEARNER_BEGIN', flush=True)
        encoded = base64.b64encode(blob).decode()
        for index in range(0, len(encoded), 64):
            print(encoded[index:index+64], flush=True)
        print('STARTUP_LEARNER_END', flush=True)
    progress(status='completed', current=None, child_pid=None)
except BaseException:
    save(local / 'failure.json', dict(error=traceback.format_exc()))
    progress(status='failed', error=traceback.format_exc())
    raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
