"""Bounded trajectory recording and exact-state label replay on one GPU."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
out = Path(os.environ['TRAJECTORY_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
for name in ['tmp', 'logs', 'data', 'rollouts']:
    (out/name).mkdir()
manifest = json.loads((ROOT/'package_manifest.json').read_text())
for name, digest in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
schedule = json.loads(os.environ['TRAJECTORY_SCHEDULE'])
max_steps = int(os.environ.get('TRAJECTORY_STEPS', '1800'))
views = int(os.environ.get('TRAJECTORY_VIEWS', '6'))
assert 1 <= len(schedule) <= 4 and 900 <= max_steps <= 1800 and 1 <= views <= 6
state = dict(status='starting', completed=[], cloud_calls=0)
child = None
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0',
    TMPDIR=str(out/'tmp'), PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'), str(ROOT/'outputs/learning_dependencies'), os.environ.get('PYTHONPATH', '')]))


def progress(**values):
    state.update(values, updated=time.time())
    save(out/'progress.json', state)
    print('TRAJECTORY_PROGRESS '+json.dumps(state), flush=True)


def run(name, command, report):
    global child
    save(out/'logs'/(name+'_command.json'), command)
    with (out/'logs'/(name+'.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    start = time.time()
    while child.poll() is None:
        p = report.parent/'progress.json'
        measured = json.loads(p.read_text()) if p.exists() else None
        progress(status='collecting', current=name, child_pid=child.pid, child_progress=measured, elapsed_s=time.time()-start)
        failure = report.parent/'failure.json'
        if failure.exists():
            raise RuntimeError(failure.read_text())
        if time.time()-start > 2400:
            raise TimeoutError(name)
        time.sleep(15)
    if child.returncode or not report.exists():
        raise RuntimeError((out/'logs'/(name+'.log')).read_text()[-12000:])
    progress(completed=state['completed']+[name], child_pid=None)


try:
    save(out/'protocol.json', dict(schedule=schedule, max_steps=max_steps, views=views, package=manifest,
        cloud_calls=0, scope='Fixed-public-color control data. Exact physics replay with training-only symbol variants. No semantic success claim.'))
    subprocess.run([sys.executable, '-c', 'import isaaclab,toml,torch; print("CLOUD_CHILD_IMPORT_OK",flush=True)'], check=True, env=env)
    results = []
    for entry in schedule:
        seed, split, mode = entry['seed'], entry['split'], entry['mode']
        assert split in ('train', 'calibration', 'audit') and mode in ('none', 'all')
        name = f'{split}_{seed}'
        rollout, data = out/'rollouts'/name, out/'data'/name
        reuse = os.environ.get('TRAJECTORY_REUSE_ROLLOUT_ROOT')
        if reuse:
            rollout = Path(reuse)/name
            prior = json.loads((rollout/'protocol.json').read_text())
            assert prior['seed']==seed and prior['split']==split and prior['mode']==mode and prior['max_steps']==max_steps
            assert prior['source_sha256']['rollout.py']==hashlib.sha256((ROOT/'scripts/astra_symbol_trajectory_v1/rollout.py').read_bytes()).hexdigest()
            assert json.loads((rollout/'report.json').read_text())['status']=='completed'
            save(out/'rollouts'/(name+'_reuse.json'), dict(source=str(rollout), verified_complete=True))
        else:
            run(name+'_rollout', [sys.executable, str(ROOT/'scripts/astra_symbol_trajectory_v1/rollout.py'),
                '--seed', str(seed), '--split', split, '--mode', mode, '--output', str(rollout), '--max-steps', str(max_steps),
                '--headless', '--device', 'cuda:0']+KIT, rollout/'report.json')
        run(name+'_render', [sys.executable, str(ROOT/'scripts/astra_symbol_trajectory_v1/render.py'),
            '--source', str(rollout), '--output', str(data), '--views', str(views),
            '--headless', '--device', 'cuda:0']+KIT, data/'manifest.json')
        result = json.loads((data/'manifest.json').read_text())
        assert result['views'] > 0 and result['readable'] > 0 and result['replay_max_error'] <= 1e-6
        results.append(result)
        save(out/'partial_results.json', results)
        print('TRAJECTORY_SHARD '+json.dumps(result), flush=True)
        payload = (data/'label_preview.jpg').read_bytes()
        assert len(payload) < 1000000
        print('TRAJECTORY_PREVIEW_SHA256 '+name+' '+hashlib.sha256(payload).hexdigest(), flush=True)
        print('TRAJECTORY_PREVIEW_BEGIN '+name, flush=True)
        encoded = base64.b64encode(payload).decode()
        for i in range(0, len(encoded), 64):
            print(encoded[i:i+64], flush=True)
        print('TRAJECTORY_PREVIEW_END '+name, flush=True)
    save(out/'result.json', dict(status='completed', shards=results, cloud_calls=0))
    progress(status='completed', current=None, child_pid=None)
except BaseException as error:
    save(out/'failure.json', dict(error=traceback.format_exc()))
    progress(status='failed', error=str(error))
    raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
