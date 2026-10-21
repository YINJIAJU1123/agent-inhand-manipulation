"""Finite local cache + two matched model fits. Does not start cloud evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.append(str(Path(__file__).resolve().parents[1]/'semantic_completion_v2'))
from assets import ROOT, PYTHON, STUDENT, ADAPTER, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    data = Path(args.data).resolve()
    summary = json.loads((data.parent/'collection_summary.json').read_text())
    if summary['status'] != 'collection_completed_and_audited' or summary['sensor_audit']['shards_passed'] != 10:
        raise RuntimeError('Collection audit required before training')
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    source_paths = list((ROOT/'scripts/semantic_completion_v3').glob('*.py'))
    source_paths += [ROOT/'scripts/semantic_completion_v2'/name for name in ('assets.py', 'features.py', 'sequential.py')]
    source_paths += [ROOT/'scripts/semantic_completion_v1/common.py', ROOT/'source/BrainCo_DexHand/BrainCo_DexHand/algo/agentic/visual_student.py']
    frozen = {str(path): sha(path) for path in source_paths}
    assets = {str(path): sha(path) for path in (STUDENT, ADAPTER)}
    for path in source_paths:
        dest = out/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    save(out/'protocol.json', dict(data=str(data), epochs=args.epochs, source_sha256=frozen,
        assets_sha256=assets, collection_summary_sha256=sha(data.parent/'collection_summary.json'),
        variants=['compact', 'dense'], formal_seeds_untouched=[86101, 86102],
        purpose='Matched local dense RGB-D evidence pilot. No API traffic, no production checkpoint replacement.',
        cloud_calls=0, formal_astra_episodes=0))
    state = dict(pid=os.getpid(), status='starting', completed=[])

    def progress(**kwargs):
        state.update(kwargs, updated=time.time())
        save(out/'progress.json', state)
        current = dict(updated=state['updated'], status=state['status'], active_pipeline=str(out),
            model_training_active=state.get('stage') in ('compact', 'dense') and bool(state.get('child_pid')),
            cloud_calls=0, formal_astra_episodes=0, cloud_transfer_confirmation='pending',
            collection=str(data.parent), collection_completed=True, current_candidate=None)
        save(ROOT/'outputs/semantic_completion_v3/current.json', current)

    def interrupt(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    lock = Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
    child = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
        jobs = [('cache', [PYTHON, str(Path(__file__).with_name('cache_dense.py')), '--data', str(data), '--output', str(out/'cache')])]
        for variant in ('compact', 'dense'):
            jobs.append((variant, [PYTHON, str(Path(__file__).with_name('train_dense.py')), '--cache', str(out/'cache'),
                '--output', str(out/variant), '--variant', variant, '--epochs', str(args.epochs)]))
        for name, command in jobs:
            for path, digest in {**frozen, **assets}.items():
                if sha(path) != digest:
                    raise RuntimeError(f'Frozen source/asset changed: {path}')
            with (out/'logs'/f'{name}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                while child.poll() is None:
                    progress(status='caching' if name == 'cache' else 'training', stage=name,
                             child_pid=child.pid, stage_elapsed_s=time.monotonic()-started)
                    if time.monotonic()-started > 5400:
                        raise TimeoutError(f'{name} exceeded 90 minute bound')
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'{name} exited {child.returncode}, see {out/"logs"/f"{name}.log"}')
                progress(completed=state['completed']+[name], child_pid=None)
        results = {variant: json.loads((out/variant/'result.json').read_text()) for variant in ('compact', 'dense')}
        save(out/'comparison.json', dict(variants=results, cloud_calls=0, formal_astra_episodes=0,
            note='These are offline development first-stop results. Live shared-trajectory validation must precede promotion.'))
        progress(status='completed', child_pid=None, stage=None, next_stage='review_then_live_shadow_validation')
    except BaseException as exc:
        save(out/'failure.json', dict(error=traceback.format_exc()))
        progress(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc), child_pid=None)
        raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        lock.close()


if __name__ == '__main__':
    main()
