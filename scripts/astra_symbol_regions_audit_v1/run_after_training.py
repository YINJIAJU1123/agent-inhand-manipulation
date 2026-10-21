"""Bounded dependency runner for one already-submitted job; no new jobs or Astra calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BCTRL = ['/home/yin/.local/bin/bctrl', '--region', 'cn-wulanchabu', 'dlc']


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-source', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    child = None
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ['retrieve.py', 'evaluate.py']}
    state = dict(status='waiting_for_training', pid=os.getpid(), job_id=args.job_id, astra_calls=0)
    save(args.output / 'protocol.json', dict(job_id=args.job_id, cache=str(args.cache), script_sha256=hashes,
         max_wait_seconds=10800, poll_seconds=45, checkpoint_selection='Use the training job calibration-selected checkpoint exactly once.',
         intent='Retrieve a completed model and evaluate measured RGB from recorded requests. Never submit, cancel, restart or deploy a cloud job. No new Astra requests.'))

    def progress(**values):
        state.update(values, updated=time.time())
        save(args.output / 'progress.json', state)
        print(json.dumps(state), flush=True)

    def run(script, arguments, timeout):
        nonlocal child
        if hashlib.sha256((HERE / script).read_bytes()).hexdigest() != hashes[script]:
            raise RuntimeError('Frozen evaluation source changed')
        with (args.output / (script + '.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, str(HERE / script)] + arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        beginning = time.monotonic()
        while child.poll() is None:
            progress(status='retrieving' if script == 'retrieve.py' else 'shadowing', child_pid=child.pid, child_elapsed_s=time.monotonic() - beginning)
            if time.monotonic() - beginning > timeout:
                raise TimeoutError(script + ' exceeded its deadline')
            time.sleep(5)
        if child.returncode:
            raise RuntimeError((args.output / (script + '.log')).read_text()[-6000:])
        progress(child_pid=None)

    try:
        while True:
            if time.monotonic() - started > 10800:
                raise TimeoutError('Bounded watcher ended; cloud task status is not inferred from timeout')
            try:
                response = subprocess.run(BCTRL + ['show', args.job_id, '--json'], capture_output=True, text=True, timeout=40, check=True)
                data = json.loads(response.stdout)
            except (subprocess.SubprocessError, json.JSONDecodeError) as error:
                progress(observation_error=type(error).__name__, status='waiting_for_training')
                time.sleep(45)
                continue
            status = data['Status']
            progress(status='waiting_for_training', job_status=status, observation_error=None)
            if status == 'Succeeded':
                break
            if status in ('Failed', 'Stopped', 'Cancelled', 'Canceled'):
                progress(status='not_started', reason='Training job ended: ' + status)
                return
            response = subprocess.run(BCTRL + ['logs', args.job_id, '--no-follow', '--lines', '300'], capture_output=True, text=True, timeout=45)
            if response.returncode == 0:
                text = re.sub(r'\x1b\[[0-9;]*m', '', response.stdout)
                (args.output / 'latest_job.log').write_text(text)
                latest = None
                for part in text.split('SYMBOL_REGION_PROGRESS')[1:]:
                    try:
                        latest = json.JSONDecoder().raw_decode(''.join(part.splitlines()).lstrip())[0]
                    except ValueError:
                        pass
                if latest:
                    progress(job_progress=latest)
            time.sleep(45)
        run('retrieve.py', ['--job-id', args.job_id, '--output', str(args.output / 'model')], 180)
        extra = ['--model-source', str(args.model_source.resolve())] if args.model_source else []
        run('evaluate.py', ['--checkpoint', str(args.output / 'model/head.pt'), '--cache', str(args.cache), '--output', str(args.output / 'shadow')] + extra, 600)
        result = json.loads((args.output / 'shadow/result.json').read_text())
        save(args.output / 'result.json', {key: value for key, value in result.items() if key != 'rows'})
        progress(status='completed', shadow_diagnostic_passed=result['shadow_diagnostic_passed'], cloud_eligible=False, elapsed_s=time.monotonic() - started)
    except BaseException as error:
        progress(status='failed', error=str(error))
        save(args.output / 'failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)


if __name__ == '__main__':
    main()
