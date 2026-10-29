"""Read one frozen suite snapshot without launching, retrying or altering trials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite = args.suite.resolve()
    launch = json.loads((suite / 'launch.json').read_text())
    issues = []

    def read_json(path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            issues.append(dict(path=str(path),kind=type(error).__name__))
            return None

    def status(path):
        return path.read_text().strip() if path.exists() else None

    def process(pid, marker='run_formal_suite.sh'):
        root = Path('/proc') / str(pid)
        try:
            state = next(line.split(':', 1)[1].strip() for line in
                         (root / 'status').read_text().splitlines() if line.startswith('State:'))
            command = (root / 'cmdline').read_bytes()
            return dict(pid=pid,live=not state.startswith('Z'),state=state,
                        expected_launcher=marker in command.decode(errors='replace'))
        except (OSError, StopIteration):
            return dict(pid=pid,live=False)

    def complete_lines(path, tail=False):
        with path.open('rb') as stream:
            if tail:
                stream.seek(max(0, path.stat().st_size - 65536))
            data = stream.read()
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            lines.pop()  # The writer may still be appending this record.
        if tail:
            lines = lines[-1:]
        return [json.loads(line) for line in lines]

    report = dict(at=datetime.now(timezone.utc).isoformat(),suite=str(suite),
        scope='Live monitoring only. Final metric and pairing verification remain in summarize_formal.py.',
        release_sha256=launch['release_sha256'],planned_raw_trials=launch['planned_raw_trials'],
        queues={str(gpu):process(pid) for gpu,pid in launch['queues'].items()},
        suite_terminal=status(suite / 'process_status.txt'),conditions={})
    for name, condition in launch['conditions'].items():
        root = suite.parent / condition['root']
        prefix = condition['prefix']
        item = dict(exists=root.exists(),recorded=0,scope=condition['scope'])
        if root.exists():
            rows = [row for path in root.glob(prefix + '*/episodes.jsonl') for row in complete_lines(path)]
            item.update(recorded=len(rows),unique_trial_seeds=len({r['trial_seed'] for r in rows}),
                service_exclusions=sum(bool(row.get('exclusion_reason')) for row in rows),
                stage=status(root / 'current_stage.txt'),terminal=status(root / 'process_status.txt'))
            item['completed_physics_checks'] = {}
            for path in root.glob(prefix + '*.validity.json'):
                check = read_json(path)
                if check is not None:
                    item['completed_physics_checks'][path.stem] = {
                        key:check.get(key) for key in ('exit_code','physical_log_valid','capacity_warning')}
            steps = list(root.glob(prefix + '*/steps.jsonl'))
            if steps:
                path = max(steps,key=lambda p:p.stat().st_mtime)
                tail = complete_lines(path,tail=True)
                if tail:
                    item.update(latest_episode=tail[-1].get('episode'),latest_step=tail[-1].get('step'),
                                latest_step_file_age_s=round(time.time()-path.stat().st_mtime,2))
        report['conditions'][name] = item
    report['recoveries'] = {}
    for path in sorted((suite/'recovery_launches').glob('*.json')):
        recovery = read_json(path)
        if recovery is None:continue
        directory = Path(recovery['directory']).resolve()
        if not directory.is_relative_to(suite.parent):
            raise ValueError('Recovery directory is outside this output workspace')
        report['recoveries'][path.stem] = dict(
            process=process(recovery['pid'],recovery['command_marker']),
            original_queue=recovery['original_queue'],conditions=recovery['conditions'],
            stage=status(directory/'current_stage.txt'),terminal=status(directory/'process_status.txt'),
            preserved_service_exclusion=recovery['preserved_service_exclusion'],
            completion=read_json(directory/'recovery_completion.json'))
    report['recorded_raw_trials'] = sum(item['recorded'] for item in report['conditions'].values())
    report['read_issues'] = issues
    rendered = json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix+'.tmp')
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered,end='')


if __name__ == '__main__':
    main()
