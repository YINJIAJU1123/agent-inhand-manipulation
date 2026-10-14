"""Audit and summarize completed v4 reports; missing modes remain not run."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory')
a = p.parse_args()
root = Path(a.directory)
reports = []
for path in sorted(root.glob('*/report.json')):
    report = json.loads(path.read_text())
    mode = report['metadata']['modes'][0]
    seed = report['metadata']['seeds'][0]
    initial = json.loads((path.parent / f'initial_{seed}_{mode}.json').read_text())
    assert len(report['episodes']) == 6
    assert {e['case'] for e in report['episodes']} == set(range(6))
    assert all(e['calls'] <= report['metadata']['call_budget'] for e in report['episodes'])
    reports.append((path, mode, seed, report, initial))
if not reports:
    raise RuntimeError('No completed evaluation reports')
pairing = {}
for seed in sorted({r[2] for r in reports}):
    group = [r for r in reports if r[2] == seed]
    pairing[str(seed)] = {'modes': [r[1] for r in group],
                          'initial_states_equal': all(r[4] == group[0][4] for r in group)}
fixed = ['checkpoint_sha256', 'adapter_sha256', 'call_budget', 'max_steps', 'source_sha256']
shared = {name: all(r[3]['metadata'][name] == reports[0][3]['metadata'][name] for r in reports) for name in fixed}
summary = {'benchmark': 'two_track_semantic_agent_v4_development', 'pairing': pairing,
           'shared_protocol': shared, 'modes': {}, 'limits': 'Small development evaluation; no statistical superiority or autonomous completion claim.'}
for mode in ['latent', 'oracle', 'astra']:
    episodes = [e for r in reports if r[1] == mode for e in r[3]['episodes']]
    if not episodes:
        summary['modes'][mode] = {'status': 'not_run'}
        continue
    n = len(episodes)
    metrics = {key: sum(e[key] for e in episodes) for key in ['success', 'grounding_correct', 'drop', 'calls', 'stop_claimed', 'false_stop']}
    metrics.update(status='completed', episodes=n, success_rate=metrics['success']/n)
    metrics['by_symbol'] = {symbol: {'episodes': sum(e['target_symbol'] == symbol for e in episodes),
                                    'success': sum(e['success'] for e in episodes if e['target_symbol'] == symbol)}
                            for symbol in sorted({e['target_symbol'] for e in episodes})}
    summary['modes'][mode] = metrics
summary['reports'] = [str(r[0].relative_to(root)) for r in reports]
(root / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
assert all(shared.values()), 'Evaluation source or protocol drift'
assert all(x['initial_states_equal'] for x in pairing.values()), 'Unpaired initial states'
