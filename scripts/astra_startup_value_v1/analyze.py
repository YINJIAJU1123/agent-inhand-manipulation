"""Summarize a complete startup action pilot without fitting or deployment."""
import argparse
import json
from pathlib import Path

ARMS = [f'skill{i}' for i in range(6)] + ['latch']


def analyze(root):
    root = Path(root)
    groups = []
    for record in sorted(root.glob('*/record/report.json')):
        parent = record.parent.parent
        reports = {m: json.loads((parent / m / 'report.json').read_text()) for m in ARMS + ['pose']}
        expected = json.loads(record.read_text())['decision_inputs_sha256']
        for mode, report in reports.items():
            audit = json.loads((parent / mode / 'initial_audit.json').read_text())
            if report['status'] != 'completed' or audit['decision_inputs_sha256'] != expected:
                raise RuntimeError('Incomplete or unmatched comparison')
            if any(v != 0 for v in audit['physical_errors'].values()):
                raise RuntimeError('Initial physical mismatch')
        groups.append(reports)
    if not groups:
        raise RuntimeError('No complete groups')
    totals = {m: {k: sum(g[m][k] for g in groups) for k in
        ('cases', 'completed', 'drops', 'startup_drops', 'post_handoff_drops')} for m in ARMS + ['pose']}
    n = totals['latch']['cases']
    oracle = sum(any(g[m]['episodes'][i]['completed'] for m in ARMS)
                 for g in groups for i in range(g['latch']['cases']))
    best = max(ARMS, key=lambda m: (totals[m]['completed'], -totals[m]['drops']))
    return dict(status='completed', nominal_states=n, by_mode=totals,
        descriptive_best_fixed=best, oracle_completed=oracle,
        oracle_gain_over_best_fixed=oracle - totals[best]['completed'],
        opportunity_gate_passed=n >= 12 and oracle - totals[best]['completed'] >= 2,
        cloud_calls=0,
        note='Oracle uses future labels and is not a policy. Pilot is development only. '
             'Mixed per-case choices still require separate closed-loop validation. '
             'This gate only permits independently split training; it never promotes a controller.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.data)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
