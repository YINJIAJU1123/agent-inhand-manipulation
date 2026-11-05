"""Describe completed agent pairs from public call and control logs.

This offline audit never changes actions, raw outcomes or success definitions.
Semantic labels below are evaluator-only annotations of recorded model outputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from summarize import compare, read_run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    index = .9 * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                p90=values[lower] + (values[upper] - values[lower]) * (index - lower),
                maximum=values[-1])


def describe(root):
    run = read_run(root, True)
    if not run['summary']['complete']:
        raise ValueError('Analysis requires all forty physically valid raw trials: ' + str(root))
    manifests = {}
    control = {}
    for fragment in run['summary']['runs']:
        directory = Path(fragment['path'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('formal_trials_used'):
            raise ValueError('This analysis is for development evidence only')
        manifests[str(directory)] = manifest
        counters = {}
        with (directory / 'steps.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                counts = counters.setdefault(row['episode'], Counter())
                counts['logged_steps'] += 1
                counts['pending_steps'] += int(row['pending'])
                counts['unbound_steps'] += int(not row['known'])
        control[str(directory)] = counters
    records = {}
    all_latencies = []
    tokens = Counter()
    calls_with_usage = 0
    for key, row in sorted(run['records'].items()):
        directory = Path(row['source']).parent
        manifest = manifests[str(directory)]
        colors = manifest['goal_manifest']['colors']
        calls_path = directory / f"episode_{row['episode']:03d}" / 'calls.json'
        calls = json.loads(calls_path.read_text())
        if len(calls) != row['actual_model_calls']:
            raise ValueError('Recorded calls differ from episode accounting: ' + key)
        ledger = []
        bindings = []
        for call in calls:
            result = call.get('result') or {}
            applied = call.get('applied_step')
            tool = result.get('tool')
            binding = None
            if manifest['provider'] == 'astra':
                arguments = result.get('arguments', {})
                if applied is not None and tool in ('reorient_target', 'verify_target', 'stop'):
                    binding = dict(color=arguments['color'], symbol=arguments['goal_symbol'])
                evidence = arguments.get('evidence')
            else:
                if applied is not None and result.get('accepted'):
                    binding = dict(color=colors[int(result['skill'])], symbol=row['symbol'])
                evidence = result.get('parsed', {}).get('evidence') if isinstance(result.get('parsed'), dict) else None
                tool = 'accepted_goal' if result.get('accepted') else 'unaccepted_view'
            correct = (binding['color'] == colors[row['goal_index']] and binding['symbol'] == row['symbol']) if binding else None
            if binding is not None:
                bindings.append(dict(step=applied, correct=correct, **binding))
            usage = result.get('usage') or {}
            if usage:
                calls_with_usage += 1
                for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                    if field in usage:
                        tokens[field] += usage[field]
            latency = call['elapsed_s']
            all_latencies.append(latency)
            ledger.append(dict(call=call['call'], request_step=call['request_step'],
                               applied_step=applied, elapsed_s=latency, tool=tool,
                               binding=binding, correct_binding=correct, evidence=evidence,
                               error=call.get('error')))
        correct_bindings = [event for event in bindings if event['correct']]
        records[key] = dict(
            trial_seed=row['trial_seed'], symbol=row['symbol'], success=row['success'],
            reason=row['reason'], failure_subtype=row['failure_subtype'],
            drop_context=row.get('drop_context'), exclusion_reason=row['exclusion_reason'],
            initial_state_sha256=row['initial_state']['sha256'],
            simulated_duration_s=row['steps'] / 30,
            final_error_rad=row['final_error_rad'], minimum_error_rad=row['minimum_error_rad'],
            stop_measurement=row['stop_measurement'],
            pending_simulated_s=control[str(directory)][row['episode']]['pending_steps'] / 30,
            unbound_simulated_s=control[str(directory)][row['episode']]['unbound_steps'] / 30,
            calls=row['actual_model_calls'],
            first_binding=bindings[0] if bindings else None,
            first_correct_binding_step=correct_bindings[0]['step'] if correct_bindings else None,
            wrong_bindings_delivered=sum(not event['correct'] for event in bindings),
            ever_correctly_bound=bool(correct_bindings),
            calls_sha256=sha(calls_path), manifest_sha256=sha(directory / 'manifest.json'),
            public_calls=ledger)
    rows = list(records.values())
    summary = dict(
        success=run['summary']['success'], raw_trials=40,
        failures=run['summary']['failure_contexts'],
        request_latency_s=distribution(all_latencies),
        calls_per_trial=distribution([r['calls'] for r in rows]),
        first_binding_counts=dict(Counter('none' if not r['first_binding'] else
                                         'correct' if r['first_binding']['correct'] else 'wrong' for r in rows)),
        trials_ever_correctly_bound=sum(r['ever_correctly_bound'] for r in rows),
        trials_with_wrong_binding=sum(r['wrong_bindings_delivered'] > 0 for r in rows),
        first_correct_binding_simulated_s=distribution([r['first_correct_binding_step'] / 30 for r in rows if r['first_correct_binding_step'] is not None]),
        pending_simulated_s_per_trial=distribution([r['pending_simulated_s'] for r in rows]),
        successful_simulated_duration_s=distribution([r['simulated_duration_s'] for r in rows if r['success']]),
        recorded_provider_token_usage=dict(tokens) if calls_with_usage else None,
        calls_with_reported_token_usage=calls_with_usage,
        per_symbol={symbol: dict(raw_trials=sum(r['symbol'] == symbol for r in rows),
                                 success=sum(r['symbol'] == symbol and r['success'] for r in rows))
                    for symbol in sorted({r['symbol'] for r in rows})})
    return run, records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a, ar, sa = describe(args.a)
    b, br, sb = describe(args.b)
    pairing = compare(a, b)
    if pairing['mismatched_trial_seeds']:
        raise ValueError('The comparison does not share exact initial states')
    discordant = {key: dict(a=ar[key], b=br[key]) for key in ar if ar[key]['success'] != br[key]['success']}
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope='Completed actual-model development pair. Descriptive mechanism and cost audit, not formal evidence or causal attribution.',
                  analysis_source_sha256=sha(Path(__file__)),
                  conditions=dict(a=args.a.name, b=args.b.name),
                  summaries=dict(a=sa, b=sb), pairing=pairing, discordant_trials=discordant,
                  cautions=[
                      'Same initial state does not guarantee the same stochastic physical trajectory or model responses.',
                      'Recorded request latency is caller wall time. Shared GPU load prevents interpreting simulated duration as real-time throughput.',
                      'Binding and successful-duration summaries are conditional descriptive quantities, not alternative success metrics.',
                      'Provider token fields are recorded API response usage, not an account allowance or a monetary estimate.',
                      'Raw paired comparisons retain all recorded exclusions. Inspect exclusion fields before eligible comparisons.'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary', report), ('trials', dict(a=ar, b=br))]:
        temporary = args.output / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(args.output / (name + '.json'))
    print(json.dumps(dict(summaries=report['summaries'], discordant_trials=list(discordant))))


if __name__ == '__main__':
    main()
