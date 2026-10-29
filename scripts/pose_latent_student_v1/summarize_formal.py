"""Export a completed frozen suite, preserving raw and matched-eligible results.

This post-processing entrypoint never launches trials, selects weights, retries
requests, or changes episode success. Development records are not accepted.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True,
                        help='Completed suite directory containing launch.json and formal_release.json')
    parser.add_argument('--source_directory', type=Path, required=True,
                        help='Evaluator source snapshot bound into the frozen release')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_directory.resolve()))
    from formal_release import FORMAL_CONDITIONS, FORMAL_SEEDS, digest, validate_contracts, verify_runtime
    from summarize import compare, read_run, wilson_interval

    suite = args.suite.resolve()
    release_path = suite / 'formal_release.json'
    release = load_json(release_path)
    launch = load_json(suite / 'launch.json')
    validate_contracts(release)
    release_sha = digest(release_path)
    if launch['release_sha256'] != release_sha or set(launch['conditions']) != set(FORMAL_CONDITIONS):
        raise ValueError('Launch does not match the complete frozen release')
    sources = {path.name: digest(path) for path in sorted(args.source_directory.glob('*.py'))}
    if sources != release['sources_sha256']:
        raise ValueError('Analysis helpers must match the frozen evaluator source snapshot')
    expected = {str(seed * 1000 + i) for seed in FORMAL_SEEDS for i in range(20)}
    data, diagnostics, manifests, raw_trials = {}, {}, {}, []
    roots = suite.parent

    for name in FORMAL_CONDITIONS:
        item = launch['conditions'][name]
        root = roots / item['root']
        prefix = item['prefix']
        diagnostic = name in FORMAL_CONDITIONS[:2]
        kind = 'diagnostic' if diagnostic else 'agent'
        paths = sorted(root.glob(prefix + '*/episodes.jsonl'))
        actual_sources = {path.name: digest(path) for path in sorted((root / 'sources').glob('*.py'))}
        if actual_sources != sources:
            raise ValueError('Saved evaluator source mismatch: ' + name)
        diagnostic_rows = {}
        for path in paths:
            run = path.parent
            manifest = load_json(run / 'manifest.json')
            result = load_json(run / 'result.json')
            validity = load_json(run.with_suffix('.validity.json'))
            verify_runtime(release, name, manifest, kind=kind)
            if (manifest.get('formal_trials_used') is not True
                    or manifest.get('formal_condition') != name
                    or manifest.get('formal_release_sha256') != release_sha
                    or result.get('formal_condition') != name
                    or result.get('formal_release_sha256') != release_sha
                    or result.get('formal_' + ('diagnostic' if diagnostic else 'agent') + '_result') is not True
                    or digest(run / 'formal_release.json') != release_sha
                    or validity.get('exit_code') != 0
                    or validity.get('physical_log_valid') is not True):
                raise ValueError('Unverified formal run: ' + str(run))
            lines = path.read_text().splitlines(keepends=True)
            if any(not line.endswith('\n') for line in lines):
                raise ValueError('Truncated episode record: ' + str(path))
            rows = [json.loads(line) for line in lines]
            if result['recorded_episodes'] != len(rows):
                raise ValueError('Recorded count disagrees with raw records: ' + str(run))
            manifests[name + ':' + run.name] = dict(
                manifest=manifest, manifest_sha256=digest(run / 'manifest.json'),
                episodes_sha256=digest(path), result_sha256=digest(run / 'result.json'),
                validity_sha256=digest(run.with_suffix('.validity.json')))
            if diagnostic:
                if result['status'] != 'completed' or result['requested_episodes'] != len(rows):
                    raise ValueError('Incomplete policy diagnostic: ' + str(run))
                for row in rows:
                    key = str(row['trial_seed'])
                    if key in diagnostic_rows:
                        raise ValueError('Duplicate diagnostic trial: ' + key)
                    diagnostic_rows[key] = row
                    raw_trials.append(dict(row, condition=name, scope='low_level_diagnostic'))
        if diagnostic:
            if set(diagnostic_rows) != expected:
                raise ValueError('Diagnostic does not contain the forty reserved trial slots: ' + name)
            rows = list(diagnostic_rows.values())
            reached = sum(row['reached_before_drop'] for row in rows)
            diagnostics[name] = dict(raw_trials=40, reached_before_drop=reached,
                reach_before_drop_rate=reached / 40, wilson_95=wilson_interval(reached, 40),
                drops=sum(row['dropped'] for row in rows),
                initially_reached=sum(row['initially_reached'] for row in rows),
                reasons=dict(Counter(row['reason'] for row in rows)),
                metric_scope='Physical reach-before-drop under the released environment horizon. Not autonomous semantic success.')
        else:
            class ConditionRoot:
                def glob(self, pattern):
                    return root.glob(prefix + '*/episodes.jsonl')
            current = read_run(ConditionRoot(), True)
            if not current['summary']['complete'] or set(current['records']) != expected:
                raise ValueError('Semantic condition does not contain forty valid reserved slots: ' + name)
            for row in current['records'].values():
                if row.get('exclusion_reason') not in (None, 'service_error', 'request_latency_over_120s'):
                    raise ValueError('Unregistered service exclusion')
                exported = {key: value for key, value in row.items() if key != 'source'}
                raw_trials.append(dict(exported, condition=name, scope='autonomous_semantic_agent'))
            data[name] = current

    def matched_group(names, contrasts):
        excluded = {key for key in expected if any(
            data[name]['records'][key].get('exclusion_reason') is not None for name in names)}
        eligible = expected - excluded
        outcomes = {}
        for name in names:
            successes = sum(data[name]['records'][key]['success'] for key in eligible)
            outcomes[name] = dict(success=successes, eligible_trials=len(eligible),
                success_rate=successes / len(eligible) if eligible else None,
                wilson_95=wilson_interval(successes, len(eligible)))
        pairs = {}
        for a, b in contrasts:
            raw = compare(data[a], data[b])
            if raw['mismatched_trial_seeds']:
                raise ValueError('Compared conditions have mismatched initial states or goals: ' + a + ' / ' + b)
            paired = compare(data[a], data[b], eligible_keys=eligible)
            paired['exact_mcnemar_p_unadjusted'] = paired.pop('raw_exact_mcnemar_p')
            pairs[a + '__' + b] = dict(raw=raw, matched_eligible=paired)
        return dict(conditions=list(names), raw_trials_per_condition=40,
            excluded_trial_seeds=sorted(excluded), eligible_trials=len(eligible),
            exclusion_reasons={key: {name: data[name]['records'][key]['exclusion_reason']
                for name in names if data[name]['records'][key]['exclusion_reason'] is not None}
                for key in sorted(excluded)}, outcomes=outcomes, contrasts=pairs)

    full, fine, rule = 'astra_full', 'finetuned_vlm_full', 'rule_vlm_full'
    base = 'fixed_slow_no_execution_recovery'
    wait = 'learned_waiting_only'
    execute = 'learned_execution_only_with_fixed_slow_waiting'
    providers = matched_group((full, fine, rule), ((full, fine), (full, rule)))
    factorial = matched_group((full, base, wait, execute),
                             ((full, base), (full, wait), (full, execute), (wait, base), (execute, base)))
    if len(raw_trials) != 320:
        raise ValueError('Expected exactly 320 raw trial records')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed frozen pose-latent suite on reserved seeds. Cube scene only.',
        formal_release_sha256=release_sha, analysis_source_sha256=digest(Path(__file__)),
        complete=True, raw_trials=320, raw_trials_per_condition=40,
        semantic_conditions={name: item['summary'] for name, item in data.items()},
        low_level_diagnostics=diagnostics, provider_comparison=providers,
        astra_waiting_execution_factorial=factorial,
        statistics='Wilson intervals and two-sided exact paired McNemar tests. P values are unadjusted across the listed contrasts and do not automatically establish model superiority.',
        exclusions='Symmetric group-wide service exclusions only. The provider triplet and Astra factorial have separately stated eligible sets. All raw forty-trial outcomes are retained.',
        limitations='Matched initial states are not a guarantee of bitwise physical trajectories. Low-level reach is not semantic completion. No multi-object or real-hardware evidence is implied.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary', report), ('trials', raw_trials), ('manifests', manifests)]:
        write_json(args.output / (name + '.json'), value)
    print(json.dumps(dict(complete=True, raw_trials=320,
        semantic_success={name: item['summary']['success'] for name, item in data.items()},
        output=str(args.output))))


if __name__ == '__main__':
    main()
