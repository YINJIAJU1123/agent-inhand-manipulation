"""Compare every bilingual execution with its object-specific paper release.

Read-only with respect to all source cohorts. Report deployment/configuration
differences separately from outcomes; this is not a causal attribution test.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/multilingual_eval_20261104_v2'
CUBE = ROOT / 'outputs/pose_latent_student_v1/refit_bare_comparison_20261031_v1/completed_snapshot'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific(value):
    """Ignore relocation paths, retain their recorded content hashes."""
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items()
                if k not in ('path', 'checkpoint', 'source_environment')}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value


def signature(d, obj):
    if obj == 'cube':
        fields = ('student_sha256', 'original_student_sha256', 'physical_binding',
                  'pose_estimator', 'camera', 'goal_manifest', 'explicit_task_overrides',
                  'maintenance', 'actor_sha256', 'execution_actor_sha256',
                  'completion_sha256', 'completion_kind', 'stop_mode',
                  'completion_during_wait', 'measured_stop_threshold_rad', 'request_budget')
        result = {k: scientific(d.get(k)) for k in fields}
        result['visual_mode'] = d.get('visual_intervention', {}).get('mode', 'native')
        result['agent_protocol'] = d.get('agent_protocol', 'feedback')
    else:
        fields = ('maintenance', 'actor_sha256', 'stop_sha256', 'stop_mode',
                  'stop_hz', 'call_limit', 'semantic_sha256')
        result = {k: d.get(k) for k in fields}
        for k in ('student_sha256', 'source_checkpoint_sha256', 'physical_binding',
                  'goal_manifest', 'noise_config', 'explicit_overrides', 'camera',
                  'visual_override', 'pose_estimator_used', 'pose_source'):
            result['runtime.' + k] = scientific(d['runtime'].get(k))
    for k in ('checkpoint_sha256', 'adapter_sha256', 'backend', 'frame_size', 'quantize'):
        result['upstream.' + k] = d['upstream'].get(k)
    return result


def references(obj, method):
    if obj == 'cube':
        condition = dict(astra='astra_full', rule='rule_vlm_bare', vlm='finetuned_vlm_bare')[method]
        return sorted((CUBE / 'runs' / condition).glob('*/manifest.json'))
    return sorted((ROOT / 'outputs/q1_inhand4_20261102' / obj / 'formal_v1'
                   / ('identity_' + method)).glob('**/manifest.json'))


def outcomes(rows):
    latencies = [x for r in rows for x in r.get('latencies_s', []) if isinstance(x, (int, float))]
    return dict(recorded=len(rows), success=sum(bool(r['success']) for r in rows),
                reasons=dict(Counter(r['reason'] for r in rows)),
                drop_phases=dict(Counter(r.get('drop_phase') for r in rows if r['reason'] == 'drop')),
                call_latency_median_s=statistics.median(latencies) if latencies else None,
                calls_with_recorded_latency=len(latencies))


def audit():
    cells = {}
    reference_report = {}
    for obj in ('cube', 'duck', 'hedgehog'):
        for method in ('astra', 'rule', 'vlm'):
            refs = references(obj, method)
            if not refs:
                raise ValueError(f'Missing paper reference: {obj}/{method}')
            ref = signature(read(refs[0]), obj)
            if any(signature(read(p), obj) != ref for p in refs):
                raise ValueError(f'Inconsistent reference configurations: {obj}/{method}')
            ref_rows = [json.loads(line) for p in refs for line in
                        (p.parent / 'episodes.jsonl').read_text().splitlines()]
            reference_report[f'{obj}/{method}'] = dict(
                manifests={str(p.relative_to(ROOT)): digest(p) for p in refs},
                signature=ref, outcomes=outcomes(ref_rows))
            for condition in ('en_standard', 'en_rewrite', 'zh_standard', 'zh_rewrite'):
                cell = BASE / 'runs' / obj / condition / method
                paths = sorted(cell.glob('**/evaluation/manifest.json'))
                rows, differences, transport = [], {}, Counter()
                for p in paths:
                    sig = signature(read(p), obj)
                    diff = {k: dict(reference=ref[k], language=sig[k]) for k in ref if ref[k] != sig[k]}
                    differences[str(p.relative_to(ROOT))] = diff
                    rows += [json.loads(line) for line in (p.parent / 'episodes.jsonl').read_text().splitlines()]
                for p in cell.glob('**/requests/request_*.json'):
                    d = read(p)
                    if d.get('status') == 'completed':
                        transport[(d.get('api_base_url', 'not_recorded'), d.get('auth_method', 'not_recorded'),
                                   d.get('model', 'not_recorded'))] += 1
                if len(rows) != 60 or len({r.get('index', r.get('episode')) for r in rows}) != 60:
                    raise ValueError(f'Incomplete or duplicate language panel: {cell}')
                cells[f'{obj}/{condition}/{method}'] = dict(
                    manifests={str(p.relative_to(ROOT)): digest(p) for p in paths},
                    differences=differences, outcomes=outcomes(rows),
                    observed_transport=[dict(endpoint=k[0], auth_method=k[1], returned_model=k[2], calls=n)
                                        for k, n in sorted(transport.items())],
                    configuration_matches=not any(differences.values()))
    return dict(status='configuration_mismatch_confirmed', references=reference_report, cells=cells,
                mismatched_cells=[k for k, v in cells.items() if not v['configuration_matches']],
                scope='Recorded deployment signatures across all 36 cells; not complete historical source recovery or a causal explanation of outcome changes',
                conclusions=[
                    'Cube Astra has the wrong waiting actor, omits the separate execution actor, and uses a learned pose-correction gate instead of the paper measured gate.',
                    'The executor reuses the waiting actor during execution when no execution actor is supplied; execution was not simply disabled.',
                    'Duck/hedgehog use their own correct learned stop gates. Do not replace them with the cube measured protocol.',
                    'Native cube language images align with the native-image release, not the reconstructed identity mapping benchmark.',
                    'Gateway transport, inference hardware, concurrency, trial seeds and wording also differ between historical cohorts; repair cannot promise 35/40 performance.',
                    'Minimum configuration repair is 240 cube Astra trials. User selected a new complete 720-trial cube comparison with contemporaneous baselines.',
                ])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], cells=len(result['cells']),
                          mismatched_cells=result['mismatched_cells']), ensure_ascii=False))
