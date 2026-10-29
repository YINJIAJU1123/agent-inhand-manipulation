"""Export completed pose-latent development conditions for the paper.

Raw trial slots and negative variants are preserved. This does not select a
method, change scoring, or export partial conditions as final results.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from summarize import read_run

TRIAL_FIELDS = (
    'trial_seed','episode','goal_index','symbol','reason','success','exclusion_reason',
    'failure_subtype','drop_context','declared_goal','declared_symbol','steps',
    'first_skill_delivery_step','minimum_error_rad','final_error_rad','stop_measurement',
    'actual_model_calls','latencies_s',
)
MANIFEST_FIELDS = (
    'student_sha256','original_student_sha256','evaluator_sha256','source_sha256',
    'actor_sha256','execution_actor_sha256','completion_sha256','pose_estimator',
    'goal_manifest','provider','upstream','maintenance','seed','episodes','stopping',
    'physics','goal_input','request_budget','service_exclusion','formal_trials_used',
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run_map',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.root=args.root.resolve()
    runmap=json.loads(args.run_map.read_text())
    source_summary=json.loads(args.summary.read_text())
    conditions={};trials=[];manifests={}
    for name,spec in runmap.items():
        if not isinstance(spec,dict) or not spec.get('real_model'):
            continue
        if not source_summary['results'].get(name,{}).get('complete'):
            continue
        root=args.root/spec['root']
        data=read_run(root,True)
        if not data['summary']['complete']:
            raise ValueError('Completed source summary disagrees with raw files: '+name)
        if data['summary']['success']!=source_summary['results'][name]['success']:
            raise ValueError('Source success recount mismatch: '+name)
        conditions[name]=data['summary']
        for key,row in sorted(data['records'].items()):
            if row['trial_seed']//1000 not in (182921,182922):
                raise ValueError('This export is restricted to the original development split')
            record={key:row.get(key) for key in TRIAL_FIELDS}
            record.update(condition=name,initial_state_sha256=row['initial_state']['sha256'])
            trials.append(record)
        for path in sorted(root.glob('*/manifest.json')):
            manifest=json.loads(path.read_text())
            if manifest['formal_trials_used']:
                raise ValueError('Formal records cannot be exported as development')
            value={key:manifest.get(key) for key in MANIFEST_FIELDS}
            value['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            manifests[name+':'+path.parent.name]=value
    if len(trials)!=40*len(conditions) or len({(r['condition'],r['trial_seed']) for r in trials})!=len(trials):
        raise ValueError('Missing or duplicate development slots')
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Pose-latent Newton migration development results. Completed actual-model conditions only. Reused development seeds, not formal test evidence. Historical RGB-D/36-trial cohorts must remain separate.',
        completed_semantic_conditions=conditions,
        comparisons={key:value for key,value in source_summary['semantic_ablations'].items()
                     if value['complete'] and set(value['eligible_success']).issubset(conditions)},
        formal_seeds_used=False,model_superiority_established=False)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary',summary),('trials',trials),('manifests',manifests)]:
        encoded=json.dumps(value,indent=2).replace(str(args.root.resolve())+'/', '')
        temporary=args.output/(name+'.tmp')
        temporary.write_text(encoded+'\n')
        temporary.replace(args.output/(name+'.json'))
    print(json.dumps(dict(completed_conditions=len(conditions),raw_rows=len(trials))))


if __name__=='__main__':
    main()
