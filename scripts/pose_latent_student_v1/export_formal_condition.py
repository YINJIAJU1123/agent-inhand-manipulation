"""Archive one completed semantic condition; final matched comparisons stay separate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--source_directory',type=Path,required=True)
    parser.add_argument('--condition',required=True)
    args=parser.parse_args()
    suite=args.suite.resolve();src=args.source_directory.resolve()
    launch=json.loads((suite/'launch.json').read_text())
    condition=launch['conditions'][args.condition]
    if condition['scope']!='autonomous_semantic_agent':
        raise ValueError('Use the separate low-level diagnostic exporter')
    root=suite.parent/condition['root'];prefix=condition['prefix']
    terminal=root/'process_status.txt'
    if not terminal.exists():
        print(json.dumps(dict(complete=False,condition=args.condition)))
        return
    if terminal.read_text().strip()!='stage=finished\nexit_code=0':
        raise ValueError('Condition did not finish cleanly')
    sys.path.insert(0,str(src))
    from formal_release import verify_runtime,digest,FORMAL_SEEDS,validate_contracts
    from summarize import read_run
    release=json.loads((suite/'formal_release.json').read_text())
    validate_contracts(release)
    sha=digest(suite/'formal_release.json')
    if launch['release_sha256']!=sha:
        raise ValueError('Suite launch/release mismatch')
    for directory in (src,root/'sources'):
        if {p.name:digest(p) for p in directory.glob('*.py')}!=release['sources_sha256']:
            raise ValueError('Evaluator source snapshot changed')
    current=read_run(root,True);summary=current['summary']
    expected={str(seed*1000+i) for seed in FORMAL_SEEDS for i in range(20)}
    if not summary['complete'] or not summary['physics_log_valid'] or set(current['records'])!=expected:
        raise ValueError('Incomplete or invalid forty-trial condition')
    files={};cost=[];latencies=[]
    # A recorded service failure can leave a valid prefix and an unvisited-tail fragment.
    runs=sorted(path.parent for path in root.glob(prefix+'*/episodes.jsonl'))
    for run in runs:
        manifest=json.loads((run/'manifest.json').read_text())
        result=json.loads((run/'result.json').read_text())
        validity=json.loads(run.with_suffix('.validity.json').read_text())
        verify_runtime(release,args.condition,manifest)
        if not (manifest['formal_release_sha256']==result['formal_release_sha256']==sha
                and manifest['formal_condition']==result['formal_condition']==args.condition
                and manifest['formal_trials_used'] and result['formal_agent_result']
                and validity['physical_log_valid'] and validity['exit_code']==0
                and digest(run/'formal_release.json')==sha):
            raise ValueError('Run is not bound to the frozen formal condition')
        for name in ('manifest.json','result.json','episodes.jsonl'):
            files[run.name+'/'+name]=(run/name).read_text()
        files[run.name+'.validity.json']=run.with_suffix('.validity.json').read_text()
        for row in (json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()):
            calls_path=run/f"episode_{row['episode']:03d}"/'calls.json'
            calls=json.loads(calls_path.read_text())
            if len(calls)!=row['actual_model_calls']:
                raise ValueError('Call count disagrees with episode record')
            for call in calls:
                response=call.get('result') or {};usage=response.get('usage')
                item=dict(trial_seed=row['trial_seed'],call=call['call'],elapsed_s=call['elapsed_s'],
                    error=call.get('error'),calls_file_sha256=digest(calls_path),usage_available=usage is not None)
                if usage is not None:
                    item.update(input_tokens=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),
                        total_tokens=usage.get('total_tokens'),
                        cached_input_tokens=usage.get('input_tokens_details',{}).get('cached_tokens'),
                        reasoning_output_tokens=usage.get('output_tokens_details',{}).get('reasoning_tokens'))
                cost.append(item);latencies.append(call['elapsed_s'])
    if len(cost)!=summary['actual_calls']:
        raise ValueError('Aggregate call count mismatch')
    known=[row for row in cost if row['usage_available']]
    def token_sum(key):
        values=[row[key] for row in known if row.get(key) is not None]
        return sum(values) if values else None
    latencies.sort()
    def percentile(q):
        if not latencies:return None
        position=(len(latencies)-1)*q;left=int(position);right=min(left+1,len(latencies)-1)
        return latencies[left]+(latencies[right]-latencies[left])*(position-left)
    resources=dict(actual_calls=len(cost),calls_with_usage=len(known),
        reported_total_tokens=token_sum('total_tokens'),reported_input_tokens=token_sum('input_tokens'),
        reported_output_tokens=token_sum('output_tokens'),latency_p50_s=percentile(.5),
        latency_p95_s=percentile(.95),latency_max_s=max(latencies) if latencies else None,
        scope='Recorded model tokens and client latency. Missing token usage is null, not zero. No currency, account quota, or cross-provider GPU-compute equivalence is implied.')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),condition=args.condition,
        formal_release_sha256=sha,summary=summary,resources=resources,matched_panel_pending=True,
        analysis_source_sha256=digest(Path(__file__)),
        scope='Complete raw forty-trial formal condition. Group-wide exclusions and final comparisons are computed by summarize_formal.py after all arms complete.')
    print(json.dumps(dict(complete=True,report=report,files=files,call_resources=cost)))


if __name__=='__main__':
    main()
