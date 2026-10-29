"""Audit complete independent cohorts and export raw plus matched eligible results."""
import argparse,hashlib,importlib.util,itertools,json,math,statistics
from collections import Counter
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-radius,center+radius]
def mcnemar(a,b):
    n=a+b
    return min(1.,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/2**n) if n else 1.
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=read(a.root/'formal_release.json');rh=sha(a.root/'formal_release.json')
    if r['format'] not in ('cube_v1_independent_replication_v1','cube_controller_goal_ablation_v1'):
        raise ValueError('Unsupported registered study')
    spec=importlib.util.spec_from_file_location('study_guard',a.root/'sources/formal_release.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    g.validate_contracts(r)
    if {p.name:sha(p) for p in (a.root/'sources').glob('*.py')}!=r['sources_sha256']:
        raise ValueError('Saved evaluator snapshot changed')
    expected={seed*1000+i for seed in r['cohort']['seeds'] for i in range(r['cohort']['episodes_per_seed'])}
    goal_study=r['format']=='cube_controller_goal_ablation_v1'
    rows={};proof={};phase_counts={}
    for condition,entry in r['conditions'].items():
        base=a.root/('formal' if goal_study else 'runs')/condition;records={};proof[condition]=[]
        provider=entry['manifest_contract']['provider'];phase_counts[condition]=Counter()
        for seed in r['cohort']['seeds']:
            run=base/(str(seed) if goal_study else provider+'_'+str(seed))
            m=read(run/'manifest.json');result=read(run/'result.json');v=read(run.with_suffix('.validity.json'))
            g.verify_runtime(r,condition,m)
            if (m['formal_trials_used'] is not True or m['formal_condition']!=condition
                    or m['formal_release_sha256']!=rh or result['formal_release_sha256']!=rh
                    or result['formal_condition']!=condition or result['formal_agent_result'] is not True
                    or sha(run/'formal_release.json')!=rh or result['status']!='completed'
                    or v['physical_log_valid'] is not True or v['exit_code']!=0):
                raise ValueError('Unverified runtime: '+str(run))
            raw=(run/'episodes.jsonl').read_text().splitlines(keepends=True)
            if any(not line.endswith('\n') for line in raw):raise ValueError('Truncated trial record')
            current=[json.loads(line) for line in raw]
            if len(current)!=20 or result['recorded_episodes']!=20:raise ValueError('Incomplete seed')
            for row in current:
                key=row['trial_seed']
                if key in records:raise ValueError('Duplicate trial')
                if row['exclusion_reason'] not in (None,'service_error','request_latency_over_120s'):
                    raise ValueError('Unregistered exclusion')
                if row['success']!=(row['reason']=='success'):raise ValueError('Contradictory success flag')
                if row['success'] and not (row['declared_goal']==row['goal_index'] and row['declared_symbol']==row['symbol']
                        and math.isfinite(row['final_error_rad']) and row['final_error_rad']<=.16):
                    raise ValueError('Success failed independent binding/pose verification')
                records[key]=row
            if goal_study:
                contract=m['controller_goal_intervention'];mode=contract['waiting_goal_input'];shift=contract['mode']=='cyclic_shift'
                with (run/'steps.jsonl').open() as stream:
                    for line in stream:
                        step=json.loads(line);index=None if mode=='center' else (step['active_goal']+int(shift))%6
                        if step['controller_goal_input']!=mode or step['controller_goal_index']!=index:
                            raise ValueError('Controller input deviated from registered intervention')
                        phase_counts[condition]['pending' if step['pending'] else 'known_execution' if step['known'] else 'unbound_execution']+=1
            proof[condition].append({str(p.relative_to(a.root)):sha(p) for p in
                (run/'manifest.json',run/'episodes.jsonl',run/'result.json',run.with_suffix('.validity.json'))})
        if set(records)!=expected:raise ValueError('Missing or replaced trial slots')
        rows[condition]=records
    for key in expected:
        first=next(iter(rows.values()))[key]
        if any(row[key]['initial_state']!=first['initial_state'] or row[key]['goal_index']!=first['goal_index'] for row in rows.values()):
            raise ValueError('Paired initial physical state or task differs')
    excluded={k for k in expected if any(v[k]['exclusion_reason'] is not None for v in rows.values())}
    eligible=expected-excluded;outcomes={}
    for name,records in rows.items():
        values=list(records.values());raw_success=sum(v['success'] for v in values);success=sum(records[k]['success'] for k in eligible)
        latencies=[t for v in values for t in v['latencies_s']]
        outcomes[name]=dict(raw_trials=len(values),raw_success=raw_success,raw_rate=raw_success/len(values),
            raw_drops=sum(v['reason']=='drop' for v in values),raw_reasons=dict(Counter(v['reason'] for v in values)),
            own_service_exclusions=sum(v['exclusion_reason'] is not None for v in values),common_eligible_trials=len(eligible),
            common_eligible_success=success,common_eligible_rate=success/len(eligible) if eligible else None,wilson_95=wilson(success,len(eligible)),
            actual_calls=sum(v['actual_model_calls'] for v in values),median_call_latency_s=statistics.median(latencies) if latencies else None,
            drop_phases=dict(Counter(v['drop_phase'] for v in values if v['reason']=='drop')))
    pairs=[]
    for left,right in itertools.combinations(rows,2):
        l=sum(rows[left][k]['success'] and not rows[right][k]['success'] for k in eligible)
        rr=sum(rows[right][k]['success'] and not rows[left][k]['success'] for k in eligible)
        pairs.append(dict(left=left,right=right,left_only=l,right_only=rr,exact_mcnemar_p=mcnemar(l,rr),scope='Exploratory paired test, no multiple-testing adjustment'))
    summary=dict(status='completed_and_audited',release_sha256=rh,cohort=r['cohort'],common_excluded_trial_ids=sorted(excluded),
        initial_states_match=True,outcomes=outcomes,paired_comparisons=pairs,controller_phase_step_counts={k:dict(v) for k,v in phase_counts.items()})
    a.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary',summary),('evidence',proof),('trials',{k:list(v.values()) for k,v in rows.items()})]:
        (a.output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
