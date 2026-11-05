"""Summarize matched development rollouts without changing raw trial records."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from run_logged import capacity_warning

def wilson_interval(success,total):
    if not total:return None
    z=1.959963984540054
    rate=success/total
    denominator=1+z*z/total
    center=(rate+z*z/(2*total))/denominator
    radius=z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def exact_mcnemar(a_only,b_only):
    discordant=a_only+b_only
    if not discordant:return 1.
    return min(1.,2*sum(math.comb(discordant,k) for k in range(min(a_only,b_only)+1))/2**discordant)


def read_run(root,real_model):
    records={}
    results=[]
    runs=[]
    physics_issues=[]
    physics_checks_pending=[]
    fragments_valid=[]
    for path in sorted(root.glob('*/episodes.jsonl')):
        run=path.parent
        log=run.parent/(run.name+'.log')
        if log.exists():
            warnings=[line.strip() for line in log.read_text(errors='replace').splitlines() if capacity_warning(line)]
            if warnings:physics_issues.append(dict(log=str(log),count=len(warnings),first=warnings[0]))
        validity=log.with_suffix('.validity.json')
        if validity.exists():
            if not json.loads(validity.read_text())['physical_log_valid']:
                physics_issues.append(dict(log=str(log),validity=str(validity)))
        elif (run.parent/'sources/run_logged.py').exists():
            physics_checks_pending.append(str(validity))
        result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
        if result is not None:results.append(result)
        data=[json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
        fragment_valid=(result is not None and result['recorded_episodes']==len(data)
            and ((result['status']=='completed' and result['requested_episodes']==len(data))
                 or (result['status']=='service_blocked' and bool(data)
                     and data[-1].get('exclusion_reason') is not None)))
        fragments_valid.append(fragment_valid)
        diagnostics={}
        if real_model:
            # Earlier development logger used "known semantic binding" for its
            # startup category. Reconstruct actual first delivered color skill
            # from public controller mode, preserving all original files.
            with (run/'steps.jsonl').open() as stream:
                for line in stream:
                    if not line.endswith('\n'):break
                    row=json.loads(line)
                    episode=row['episode']
                    state=diagnostics.setdefault(episode,dict(first_skill=None,last=None))
                    if state['first_skill'] is None and row['mode']!='startup':
                        state['first_skill']=row['step']-1
                    state['last']=row
        for row in data:
            row=dict(row)
            key=str(row['trial_seed'])
            if key in records:raise ValueError('Duplicate trial seed')
            row['recorded_drop_phase']=row['drop_phase']
            if real_model and row['reason']=='drop':
                diag=diagnostics[row['episode']]
                row['drop_phase']=('startup_wait' if diag['first_skill'] is None else
                                   'wait' if diag['last']['pending'] else 'execution')
                row['drop_context']=('before_first_skill' if diag['first_skill'] is None else
                    'inspection_without_binding' if not diag['last']['known'] else
                    'correct_bound_goal' if row['declared_goal']==row['goal_index'] else 'wrong_bound_goal')
            if row['reason']=='false_stop':
                row['failure_subtype']=('semantic_wrong_binding' if real_model and row['declared_goal']!=row['goal_index']
                                        else 'pose_false_stop')
            elif row['reason']=='drop':row['failure_subtype']='drop_'+str(row['drop_phase'])
            else:row['failure_subtype']=row['reason']
            row['source']=str(path)
            records[key]=row
        runs.append(dict(path=str(run),records=len(data),fragment_valid=fragment_valid,
            result_status=result['status'] if result else None,
            episodes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()))
    seeds=sorted({r['trial_seed']//1000 for r in records.values()})
    expected={str(seed*1000+i) for seed in seeds for i in range(20)}
    complete=(len(seeds)==2 and len(records)==40 and set(records)==expected
        and len(results)==len(runs) and all(fragments_valid) and not physics_issues and not physics_checks_pending)
    return dict(records=records,summary=dict(planned_raw_trials=40,observed_trials=len(records),
        complete=complete,physics_log_valid=(None if physics_checks_pending or not runs else not physics_issues),
        physics_checks_pending=physics_checks_pending,physics_issues=physics_issues,success=sum(r['success'] for r in records.values()),
        failure_subtypes=dict(Counter(r['failure_subtype'] for r in records.values())),
        failure_contexts=dict(Counter(r.get('drop_context',r['failure_subtype']) for r in records.values() if not r['success'])),
        raw_success_rate=sum(r['success'] for r in records.values())/40 if complete else None,
        raw_wilson_95=wilson_interval(sum(r['success'] for r in records.values()),40) if complete else None,
        recorded_exclusions=sum(r.get('exclusion_reason') is not None for r in records.values()),
        actual_calls=sum(r.get('actual_model_calls',0) for r in records.values()),
        per_seed={str(seed):dict(trials=sum(r['trial_seed']//1000==seed for r in records.values()),
                                success=sum(r['trial_seed']//1000==seed and r['success'] for r in records.values()))
                  for seed in sorted({r['trial_seed']//1000 for r in records.values()})},
        runs=runs))

def compare(a,b,eligible_keys=None,cohort_complete=True):
    common=set(a['records'])&set(b['records'])
    keys=sorted(common if eligible_keys is None else common&set(eligible_keys))
    complete=(cohort_complete and len(common)==40 and a['summary']['complete'] and b['summary']['complete'])
    mismatches=[]
    outcomes=Counter()
    for key in keys:
        x,y=a['records'][key],b['records'][key]
        if x['goal_index']!=y['goal_index'] or x['initial_state']['sha256']!=y['initial_state']['sha256']:
            mismatches.append(key)
        outcomes['both_success' if x['success'] and y['success'] else
                 'a_only_success' if x['success'] else 'b_only_success' if y['success'] else 'both_fail']+=1
    return dict(paired_trials=len(keys),exact_initial_state_matches=len(keys)-len(mismatches),
                mismatched_trial_seeds=mismatches,outcomes=dict(outcomes),
                complete=complete,
                raw_exact_mcnemar_p=(exact_mcnemar(outcomes['a_only_success'],outcomes['b_only_success'])
                    if complete and keys and not mismatches else None))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run_map',type=Path,help='JSON condition name to output directory mapping')
    a=p.parse_args()
    roots=dict(continue_control='control_continuous_20261029_v1',learned_control='control_continuous_20261029_v1',
               fixed_slow_control='control_slow_20261029_v1',corrected_stop_control='control_stop_correction_20261029_v1',
               learned_wait_only='control_wait_only_20261029_v1',loaded_reference='control_reference_loaded_20261029_v1',
               astra='agent_astra_20261029_v2',finetuned_vlm='agent_vlm_20261029_v1',rule_vlm='agent_rule_20261029_v1')
    if a.run_map:roots=json.loads(a.run_map.read_text())
    data={}
    for name,spec in roots.items():
        if isinstance(spec,str):
            prefix=name.split('_')[0] if name in ('continue_control','learned_control') else '*'
            spec=dict(root=spec,prefix=prefix,real_model=name in ('astra','finetuned_vlm','rule_vlm'))
        root=a.root/spec['root']
        prefix=spec.get('prefix','*')
        class FilteredRoot:
            def glob(self,pattern):
                return root.glob(prefix+'*/episodes.jsonl')
        data[name]=read_run(FilteredRoot(),spec.get('real_model',False))
    control_comparisons={}
    for x,y in [('continue_control','fixed_slow_control'),('continue_control','learned_wait_only'),('continue_control','learned_control'),
                ('continue_control','corrected_stop_control'),('learned_control','learned_wait_only'),
                ('continue_control','loaded_reference'),('loaded_reference','corrected_stop_control'),
                ('continue_control','learned_control_v2'),('learned_control_v2','learned_wait_only_v2'),
                ('learned_wait_only','wait1_execution2'),('wait1_execution2','wait1_execution3'),
                ('learned_wait_only','wait1_async_stop'),('event_pose','event_evidence'),
                ('event_pose','event_conjunction'),('event_evidence','event_conjunction'),
                ('event_pose','event_continuous_correction'),('event_evidence','event_continuous_correction')]:
        if x in data and y in data:control_comparisons[x+'__'+y]=compare(data[x],data[y])
    def matched_semantic(names):
        common=set.intersection(*(set(data[name]['records']) for name in names))
        excluded={key for key in common if any(data[name]['records'][key].get('exclusion_reason') is not None for name in names)}
        eligible=common-excluded
        complete=all(data[name]['summary']['complete'] for name in names) and len(common)==40
        eligible_pairing={}
        for name in names[1:]:
            paired=compare(data[names[0]],data[name],eligible_keys=eligible,cohort_complete=complete)
            paired['exact_mcnemar_p']=paired.pop('raw_exact_mcnemar_p')
            eligible_pairing[name]=paired
        return dict(complete=complete,
            matched_observed_trials=len(common),matched_excluded_trial_seeds=sorted(excluded),
            matched_eligible_trials=len(eligible),
            eligible_success={name:sum(data[name]['records'][key]['success'] for key in eligible) for name in names},
            eligible_wilson_95=({name:wilson_interval(sum(data[name]['records'][key]['success'] for key in eligible),len(eligible)) for name in names}
                if all(data[name]['summary']['complete'] for name in names) else None),
            eligible_pairing=eligible_pairing,
            pairing={name:compare(data[names[0]],data[name]) for name in names[1:]})
    semantic=matched_semantic(('astra','finetuned_vlm','rule_vlm'))
    ablation_groups={
        'waiting_only_providers':('astra_wait_only','finetuned_vlm_wait_only','rule_vlm_wait_only'),
        'astra_waiting':('astra','astra_wait_only'),
        'finetuned_vlm_waiting':('finetuned_vlm','finetuned_vlm_wait_only'),
        'rule_vlm_waiting':('rule_vlm','rule_vlm_wait_only'),
        'astra_execution':('astra_wait_only','astra_execution_v3'),
        'astra_total_control':('astra','astra_execution_v3'),
        'astra_execution_binding':('astra_execution_v3','astra_execution_v4'),
        'astra_execution_timing':('astra_execution_v4','astra_execution_v5'),
        'astra_waiting_vs_timing':('astra_wait_only','astra_execution_v5'),
        'astra_execution_schedule':('astra_execution_v5','astra_execution_v6'),
        'astra_waiting_vs_schedule':('astra_wait_only','astra_execution_v6'),
        'astra_execution_inspection_balance':('astra_execution_v6','astra_execution_v10'),
        'astra_waiting_vs_inspection_balance':('astra_wait_only','astra_execution_v10'),
        'timing_execution_providers':('astra_execution_v5','finetuned_vlm_execution_v5','rule_vlm_execution_v5'),
        'vlm_execution_timing_vs_schedule':('finetuned_vlm_execution_v5','finetuned_vlm_execution_v6'),
        'rule_execution_timing_vs_schedule':('rule_vlm_execution_v5','rule_vlm_execution_v6'),
        'schedule_execution_providers':('astra_execution_v6','finetuned_vlm_execution_v6','rule_vlm_execution_v6'),
        'astra_vs_adapted_vlm_execution_v6':('astra_execution_v6','finetuned_vlm_execution_v6'),
        'vlm_waiting_vs_schedule':('finetuned_vlm_student_signal','finetuned_vlm_execution_v6'),
        'rule_waiting_vs_schedule':('rule_vlm_wait_only','rule_vlm_execution_v6'),
        'rule_slow_vs_schedule':('rule_vlm','rule_vlm_execution_v6'),
        'vlm_student_view_adaptation':('finetuned_vlm_wait_only','finetuned_vlm_student_signal'),
        'astra_full_vs_waiting':('astra_wait_only','astra_full_candidate'),
        'vlm_full_vs_adapted_waiting':('finetuned_vlm_student_signal','finetuned_vlm_full_candidate'),
        'full_candidate_providers':('astra_full_candidate','finetuned_vlm_full_candidate','rule_vlm_full_candidate'),
        'astra_full_candidate_stop':('astra_full_pose','astra_full_candidate')}
    semantic_ablations={label:matched_semantic(names) for label,names in ablation_groups.items()
                       if all(name in data for name in names)}
    report=dict(scope='new-backend development only; control isolation and actual semantic agents kept separate',
        raw_logs_preserved=True,success_definition_unchanged=True,
        uncertainty_note='Wilson intervals and unadjusted exact McNemar p values are descriptive development diagnostics. These reused seeds and multiple explored conditions are not confirmatory evidence. Raw paired p values include recorded exclusions; eligible_pairing uses the same group-wide exclusions as eligible_success and requires all forty raw slots in every group.',
        caution='Exact initial states do not prove bitwise rollout reproducibility. Startup numerical divergence was observed before the stop intervention; small outcome differences need the loaded-reference audit.',
        diagnostic_correction='startup drop means no color skill delivered, reconstructed from public mode; no result re-scoring',
        results={name:value['summary'] for name,value in data.items()},
        control_pairs=control_comparisons,semantic_matched=semantic,semantic_ablations=semantic_ablations)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.output)
    print(json.dumps(dict(results={n:{k:v[k] for k in ('observed_trials','complete','success','failure_subtypes')}
                                   for n,v in report['results'].items()},control_pairs=control_comparisons,
                         semantic_matched=semantic),indent=2))
if __name__=='__main__':main()
