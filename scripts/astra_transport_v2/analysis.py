"""Separate descriptive diagnostic, with request-level and episode-level counts."""
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def save(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def quantiles(x):return dict(zip(['p50','p90','max'],map(float,np.quantile(x,[.5,.9,1])))) if x else None

def analyze_coordination_raw(out):
    arms={f'{h}/extra{d}':dict(episodes=[],responses=[]) for h in ['direct','learned'] for d in [0,6]};states={};comparisons=0;maxerr=0.
    for folder in sorted(out.glob('eval_*')):
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json');m=report['metadata'];key=f"{m['handoff_mode']}/extra{m['extra_delay_s']:g}"
        arms[key]['episodes']+=report['episodes'];arms[key]['responses']+=read(folder/'response_audit.json')
        state=read(folder/'initial.json');seed=m['seed']
        if seed in states:
            for field in state:maxerr=max(maxerr,float(np.max(np.abs(np.asarray(state[field])-np.asarray(states[seed][field])))))
            comparisons+=1
        else:states[seed]=state
    if maxerr>1e-6:raise RuntimeError('Initial state mismatch')
    result=dict(planned_episodes=48,completed_episodes=sum(len(x['episodes']) for x in arms.values()),initial_comparisons=comparisons,initial_max_abs_error=maxerr,
        evidence='Development diagnostic: real Astra with additional controlled simulated delay only after a skill is known. Request observations and live stochastic responses differ after trajectories diverge. Motion alone is not a stale/invalid-instruction label.',
        smoke_excluded=True,arms={},paired={},balanced_seeds=[])
    for key,data in arms.items():
        es=data['episodes'];rs=[r for r in data['responses'] if r['status']=='delivered'];known=[r for r in rs if r['previous_decision']['decision']['color'] is not None]
        result['arms'][key]=dict(episodes=len(es),success=sum(r['success'] for r in es),success_rate=sum(r['success'] for r in es)/len(es) if es else None,
            drop=sum(r['drop'] for r in es),wait_drop=sum(r['wait_drop'] for r in es),false_stop=sum(r['false_stop'] for r in es),calls=sum(r['calls'] for r in es),
            delivered_responses=len(rs),known_skill_responses=len(known),undelivered_requests=sum(r['status']!='delivered' for r in data['responses']),
            known_skill_image_age_s=quantiles([r['request_image_age_s'] for r in known]),
            known_skill_rotation_drift_rad=quantiles([r['privileged_state_drift']['object_rotation_rad'] for r in known]),
            known_skill_translation_drift_m=quantiles([r['privileged_state_drift']['object_translation_m'] for r in known]),
            known_skill_visual_rms=quantiles([r['sensor_drift']['visual_rms'] for r in known]),
            responses_followed_by_drop_within_2s=sum(r['drop_within_2s'] for r in known),
            unique_cases_with_drop_within_2s=len({(r['seed'],r['case']) for r in known if r['drop_within_2s']}),
            followup_censored=sum(r['followup_censored'] for r in known))
    for delay in [0,6]:
        base={(r['seed'],r['case']):r for r in arms[f'direct/extra{delay}']['episodes']};new=arms[f'learned/extra{delay}']['episodes'];matches=[(r,base[(r['seed'],r['case'])]) for r in new if (r['seed'],r['case']) in base]
        result['paired'][f'learned_vs_direct_extra{delay}']=dict(matched=len(matches),wins=sum(a['success'] and not b['success'] for a,b in matches),losses=sum(b['success'] and not a['success'] for a,b in matches))
    for seed in [75201,75202]:
        if all(sum(e['seed']==seed for e in d['episodes'])==6 for d in arms.values()):result['balanced_seeds'].append(seed)
    save(out/'summary.json',result);return result

def analyze_learning_raw(out):
    arms={a:[] for a in ['continue','slow','v3','learned']};initial={};comparisons=0;maxerr=0.
    import numpy as np
    for folder in sorted(out.glob('eval_*')):
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json');m=report['metadata'];arms[m['waiting']]+=report['episodes']
        key=(m['seed'],m['template_set']);value=read(folder/'initial.json')
        if key in initial:
            for field in value:maxerr=max(maxerr,float(np.max(np.abs(np.asarray(value[field])-np.asarray(initial[key][field])))))
            comparisons+=1
        else:initial[key]=value
    if maxerr>1e-6:raise RuntimeError('Matched initial states differ')
    summary=dict(completed_episodes=sum(map(len,arms.values())),planned_episodes=96,initial_comparisons=comparisons,initial_max_abs_error=maxerr,smoke_excluded=True,
        evidence='Two new seeds, descriptive pilot. Actual Astra responses/latencies differ across arms. No significance claim.',arms={})
    for name,rows in arms.items():
        summary['arms'][name]=dict(episodes=len(rows),success=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows) if rows else None,drop=sum(r['drop'] for r in rows),wait_drop=sum(r['wait_drop'] for r in rows),false_stop=sum(r['false_stop'] for r in rows),calls=sum(r['calls'] for r in rows))
    paired={}
    for ref in ['continue','slow','v3']:
        rm={(r['seed'],r['template_set'],r['case']):r for r in arms[ref]};wins=losses=n=0
        for row in arms['learned']:
            key=(row['seed'],row['template_set'],row['case'])
            if key in rm:
                n+=1;wins+=int(row['success'] and not rm[key]['success']);losses+=int(rm[key]['success'] and not row['success'])
        paired[ref]=dict(matched=n,wins=wins,losses=losses)
    summary['paired_learned_vs']=paired
    save(out/'summary.json',summary);return summary


def add_transport_audit(out,result):
    groups={};errors=[]
    for folder in sorted(out.glob('eval_*')):
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json');m=report['metadata']
        version='prior_single_attempt' if folder.is_symlink() else 'bounded_two_attempts'
        method=m.get('handoff_mode',m['waiting'])
        if 'extra_delay_s' in m:method+=f"/extra{m['extra_delay_s']:g}"
        key=version+'/'+method
        g=groups.setdefault(key,dict(episodes=0,success=0,active_error_episodes=0,late_error_events=0,no_active_error_episodes=0,no_active_error_success=0))
        for row in report['episodes']:
            active=[e for e in row['async_events'] if e['status']=='error']
            late=[e for e in row['async_events'] if e['status']=='discarded_terminal' and e.get('error')]
            g['episodes']+=1;g['success']+=int(row['success']);g['active_error_episodes']+=int(bool(active));g['late_error_events']+=len(late)
            if not active:g['no_active_error_episodes']+=1;g['no_active_error_success']+=int(row['success'])
            if active:errors.append(dict(batch=folder.name,case=row['case'],errors=active))
    result['transport_cohorts']=groups;result['active_infrastructure_errors']=errors
    result['transport_caveat']='All newly attempted formal cases remain in total success denominators, including exhausted transport retries. Error-free subsets are descriptive and potentially selected. Inherited single-attempt cases and new two-attempt cases are stratified; pooled methods are not a controlled transport comparison.'
    save(out/'summary.json',result);return result

def analyze_learning(out):return add_transport_audit(out,analyze_learning_raw(out))
def analyze_coordination(out):return add_transport_audit(out,analyze_coordination_raw(out))
