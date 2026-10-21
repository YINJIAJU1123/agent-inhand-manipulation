"""Separate descriptive diagnostic, with request-level and episode-level counts."""
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def save(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def quantiles(x):return dict(zip(['p50','p90','max'],map(float,np.quantile(x,[.5,.9,1])))) if x else None

def analyze(out):
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
if __name__=='__main__':
    import sys
    print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
