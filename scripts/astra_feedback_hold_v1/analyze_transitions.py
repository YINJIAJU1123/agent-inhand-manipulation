"""Audit imported hold transitions; pre-action probability and post-action labels differ by one step."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    verified=json.loads((a.root/'trace_retrieval.json').read_text());assert verified['status']=='completed'
    for rel,digest in verified['files'].items():
        assert hashlib.sha256((a.root/'traces'/rel).read_bytes()).hexdigest()==digest
    records=[];releases=[]
    for seed in (117101,117102):
        groups={}
        for mode in ('none','hold','feedback'):
            d=a.root/'traces/cases'/str(seed)/mode
            groups[mode]={name:json.loads((d/(name+'.json')).read_text()) for name in ('report','trace','retention_events')}
        for case in range(6):
            per_mode={mode:data['report']['episodes'][case] for mode,data in groups.items()}
            events={mode:[e for e in data['retention_events'] if e['case']==case] for mode,data in groups.items()}
            starts={mode:min((e['step'] for e in ev if e.get('rate') is not None),default=None) for mode,ev in events.items()}
            rec=dict(seed=seed,case=case,by_mode=per_mode,first_hold_step=starts,
                final_lost_vs_hold=per_mode['hold']['final_ready'] and not per_mode['feedback']['final_ready'],
                drop_absent_vs_hold=per_mode['hold']['drop'] and not per_mode['feedback']['drop'])
            if rec['drop_absent_vs_hold']:
                first=starts['feedback'];rec['feedback_treatment_before_hold_drop']=first is not None and first<per_mode['hold']['drop_step']
            records.append(rec)
            for event in events['feedback']:
                if event['event']!='retention_choice' or event['rate'] is not None:continue
                sample=next((t for t in groups['feedback']['trace'] if t['step']==event['step']),None)
                if sample is None:continue
                releases.append(dict(seed=seed,case=case,step=event['step'],phase=sample['phase'],
                    commanded_color=event['color'],pre_action_probability=sample['probabilities'][case][event['color']],
                    post_one_step_error=sample['score_errors'][case],post_one_step_ready=sample['score_errors'][case]<=.16))
    result=dict(status='completed',cloud_calls=0,source_transport_sha256=verified['transport_sha256'],
        source_file_sha256=verified['files'],cases=records,releases=releases,
        release_count=len(releases),releases_followed_by_one_step_ready=sum(r['post_one_step_ready'] for r in releases),
        temporal_caveat='Probabilities are measured before the action; errors are scored after one physical control step. These pairs cannot label the precise pre-action frame.',
        causal_caveat='After the common physical and sensor prefix, independent rendering and closed-loop trajectories can diverge. In particular a drop difference before that case receives any changed hold action cannot be attributed to its hold release.',
        scope='Descriptive development trace analysis, no new success rate or policy deployment.')
    (a.root/'transition_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('release_count','releases_followed_by_one_step_ready')}))
    print('drop differences',[(r['seed'],r['case'],r['feedback_treatment_before_hold_drop']) for r in records if r['drop_absent_vs_hold']])

if __name__=='__main__':main()
