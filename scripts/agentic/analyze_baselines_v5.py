"""Paired system comparison; confidence intervals resample whole seed clusters."""
import argparse
import json
from pathlib import Path
import numpy as np


def bootstrap_ratio(numerator, denominator):
    if len(numerator)<2:return None
    rng=np.random.default_rng(50715)
    indices=rng.integers(0,len(numerator),size=(5000,len(numerator)))
    values=np.asarray(numerator)[indices].sum(1)/np.asarray(denominator)[indices].sum(1)
    return np.quantile(values,[.025,.975]).tolist()


def analyze(root):
    freeze=json.loads((root/'freeze.json').read_text());protocol=freeze['protocol']
    rows={mode:[] for mode in protocol['modes']};initials={};reports=[]
    for path in sorted(root.glob('seed*/report.json')):
        report=json.loads(path.read_text());m=report['metadata'];mode=m['modes'][0];seed=m['seeds'][0]
        assert all(freeze['source_sha256'][k]==v for k,v in m['source_sha256'].items()), 'Source drift'
        assert m['max_steps']==protocol['max_steps'] and m['call_budget']==protocol['call_budget']
        initial=json.loads((path.parent/f'initial_{seed}_{mode}.json').read_text())
        if seed in initials:assert initials[seed]==initial,'Unpaired initial state or layout'
        else:initials[seed]=initial
        assert len(report['episodes'])==6 and {e['case'] for e in report['episodes']}==set(range(6))
        for episode in report['episodes']:
            assert episode['calls']<=protocol['call_budget']
            rows[mode].append(dict(episode,template_set=m['template_set'],instruction_offset=m['instruction_offset']))
        reports.append(str(path.relative_to(root)))
    summary={'benchmark':protocol['benchmark'],'status':'complete' if len(reports)==len(protocol['scenarios'])*len(protocol['modes']) else 'partial',
             'completed_jobs':len(reports),'planned_jobs':len(protocol['scenarios'])*len(protocol['modes']),
             'paired_initial_states_verified':True,'seed_clusters_seen':len(initials),
             'confidence_interval':'95% percentile bootstrap, 5000 resamples of whole seeds (all cases/templates together); not independent-episode binomial intervals',
             'limits':protocol['limits'],'modes':{},'reports':reports}
    for mode,episodes in rows.items():
        n=len(episodes)
        if not n:summary['modes'][mode]={'episodes':0};continue
        ids={(e['seed'],e['case'],e['template_set'],e['instruction_offset']) for e in episodes}
        assert len(ids)==n,'Duplicate episodes'
        seeds=sorted({e['seed'] for e in episodes})
        total=[sum(e['seed']==seed for e in episodes) for seed in seeds]
        wins=[sum(e['success'] for e in episodes if e['seed']==seed) for seed in seeds]
        durations=[t['elapsed_s'] for e in episodes for t in e['trace'] if 'elapsed_s' in t]
        metrics={k:sum(e[k] for e in episodes) for k in ['success','drop','grounding_correct','calls']}
        metrics.update(episodes=n,success_rate=metrics['success']/n,seed_cluster_success_ci95=bootstrap_ratio(wins,total),
                       no_accepted_goal=sum(e['final_goal'] is None for e in episodes),
                       inference_total_seconds=sum(durations),inference_call_median_seconds=float(np.median(durations)) if durations else None,
                       by_symbol={symbol:{'episodes':sum(e['target_symbol']==symbol for e in episodes),'success':sum(e['success'] for e in episodes if e['target_symbol']==symbol)} for symbol in sorted({e['target_symbol'] for e in episodes})},
                       by_template={template:{'episodes':sum(e['template_set']==template for e in episodes),'success':sum(e['success'] for e in episodes if e['template_set']==template)} for template in ['canonical','paraphrase']})
        summary['modes'][mode]=metrics
    key=lambda e:(e['seed'],e['case'],e['template_set'],e['instruction_offset'])
    left={key(e):e for e in rows['periodic']};right={key(e):e for e in rows['latent']}
    pairs=sorted(left.keys() & right.keys());seeds=sorted({k[0] for k in pairs})
    if pairs:
        for k in pairs:
            assert left[k]['instruction']==right[k]['instruction'] and left[k]['layout']==right[k]['layout']
        differences=[int(right[k]['success'])-int(left[k]['success']) for k in pairs]
        numerator=[sum(d for k,d in zip(pairs,differences) if k[0]==seed) for seed in seeds]
        denominator=[sum(k[0]==seed for k in pairs) for seed in seeds]
        summary['paired_latent_minus_rule']={'pairs':len(pairs),'seed_clusters':len(seeds),'success_rate_difference':float(np.mean(differences)),
                                            'seed_cluster_ci95':bootstrap_ratio(numerator,denominator),
                                            'latent_only_success':sum(d==1 for d in differences),'rule_only_success':sum(d==-1 for d in differences)}
    target=root/'analysis.json';tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(target)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args()
    result=analyze(Path(a.directory));print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
