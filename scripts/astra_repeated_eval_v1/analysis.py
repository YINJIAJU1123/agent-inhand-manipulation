"""Paired repeated trials: repetitions are not independent physical scenes."""
import json
from pathlib import Path
import numpy as np

def analyze_learning(out):
    arms={a:{} for a in ['continue','slow','v3','learned']};initial={};maxerr=0.;compares=0
    for folder in sorted(out.glob('eval_r*')):
        if not (folder/'report.json').exists():continue
        rep=int(folder.name.split('_')[1][1:]);r=json.loads((folder/'report.json').read_text());m=r['metadata'];state=json.loads((folder/'initial.json').read_text())
        k=(rep,m['seed'],m['template_set'])
        if k in initial:
            for field in state:maxerr=max(maxerr,float(np.max(np.abs(np.asarray(state[field])-np.asarray(initial[k][field])))))
            compares+=1
        else:initial[k]=state
        for row in r['episodes']:
            key=(rep,row['seed'],row['template_set'],row['case'])
            if key in arms[m['waiting']]:raise RuntimeError('Duplicate repeated case')
            arms[m['waiting']][key]=row
    if maxerr>1e-6:raise RuntimeError('Initial states differ')
    def metrics(rows):
        rows=list(rows);n=len(rows)
        return dict(episodes=n,success=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/n if n else None,drop=sum(r['drop'] for r in rows),wait_drop=sum(r['wait_drop'] for r in rows),false_stop=sum(r['false_stop'] for r in rows),active_error_episodes=sum(any(e['status']=='error' for e in r['async_events']) for r in rows))
    common=set.intersection(*(set(x) for x in arms.values()))
    result=dict(planned_episodes=192,completed_episodes=sum(map(len,arms.values())),initial_comparisons=compares,initial_max_abs_error=maxerr,arms={a:metrics(d.values()) for a,d in arms.items()},matched_all_arms={a:metrics(d[k] for k in sorted(common)) for a,d in arms.items()},paired_learned_vs={},by_repeat={},by_seed={},evidence='Same two previously used seeds, repeated live Astra calls. Descriptive robustness to live response/execution variability, not new independent scenes or held-out generalization. All reported cases including API failures retained.')
    for a in ['continue','slow','v3']:
        keys=set(arms[a])&set(arms['learned'])
        result['paired_learned_vs'][a]=dict(matched=len(keys),wins=sum(arms['learned'][k]['success'] and not arms[a][k]['success'] for k in keys),losses=sum(arms[a][k]['success'] and not arms['learned'][k]['success'] for k in keys))
    for index,values,name in [(0,[1,2],'by_repeat'),(1,[74101,74102],'by_seed')]:
        for v in values:result[name][str(v)]={a:metrics(r for k,r in d.items() if k[index]==v) for a,d in arms.items()}
    tmp=out/'summary.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'summary.json');return result
