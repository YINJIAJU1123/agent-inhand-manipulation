"""Aggregate completed async episodes, audit initial states, cluster uncertainty by seed."""
import argparse,json,math
from pathlib import Path
import numpy as np

def analyze(root):
    reports=[];initials={};worst=0.;comparisons=0;excluded=[]
    for path in sorted(root.glob('eval_*/report.json')):
        report=json.loads(path.read_text());m=report['metadata'];key=(m['seed'],m['template_set'])
        if any(x['status']=='error' for e in report['episodes'] for x in e['async_events']):
            excluded.append(dict(path=str(path),reason='active inference error'));continue
        initial=json.loads((path.parent/'initial.json').read_text())
        if key in initials:
            first=initials[key];assert first['layout']==initial['layout']
            for field in ['joint_pos','joint_vel','object_root','prev_targets']:
                error=float(np.max(np.abs(np.asarray(first[field])-np.asarray(initial[field]))))
                worst=max(worst,error)
                if error>1e-6:raise RuntimeError(f'Initial-state audit failed: {path.parent.name} {field} {error}')
            comparisons+=1
        else:initials[key]=initial
        reports.append(report)
    arms={};rows={};rng=np.random.default_rng(20261016)
    for report in reports:
        m=report['metadata'];arm=m['mode']+'/'+m['waiting'];rows.setdefault(arm,[]).extend(report['episodes'])
    for arm,episodes in rows.items():
        events=[x for e in episodes for x in e['async_events']]
        latency=[x['wall_latency_s'] for x in events if 'wall_latency_s' in x]
        seeds=sorted({e['seed'] for e in episodes})
        seed_rates=np.array([np.mean([e['success'] for e in episodes if e['seed']==s]) for s in seeds])
        ci=np.quantile(rng.choice(seed_rates,size=(5000,len(seeds)),replace=True).mean(1),[.025,.975]).tolist() if len(seeds)>1 else None
        arms[arm]=dict(episodes=len(episodes),seeds=len(seeds),success=sum(e['success'] for e in episodes),success_rate=np.mean([e['success'] for e in episodes]),seed_cluster_bootstrap_95=ci,drop=sum(e['drop'] for e in episodes),wait_drop=sum(e['wait_drop'] for e in episodes),false_stop=sum(e['false_stop'] for e in episodes),calls=sum(e['calls'] for e in episodes),inference_errors=sum(bool(x.get('error')) for x in events),delivered=sum(x['status']=='delivered' for x in events),late_discarded=sum(x['status']=='discarded_terminal' for x in events),wall_latency_p50_p95_max=np.quantile(latency,[.5,.95,1]).tolist() if latency else None,pending_physics_steps=sum(e['counts']['pending_steps'] for e in episodes),cold_physics_steps=sum(e['counts']['cold_steps'] for e in episodes))
    paired={}
    for backend in ['astra','latent']:
        learned={(e['seed'],e['template_set'],e['case']):e for e in rows.get(backend+'/learned',[])}
        for baseline in ['continue','slow']:
            other={(e['seed'],e['template_set'],e['case']):e for e in rows.get(backend+'/'+baseline,[])}
            common=learned.keys() & other.keys()
            if not common:continue
            wins=sum(learned[k]['success'] and not other[k]['success'] for k in common)
            losses=sum(other[k]['success'] and not learned[k]['success'] for k in common)
            paired[backend+'/learned_vs_'+baseline]=dict(matched_cases=len(common),wins=wins,losses=losses,rate_difference=(wins-losses)/len(common),caveat='matched initial conditions; independent live inference latency and sampling')
    excluded.extend(json.loads(p.read_text()) for p in root.glob('attempt_*/failure.json'))
    result=dict(excluded_infrastructure_batches=excluded,completed_episodes=sum(len(r['episodes']) for r in reports),planned_episodes=720,arms=arms,paired=paired,initial_audit=dict(comparisons=comparisons,max_abs_error=worst),smoke_excluded=True)
    temp=root/'summary.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(root/'summary.json');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();print(json.dumps(analyze(Path(a.output)),indent=2))
