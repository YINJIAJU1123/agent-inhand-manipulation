"""Preserve raw outcomes and a uniformly paired normal-service analysis."""
import json
import math
from pathlib import Path
from collections import Counter,defaultdict
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'semantic_completion_v1'))
from common import save
ARMS=('none','all')


def statistics(rows):
    n=len(rows)
    def ratio(k):return sum(bool(r[k]) for r in rows)/n if n else None
    stops=sum(r['stop_claimed'] for r in rows)
    proposals=[e for r in rows for e in r['controller_events'] if e['event']=='astra_stop_proposed']
    successes=sum(r['success'] for r in rows)
    if n:
        z=1.959963984540054; den=1+z*z/n; center=(successes/n+z*z/(2*n))/den
        radius=z*math.sqrt((successes/n)*(1-successes/n)/n+z*z/(4*n*n))/den
        interval=[center-radius,center+radius]
    else: interval=None
    return dict(descriptive_wilson_95=interval,episodes=n,successes=sum(r['success'] for r in rows),success_rate=ratio('success'),
        reaches=sum(r['reached'] for r in rows),reach_rate=ratio('reached'),
        drops=sum(r['drop'] for r in rows),waiting_drops=sum(r['wait_drop'] for r in rows),
        cold_drops=sum(r.get('cold_drop',False) for r in rows),
        false_stops=sum(r['false_stop'] for r in rows),stop_claims=stops,
        astra_stop_proposals=len(proposals),
        bad_proposals_rejected=sum(not e['scorer_proposal_correct'] and not e['accepted'] for e in proposals),
        good_proposals_rejected=sum(e['scorer_proposal_correct'] and not e['accepted'] for e in proposals),
        false_stop_rate_among_claims=sum(r['false_stop'] for r in rows)/stops if stops else None,
        reached_without_completion=sum(r['reached'] and not r['success'] for r in rows),
        budget_exhausted=sum(r['calls']>=6 and r['reason']=='horizon' for r in rows),
        mean_calls=sum(r['calls'] for r in rows)/n if n else None,
        mean_inspections=sum(sum(e['event']=='inspect' for e in r['controller_events']) for r in rows)/n if n else None,
        service_anomalies=sum(r['service_anomaly'] for r in rows),
        reasons=dict(Counter(r['reason'] for r in rows)))


def summarize(out):
    out=Path(out);rows=[];initials=defaultdict(list)
    for p in sorted(out.glob('eval_*/report.json')):
        report=json.loads(p.read_text());job=json.loads((p.parent/'job.json').read_text())
        initials[(job['repeat'],job['seed'],job['template'])].append(json.loads((p.parent/'initial.json').read_text()))
        for r in report['episodes']:
            rows.append(dict(**r,arm=job['arm'],repeat=job['repeat'],seed=job['seed'],template=job['template']))
    def key(r):return (r['repeat'],r['seed'],r['template'],r['case'])
    excluded={key(r) for r in rows if r['service_anomaly']}
    by_key=defaultdict(dict)
    for r in rows:by_key[key(r)][r['arm']]=r
    complete={k for k,v in by_key.items() if len(v)==len(ARMS)}
    paired=[r for r in rows if key(r) in complete-excluded]
    comparisons={}
    for arm in ARMS[1:]:
        pairs=[v for k,v in by_key.items() if k in complete-excluded]
        comparisons[arm]=dict(pairs=len(pairs),
            wins=sum(v[arm]['success'] and not v['none']['success'] for v in pairs),
            losses=sum(v['none']['success'] and not v[arm]['success'] for v in pairs))
    same=all(all(x==values[0] for x in values) for values in initials.values())
    audits=[json.loads(p.read_text()) for p in (out/'requests').glob('request_*.json')]
    result=dict(completed_episodes=len(rows),planned_episodes=json.loads((out/'protocol.json').read_text())['total_episodes'],
        raw={arm:statistics([r for r in rows if r['arm']==arm]) for arm in ARMS},
        normal_service_matched={arm:statistics([r for r in paired if r['arm']==arm]) for arm in ARMS},
        excluded_matched_keys=[list(k) for k in sorted(excluded)],paired_vs_no_retention=comparisons,
        source_initial_states_equal=same,
        api_attempts=len(audits),reported_tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in audits),
        usage_unknown_requests=sum(not r.get('usage') for r in audits),
        note='Two new physical seeds, six cases each, one instruction template. Twelve unique physical states per arm. Raw autonomous-stop success remains primary; normal-service analysis excludes matching keys symmetrically. Independent post-initialization image and action variation is possible. Development intervention experiment, not confirmation of 80% generalization.')
    save(out/'summary.json',result)
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(summarize(sys.argv[1]),indent=2))
