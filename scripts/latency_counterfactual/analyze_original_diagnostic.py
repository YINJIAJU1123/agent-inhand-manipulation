"""Aggregate only complete seeds; pair policies and bootstrap whole reset seeds."""
import argparse,json,random,statistics
from pathlib import Path
MODES=['continue','slow','hold','learned']

def stats(rows):
    valid=[r for r in rows if r['eligible']];survive=[r for r in valid if not r['wait_drop']]
    return dict(nominal=len(rows),eligible=len(valid),prefix_failures=len(rows)-len(valid),
        wait_drops=sum(r['wait_drop'] for r in valid),wait_survival_rate=len(survive)/len(valid) if valid else None,
        recovery_drops=sum(r['resume_drop'] for r in valid),successes=sum(r['success'] for r in valid),
        success_rate=sum(r['success'] for r in valid)/len(valid) if valid else None,
        single_step_reaches=sum(r.get('post_single_step_reach',False) for r in valid) if all('post_single_step_reach' in r for r in valid) else None,
        success_given_wait_survival=sum(r['success'] for r in survive)/len(survive) if survive else None,
        mean_recovery_seconds=statistics.mean(r['resume_steps_to_success']/30 for r in valid if r['success']) if any(r['success'] for r in valid) else None,
        mode_steps=[sum(r['mode_steps'][i] for r in valid) for i in range(3)])

def analyze(root):
    root=Path(root);rows=[];audits=[];seeds=[]
    protocol=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else {}
    original=protocol.get('version')=='original_waiting_v4'
    for path in sorted(root.glob('seed_*')):
        if not (path/'complete.json').exists():continue
        completion=json.loads((path/'complete.json').read_text())
        if original:
            if not completion.get('zero_delay_audited'):raise RuntimeError('Missing zero-delay diagnostic audit')
        elif not completion['zero_delay_consistent']:raise RuntimeError('Invalid zero-delay audit')
        batch=json.loads((path/'eval_trials.json').read_text());rows+=batch;seeds.append(batch[0]['seed'])
        audits+=json.loads((path/'state_audit.json').read_text())
    pairs={}
    for r in rows:
        key=(r['seed'],r['slot'],r['delay_s']);group=pairs.setdefault(key,{})
        if r['mode'] in group:raise RuntimeError('Duplicate paired arm')
        group[r['mode']]=r
    for group in pairs.values():
        if set(group)!=set(MODES):raise RuntimeError('Missing paired arm')
        if len({r['eligible'] for r in group.values()})!=1 or len({r['request_cache_sha256'] for r in group.values()})!=1:raise RuntimeError('Unpaired request state')
    def paired(items,baseline):
        valid=[v for v in items if v['learned']['eligible']];by_seed={}
        for v in valid:
            s=v['learned']['seed'];entry=by_seed.setdefault(s,[0,0]);entry[0]+=int(v['learned']['success'])-int(v[baseline]['success']);entry[1]+=1
        diff=sum(v[0] for v in by_seed.values())/sum(v[1] for v in by_seed.values()) if by_seed else None
        rng=random.Random(91017);values=list(by_seed.values());samples=[]
        if len(values)>1 and not original:
            for _ in range(2000):
                draw=rng.choices(values,k=len(values));samples.append(sum(x[0] for x in draw)/sum(x[1] for x in draw))
        samples.sort()
        return dict(eligible_pairs=len(valid),learned_only_success=sum(v['learned']['success'] and not v[baseline]['success'] for v in valid),
            baseline_only_success=sum(v[baseline]['success'] and not v['learned']['success'] for v in valid),
            success_rate_difference=diff,seed_cluster_bootstrap_95=[samples[49],samples[1949]] if samples else None)
    delays=sorted({r['delay_s'] for r in rows})
    result=dict(completed_seeds=seeds,completed_trials=len(rows),unique_request_states=len({(r['seed'],r['slot']) for r in rows}),
        eligible_unique_request_states=len({(r['seed'],r['slot']) for r in rows if r['eligible']}),
        by_delay={str(d):{m:stats([r for r in rows if r['delay_s']==d and r['mode']==m]) for m in MODES} for d in delays},
        overall={m:stats([r for r in rows if r['mode']==m]) for m in MODES},
        paired_by_delay={str(d):{m:paired([v for k,v in pairs.items() if k[2]==d],m) for m in MODES[:-1]} for d in delays},
        max_restore_error=max((v['max_abs_error'] for v in audits),default=None),
        limitations=['Repeated delays share request states; confidence intervals resample reset seeds.',
        'Known goal and simulated fixed verification delay; excludes semantic grounding, initial unknown goal and live API variation.',
        'Fresh-process exact prefix action replay preserves contact history; request mechanics and full zero-delay traces are audited.',
        'All arms use deterministic ambient-only rendering; appearance differs from the live-v7 benchmark, so rates are not pooled.',
        'Success requires six consecutive post-response steps; percentages differ from live agent reach scoring.'])
    if original:
        result.update(protocol='original_waiting_v4', evidence_level='two-seed exploratory diagnostic after baseline gate warning; no significance or generalization claim',
            zero_delay_audits={str(s):json.loads((root/f'seed_{s}'/'negative_control.json').read_text()) for s in seeds})
        result['engineering_gate_passed']=all(result['zero_delay_audits'][str(s)]['passed'] for s in seeds) if seeds else None
        result['protocol_amendment']=json.loads((root/'diagnostic_amendment.json').read_text())
        result['limitations']=[
            'Initial 50% baseline performance / 25pp repeat-spread engineering checks remain recorded. Failed checks are warnings in this explicitly amended diagnostic continuation, not passed validation.',
            'Only two reset seeds; repeated delays and methods share at most 24 nominal request states. No bootstrap confidence interval is reported.',
            'Known canonical goal with fixed simulated delay; no live Astra or VLM calls and no grounding measurement.',
            'Fresh-process exact action-prefix replay and cached request observations; mechanical errors must be zero.',
            'Original renderer preserved. Subsequent fresh images may differ across repetitions; pairs match request states, not complete random render trajectories.',
            'Primary success requires six consecutive post-response steps within 0.16 rad and no prior drop; single-step reach is secondary.',
            'Zero-delay controls are repeated identical-control experiments; their outcome spread measures nuisance variation.']
    tmp=root/'summary.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'summary.json');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();r=analyze(a.root);print(json.dumps({k:r[k] for k in ['completed_seeds','completed_trials','overall']},indent=2))
