"""Merge a completed baseline and learned evaluation; audit paired reset states."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--learned',required=True);p.add_argument('--output',required=True);a=p.parse_args()
roots=[Path(a.baseline),Path(a.learned)]
for root in roots:
    if not (root/'complete.json').exists():raise RuntimeError(f'Incomplete evaluation: {root}')
manifests=[json.loads((root/'manifest.json').read_text()) for root in roots]
for key in ('student_sha256','adapter_sha256','step_dt','gate_stride','camera_stride','slow_target_interpolation','success'):
    if manifests[0][key]!=manifests[1][key]:raise RuntimeError(f'Protocol mismatch: {key}')
for key in ('num_envs','resume_steps','eval_seeds','eval_delays'):
    if manifests[0]['args'][key]!=manifests[1]['args'][key]:raise RuntimeError(f'Evaluation mismatch: {key}')
rows=sum([json.loads((root/'eval_trials.json').read_text()) for root in roots],[])
lookup={}
for row in rows:
    key=(row['seed'],row['delay_s'],row['slot'])
    if row['mode'] in lookup.setdefault(key,{}):raise RuntimeError('Duplicate scored case')
    lookup[key][row['mode']]=row
modes=['continue','slow','hold','learned']
if not all(set(v)==set(modes) for v in lookup.values()):raise RuntimeError('Missing paired mode')

def stats(items):
    eligible=[r for r in items if r['eligible']]
    return dict(trials=len(items),eligible=len(eligible),prefix_failures=len(items)-len(eligible),
        successes=sum(r['success'] for r in items),success_rate=sum(r['success'] for r in items)/max(len(eligible),1),
        unconditional_success_rate=sum(r['success'] for r in items)/max(len(items),1),
        wait_drops=sum(r['wait_drop'] for r in items),resume_drops=sum(r['resume_drop'] for r in items),
        mode_steps=[sum(r['mode_steps'][i] for r in items) for i in range(3)])
result={'overall':{m:stats([r for r in rows if r['mode']==m]) for m in modes},
    'by_delay':{str(d):{m:stats([r for r in rows if r['mode']==m and r['delay_s']==d]) for m in modes}
                for d in sorted({r['delay_s'] for r in rows})},
    'paired':{},'state_audit':[],
    'limitations':['Two independent reset seeds: exploratory only; no claim of statistical significance.',
    'Simulated verification response for an already delivered goal; no live Astra.',
    'Held joint command is a prescribed primitive, not a trained hold policy.']}
common=[v for v in lookup.values() if all(v[m]['eligible'] for m in modes)]
for mode in modes[:-1]:
    result['paired'][mode]={'common_eligible':len(common),
      'learned_only_success':sum(v['learned']['success'] and not v[mode]['success'] for v in common),
      'baseline_only_success':sum(v[mode]['success'] and not v['learned']['success'] for v in common),
      'both_success':sum(v[mode]['success'] and v['learned']['success'] for v in common)}
for seed,delay in sorted({(r['seed'],r['delay_s']) for r in rows}):
    states={m:torch.load(roots[1 if m=='learned' else 0]/f'initial_{seed}_{delay:g}_{m}.pt',map_location='cpu',weights_only=False) for m in modes}
    base=states['continue']
    for mode in modes[1:]:
        mask=base['eligible']&states[mode]['eligible']
        result['state_audit'].append({'seed':seed,'delay':delay,'mode':mode,
            'reset_max_abs_difference':float((base['reset']-states[mode]['reset']).abs().max()),
            'request_max_abs_difference_common_eligible':float((base['request'][mask]-states[mode]['request'][mask]).abs().max()) if mask.any() else None,
            'eligible_mask_equal':bool(torch.equal(base['eligible'],states[mode]['eligible']))})
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['overall'],indent=2))
