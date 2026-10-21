"""Matched factorial contrasts with explicit small-sample limitations.

Distinct module name avoids the legacy runner's analyze import.
"""
import json,collections
from pathlib import Path
from design import ARMS
METHODS=['slow','v3','legacy_v4',*ARMS]

def read(path):return json.loads(path.read_text())
def save(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def counts(rows):
 good=[r for r in rows if r['eligible']]
 return dict(nominal=len(rows),eligible=len(good),successes=sum(r['success'] for r in good),
  success_rate=sum(r['success'] for r in good)/len(good) if good else None,
  wait_drops=sum(r['wait_drop'] for r in good),recovery_drops=sum(r['resume_drop'] for r in good),
  non_drop_failures=sum(not(r['success'] or r['wait_drop'] or r['resume_drop']) for r in good))
def effects(methods):
 u,p,w,b=[methods[m]['success_rate'] for m in ARMS]
 if any(v is None for v in [u,p,w,b]):return None
 return dict(pairing_without_weights_pp=100*(p-u),pairing_with_weights_pp=100*(b-w),
  weights_without_pairing_pp=100*(w-u),weights_with_pairing_pp=100*(b-p),interaction_pp=100*(b-p-w+u))

def analyze(out):
 out=Path(out);arms=[]
 for f in sorted((out/'eval').glob('*/method.json')):
  folder=f.parent
  if not (folder/'complete.json').exists():continue
  m=read(f);rows=read(folder/'eval_trials.json');a=read(folder/'goal_switch_audit.json');prefix=read(folder/'prefix_replay_audit.json')
  if len(rows)!=12 or a['max_physical_change']!=0 or prefix['max_abs_error']!=0 or not prefix['eligible_equal']:raise RuntimeError('Invalid completed evaluation')
  if [r['goal'] for r in rows]!=a['goals']:raise RuntimeError('Scored goal mismatch')
  arms.append((m,rows,folder))
 group=lambda m:(m['seed'],m['branch'],m['delay'],m['repeat'])
 groups={group(m) for m,_,_ in arms if m['delay']>0}
 balanced={g for g in groups if {m['method'] for m,_,_ in arms if group(m)==g}==set(METHODS)}
 pairs=[(m,r) for m,rows,_ in arms if group(m) in balanced for r in rows]
 pooled={method:counts([r for m,r in pairs if m['method']==method]) for method in METHODS}
 by_group={str(g):{method:counts([r for m,r in pairs if group(m)==g and m['method']==method]) for method in METHODS} for g in sorted(balanced)}
 strata={key:{str(value):{method:counts([r for m,r in pairs if m[key]==value and m['method']==method]) for method in METHODS} for value in sorted({m[key] for m,_ in pairs})} for key in ['seed','branch','delay','repeat']}
 report=dict(completed_arms=len(arms),completed_trials=sum(len(rr) for _,rr,_ in arms),balanced_groups=[list(g) for g in sorted(balanced)],
  methods=pooled,by_group=by_group,stratified=strata,factorial_success_effects=effects(pooled),
  factorial_effects_by_seed={seed:effects(d) for seed,d in strata['seed'].items()},
  zero_controls=[dict(**m,**counts(rr)) for m,rr,_ in arms if m['delay']==0],
  distinct_eligible_states=len({(m['seed'],r['slot']) for m,r in pairs if r['eligible']}),
  interpretation=['Diagnostic factorial study with one stochastic training realization per branch and only two evaluation seeds.',
   'Continued training baseline holds the original goal. Paired training exposes each same state to six correct goals.',
   'Failure weights and state schedule are frozen from preceding TRAINING records and are shared across factor levels.',
   'Explicit weighting changes losses without duplicating data. Accepted PPO steps can still differ because valid horizons and KL guards differ.',
   'Legacy V4 is a reference, not the factorial both-on cell. Its original adaptive sampling and duplication implementation differ.',
   'No live VLM/Astra calls, no automatic baseline replacement, no significance or independent-scene claim.'])
 save(out/'summary.json',report)
 if len(balanced)==16 and len(arms)==120:
  diagnostics={}
  for method in METHODS:
   rr=[r for m,r in pairs if m['method']==method and r['eligible']]
   values={}
   for flag,key in [('alive_at_response','orientation_error_at_response'),('alive_at_handoff','orientation_error_at_handoff')]:
    included=[r[key] for r in rr if r.get(flag) and r.get(key) is not None]
    values[key]=dict(count=len(included),mean=sum(included)/len(included) if included else None)
   rates=collections.defaultdict(list)
   for m,_,folder in arms:
    if m['method']!=method or group(m) not in balanced:continue
    for d in read(folder/'decisions.json'):
     phase='wait' if d['step']<m['delay']*30 else 'recovery'
     rates[phase].extend(d['rate'][i] for i,on in enumerate(d['active']) if on)
   values['rates']={phase:dict(active_decisions=len(v),mean=sum(v)/len(v)) for phase,v in rates.items() if v}
   diagnostics[method]=values
  save(out/'diagnosis.json',dict(status='completed_diagnostic',effects=effects(pooled),effects_by_seed=report['factorial_effects_by_seed'],
   recovery_diagnostics=diagnostics,conditional_error_note='Response/handoff error means include only still-active surviving cases and are selection-conditioned. Not causal evidence by themselves.',
   baseline_decision='Keep frozen V3. Review effect directions across seeds/goal branches, drops and non-drop failures before changing the recovery policy.',
   next_strategy_candidates=['If weighting regresses both pairing conditions, remove weighting before adding new recovery mechanisms.',
    'If pairing regresses at both weight levels, examine goal-change coverage and base-student goal-switch reachability.',
    'If all continued-training arms regress, test an actor anchor to V3 to control global speed/residual drift.',
    'If failure concentrates after response, isolate waiting and recovery modules before changing the handoff schedule.']))
 return report
