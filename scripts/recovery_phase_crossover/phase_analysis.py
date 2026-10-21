"""Exploratory, balanced phase-crossover comparisons on already observed seeds."""
import json
from pathlib import Path

METHODS={'v3_v3':('v3','v3'),'v3_new':('v3','new'),'new_v3':('new','v3'),'new_new':('new','new')}

def read(path):return json.loads(path.read_text())
def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def counts(rows):
    rows=[r for r in rows if r['eligible']]
    n=len(rows);success=sum(r['success'] for r in rows)
    return dict(eligible=n,successes=success,success_rate=success/n if n else None,
        wait_drops=sum(r['wait_drop'] for r in rows),recovery_drops=sum(r['resume_drop'] for r in rows),
        recovery_drops_before_handoff=sum(r['resume_drop'] and not r['alive_at_handoff'] for r in rows),
        recovery_drops_after_handoff=sum(r['resume_drop'] and r['alive_at_handoff'] for r in rows),
        non_drop_failures=sum(not(r['success'] or r['wait_drop'] or r['resume_drop']) for r in rows))
def contrasts(methods):
    a,b,c,d=[methods[m]['success_rate'] for m in METHODS]
    if any(x is None for x in [a,b,c,d]):return None
    return dict(recovery_new_given_v3_wait_pp=100*(b-a),recovery_new_given_new_wait_pp=100*(d-c),
        wait_new_given_v3_recovery_pp=100*(c-a),wait_new_given_new_recovery_pp=100*(d-b))
def analyze(out):
    arms=[]
    for p in sorted((out/'eval').glob('*/method.json')):
        if not (p.parent/'complete.json').exists():continue
        m=read(p);rows=read(p.parent/'eval_trials.json');arms.append((m,rows))
    group=lambda m:(m['seed'],m['branch'],m['delay'],m['repeat'])
    groups={group(m) for m,_ in arms}
    balanced={g for g in groups if {m['method'] for m,_ in arms if group(m)==g}==set(METHODS)}
    pairs=[(m,r) for m,rows in arms if group(m) in balanced for r in rows]
    methods={name:counts([r for m,r in pairs if m['method']==name]) for name in METHODS}
    strata={key:{str(v):{name:counts([r for m,r in pairs if m[key]==v and m['method']==name]) for name in METHODS} for v in sorted({m[key] for m,_ in pairs})} for key in ['seed','branch','delay','repeat']}
    result=dict(completed_arms=len(arms),completed_trials=sum(len(rows) for _,rows in arms),balanced_groups=len(balanced),methods=methods,
        success_contrasts=contrasts(methods),stratified=strata,
        interpretation='Exploratory attribution on two previously observed seeds. Same-checkpoint cells are newly rerun controls. No retraining, live agent, independent-scene or significance claim. No automatic baseline replacement.')
    save(out/'summary.json',result);return result
