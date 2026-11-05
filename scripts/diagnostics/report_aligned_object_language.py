"""Report a separate 720-trial paper-aligned object campaign without rewriting it."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

METHODS=('astra','rule','vlm')
CONDITIONS=('en_standard','en_rewrite','zh_standard','zh_rewrite')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(b,c):
    n=b+c
    return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.


def holm(rows):
    ordered=sorted(rows,key=lambda x:x['p_exact'])
    previous=0.
    for i,row in enumerate(ordered):
        previous=max(previous,min(1.,(len(rows)-i)*row['p_exact']))
        row['p_holm']=previous


def report(base):
    cells={};errors=[];artifacts={}
    obj=read(base/'plan.json')['object']
    for condition in CONDITIONS:
        cases=read(base/'protocol'/f'{obj}_{condition}.json')['cases']
        for method in METHODS:
            rows={}
            for output in sorted((base/'runs'/condition/method).glob('block_*/attempt_*/evaluation')):
                ledger=output/'episodes.jsonl'
                if not ledger.exists():continue
                artifacts[str(ledger.relative_to(base))]=sha(ledger)
                if not (output/'result.json').exists():continue
                result=read(output/'result.json')
                chunk=[json.loads(line) for line in ledger.read_text().splitlines()]
                if result['status'] not in ('completed','service_blocked') or len(chunk)!=result['recorded']:
                    errors.append('Nonterminal or inconsistent output: '+str(output));continue
                for row in chunk:
                    i=row['index']
                    if i in rows:errors.append(f'Duplicate {condition}/{method}/{i}')
                    if i not in range(60):errors.append('Out of range index');continue
                    case=cases[i]
                    for k in ('trial_seed','goal_index','instruction'):
                        if row[k]!=case[k]:errors.append(f'Case mismatch {condition}/{method}/{i}/{k}')
                    expected_success=(row['reason']=='success')
                    if bool(row['success'])!=expected_success:errors.append('Success accounting mismatch')
                    if row['success'] and (row['declared_goal']!=row['goal_index'] or row['final_error_rad']>.16):
                        errors.append('Independent success score mismatch')
                    image=output/f'episode_{i:03d}'/'request_01.png'
                    row['_first_image_sha256']=sha(image) if image.exists() else None
                    rows[i]=row
            cells[f'{condition}/{method}']=dict(recorded=len(rows),success=sum(r['success'] for r in rows.values()),
                reasons=dict(Counter(r['reason'] for r in rows.values())),
                drop_phases=dict(Counter(r.get('drop_phase') for r in rows.values() if r['reason']=='drop')),
                excluded=sum(r.get('exclusion_reason') is not None for r in rows.values()),rows=rows)
    total=sum(c['recorded'] for c in cells.values());complete=total==720 and not errors
    comparisons=[]
    def compare(name,left,right,family):
        a,b=cells[left]['rows'],cells[right]['rows'];eligible=[];exclusions=[]
        for i in range(60):
            causes=[]
            if any(r.get('exclusion_reason') is not None for r in (a[i],b[i])):causes.append('service')
            if a[i]['initial_state']['sha256']!=b[i]['initial_state']['sha256']:causes.append('initial_state')
            if not a[i]['_first_image_sha256'] or a[i]['_first_image_sha256']!=b[i]['_first_image_sha256']:causes.append('first_image')
            if causes:exclusions.append(dict(slot=i,reasons=causes))
            else:eligible.append(i)
        only_a=sum(a[i]['success'] and not b[i]['success'] for i in eligible)
        only_b=sum(b[i]['success'] and not a[i]['success'] for i in eligible)
        comparisons.append(dict(name=name,family=family,left=left,right=right,n=len(eligible),exclusions=exclusions,
            left_success=sum(a[i]['success'] for i in eligible),right_success=sum(b[i]['success'] for i in eligible),
            left_only=only_a,right_only=only_b,p_exact=exact_mcnemar(only_a,only_b),
            sensitivity_analysis=any(any(r!='service' for r in x['reasons']) for x in exclusions)))
    if complete:
        for method in METHODS:
            for a,b in [('en_standard','zh_standard'),('en_rewrite','zh_rewrite'),('en_standard','en_rewrite'),('zh_standard','zh_rewrite')]:
                compare(f'{method}:{a}-{b}',f'{a}/{method}',f'{b}/{method}','language_and_wording')
        for condition in CONDITIONS:
            for a,b in itertools.combinations(METHODS,2):
                compare(f'{condition}:{a}-{b}',f'{condition}/{a}',f'{condition}/{b}','methods')
        for family in ('language_and_wording','methods'):holm([x for x in comparisons if x['family']==family])
    for cell in cells.values():del cell['rows']
    out=dict(status='invalid' if errors else 'complete_with_pairing_sensitivity' if any(x['sensitivity_analysis'] for x in comparisons) else 'complete' if complete else 'in_progress',
        recorded=total,planned=720,cells=cells,errors=errors,comparisons=comparisons,ledger_sha256=artifacts,
        plan_sha256=sha(base/'plan.json'),scope='Separate object-specific cohort; raw service failures retained; exact pairing checks and per-family Holm; no historical pooling')
    analysis=base/'analysis';analysis.mkdir(exist_ok=True)
    (analysis/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=[f'# Paper-aligned {obj} bilingual rerun','',f'Status: {out["status"]}; recorded {total}/720.','',
           '| Method | EN standard | EN rewrite | ZH standard | ZH rewrite |','| --- | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        values=[f'{cells[c+"/"+method]["success"]}/60' if cells[c+'/'+method]['recorded']==60 else f'pending ({cells[c+"/"+method]["recorded"]}/60 recorded)' for c in CONDITIONS]
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines+=['','Raw denominator retains service exclusions. All four prompts reuse task states; do not pool them as independent samples.',
        'Any initial-state or first-image mismatch is explicitly excluded only in a labelled sensitivity analysis. See results.json for every contrast, exclusion and corrected p-value.',
        'This cohort uses the selected official account and paper-selected object-specific controllers and stopping rules. Shared GPU load and end-to-end SSH/API latency are part of deployment; historical cohorts remain separate.']
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('status','recorded','planned','errors')}))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    args=p.parse_args();report(args.base)
