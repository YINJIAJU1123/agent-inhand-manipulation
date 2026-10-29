"""Freeze the newly matched active/center actors, never select on formal outcomes."""
import argparse,hashlib,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--parent_sources',type=Path,required=True);p.add_argument('--working_sources',type=Path,required=True)
    p.add_argument('--training_root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads(a.parent.read_text())
    if {f.name:sha(f) for f in a.parent_sources.glob('*.py')}!=parent['sources_sha256']:
        raise ValueError('Parent sources are not frozen bytes')
    evidence=json.loads((a.training_root/'verified_training_completion.json').read_text())
    if evidence['training_status']!='completed':raise ValueError('Matched training incomplete')
    a.output.mkdir(parents=True,exist_ok=False);dest=a.output/'sources';shutil.copytree(a.parent_sources,dest)
    for name in ('control_modules.py','evaluate_agent.py'):
        shutil.copy2(a.working_sources/name,dest/name)
    guard=dest/'formal_release.py'
    guard.write_text(guard.read_text()+'\n\n'+Path(__file__).with_name('goal_guard_extension.py').read_text())
    spec=importlib.util.spec_from_file_location('goal_guard',guard);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reference=json.loads(parent['parent_release_json'])['conditions']['astra_full']['manifest_contract']
    conditions={}
    for name,(mode,intervention) in m.GOAL_CONDITIONS.items():
        c=json.loads(json.dumps(reference));c['evaluator_sha256']=sha(dest/'evaluate_agent.py')
        c['controller_goal_intervention']=m.goal_intervention_contract(mode,intervention)
        for phase,field in [('waiting','actor_sha256'),('execution','execution_actor_sha256')]:
            key=phase+'_'+mode;ckpt=a.training_root/key/'maintenance_actor.pt'
            if sha(ckpt)!=evidence['models'][key]['sha256']:raise ValueError('Training checkpoint changed')
            if not evidence['models'][key]['completion']['completed']:raise ValueError('Incomplete model')
            target=a.output/'weights'/key;target.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ckpt,target/ckpt.name);c[field]=sha(ckpt)
        conditions[name]={'manifest_contract':c}
    release=dict(format=m.GOAL_ABLATION_FORMAT,status='frozen',created_at=datetime.now(timezone.utc).isoformat(),
        parent_addendum_json=a.parent.read_text(),parent_addendum_sha256=sha(a.parent),cohort=m.GOAL_COHORT,
        frozen_requirements=parent['frozen_requirements'],formal_trials_used_for_selection=False,
        sources_sha256={f.name:sha(f) for f in dest.glob('*.py')},external_sources_sha256=parent['external_sources_sha256'],
        training_evidence=evidence,training_plan_sha256=sha(a.training_root/'training_plan.json'),conditions=conditions,
        declaration='Matched active and center actors retrained with equal budget. Shuffle affects only actor goal. Student target and scoring unchanged. No outcome-based weight selection or replacements.')
    m.validate_contracts(release);(a.output/'formal_release.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'release_sha256':sha(a.output/'formal_release.json'),'conditions':list(conditions)}))
if __name__=='__main__':main()
