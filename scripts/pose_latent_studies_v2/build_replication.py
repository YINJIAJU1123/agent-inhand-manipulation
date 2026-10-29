"""Register a new 40-trial cohort without changing frozen cube-v1 evaluation."""
import argparse, hashlib, importlib.util, json, shutil
from datetime import datetime, timezone
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 parent=json.loads(a.parent.read_text())
 sources={f.name:sha(f) for f in a.sources.glob('*.py')}
 if sources!=parent['sources_sha256']:raise ValueError('Not the exact frozen addendum source directory')
 a.output.mkdir(parents=True,exist_ok=False);dest=a.output/'sources';shutil.copytree(a.sources,dest)
 extension=Path(__file__).with_name('replication_guard_extension.py').read_text()
 guard=dest/'formal_release.py';guard.write_text(guard.read_text()+'\n\n'+extension)
 release=dict(format='cube_v1_independent_replication_v1',status='frozen',created_at=datetime.now(timezone.utc).isoformat(),
  parent_addendum_json=a.parent.read_text(),parent_addendum_sha256=sha(a.parent),new_weight_selection=False,
  cohort=dict(seeds=[183411,183412],episodes_per_seed=20,raw_trials_per_condition=40),
  frozen_requirements=parent['frozen_requirements'],sources_sha256={f.name:sha(f) for f in dest.glob('*.py')},
  external_sources_sha256=parent['external_sources_sha256'],
  declaration='Independent new initial-state cohort, unchanged cube-v1 weights and scoring. Natural latency. No trial replacements or selection from these outcomes.',
  conditions=dict(astra_full=json.loads(parent['parent_release_json'])['conditions']['astra_full'],
   finetuned_vlm_bare=parent['conditions']['finetuned_vlm_bare'],rule_vlm_bare=parent['conditions']['rule_vlm_bare']))
 spec=importlib.util.spec_from_file_location('replication_guard',guard);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.validate_contracts(release)
 (a.output/'formal_release.json').write_text(json.dumps(release,indent=2)+'\n')
 print(json.dumps({'output':str(a.output),'release_sha256':sha(a.output/'formal_release.json'),'conditions':list(release['conditions']),'trials':120}))
if __name__=='__main__':main()
