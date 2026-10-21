"""Stage the prior explicit simulation allowlist plus the new collector."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
old = ROOT/'infra/astra_symbol_regions_20261022/code_train'
files = json.loads((old/'package_manifest.json').read_text())['sha256']
parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='code')
args = parser.parse_args()
if not args.stage.startswith('code') or '/' in args.stage:
    raise ValueError('Use a local code staging directory')
stage = HERE/args.stage
stage.mkdir(exist_ok=False)
allowed = ('source/', 'scripts/', 'assets/', 'configs/', 'outputs/')
copied = {}
for name, digest in files.items():
    if not name.startswith(allowed):
        continue
    source = ROOT/name
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen dependency changed: '+name)
    target = stage/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
for directory in ['scripts/astra_symbol_trajectory_v1', 'scripts/astra_observation_control_v1']:
    for source in (ROOT/directory).glob('*.py'):
        target = stage/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
shutil.copyfile(HERE/'run_cloud.py', stage/'run_cloud.py')
shutil.copyfile(old/'entrypoint.sh', stage/'entrypoint.sh')
for path in sorted(stage.rglob('*')):
    if path.is_file():
        copied[str(path.relative_to(stage))] = hashlib.sha256(path.read_bytes()).hexdigest()
(stage/'package_manifest.json').write_text(json.dumps(dict(sha256=copied, cloud_calls=0), indent=2)+'\n')
print(json.dumps(dict(files=len(copied), bytes=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()), cloud_calls=0)))
