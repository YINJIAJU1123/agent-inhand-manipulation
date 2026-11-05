"""Verify the separately delivered files without importing simulation or model code."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def verify(bundle):
    bundle=bundle.resolve();manifest=bundle/'MANIFEST.json';reference=json.loads((ROOT/'configs/bundle_reference.json').read_text())
    if digest(manifest)!=reference['manifest_sha256']:raise ValueError('Bundle version does not match this code branch')
    files=json.loads(manifest.read_text())['files'];errors=[]
    for rel,entry in files.items():
        p=(bundle/rel).resolve()
        if not p.is_relative_to(bundle) or not p.is_file() or p.stat().st_size!=entry['bytes'] or digest(p)!=entry['sha256']:errors.append(rel)
    if errors:raise ValueError('Missing/changed bundle files: '+', '.join(errors[:20]))
    print(json.dumps({'verified_files':len(files),'objects':reference['objects'],'bytes':sum(v['bytes'] for v in files.values())}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();verify(a.bundle)
