"""Canonical path relocation for the separately packaged die source environment."""
import argparse,json,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
meta=json.loads((a.bundle/'objects/die/OBJECT.json').read_text());old=meta['canonical_source_prefix'];new=str((a.bundle/'runtime/die').resolve())
def replace(x,src,dst):
    if isinstance(x,str):return dst+x[len(src):] if x==src or x.startswith(src+'/') else x
    if isinstance(x,dict):return {k:replace(v,src,dst) for k,v in x.items()}
    if isinstance(x,list):return [replace(v,src,dst) for v in x]
    if isinstance(x,tuple):return tuple(replace(v,src,dst) for v in x)
    return x
import viserdex.distillation.environment as environment
load,contract=environment.load_environment,environment.environment_contract
environment.load_environment=lambda path:replace(load(path),old,new)
environment.environment_contract=lambda data:replace(contract(data),new,old)
args=a.argv[1:] if a.argv[:1]==['--'] else a.argv
sys.argv=args;sys.path.insert(0,str(Path(args[0]).parent));runpy.run_path(args[0],run_name='__main__')
