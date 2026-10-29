"""Explicit development-only relocation of identical frozen environment assets.

Original YAML bytes and checkpoint hashes remain unchanged. Only filesystem
prefixes are relocated, and contract comparison canonicalizes them back to the
original prefix. Each declared robot/object asset must match its frozen hash.
"""
import hashlib,json
from pathlib import Path


def install(path):
    import viserdex.distillation.environment as environment
    declaration=json.loads(Path(path).read_text())
    if declaration['scope']!='local_development_only':raise ValueError('Relocation is not admitted to formal trials')
    old=declaration['original_prefix'];new=declaration['local_prefix']
    if not old.startswith('/') or not new.startswith('/') or old==new:raise ValueError('Invalid relocation')
    def replace(value,source,destination):
        if isinstance(value,str):
            return destination+value[len(source):] if value==source or value.startswith(source+'/') else value
        if isinstance(value,dict):return {k:replace(v,source,destination) for k,v in value.items()}
        if isinstance(value,list):return [replace(v,source,destination) for v in value]
        if isinstance(value,tuple):return tuple(replace(v,source,destination) for v in value)
        return value
    for asset in declaration['assets'].values():
        local=Path(replace(asset['path'],old,new))
        if hashlib.sha256(local.read_bytes()).hexdigest()!=asset['sha256']:
            raise ValueError('Relocated physical asset changed: '+str(local))
    load_original=environment.load_environment;contract_original=environment.environment_contract
    def relocated_load(path):return replace(load_original(path),old,new)
    def canonical_contract(data):return replace(contract_original(data),new,old)
    environment.load_environment=relocated_load
    environment.environment_contract=canonical_contract
    return declaration
