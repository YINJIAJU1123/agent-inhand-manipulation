"""Hash-verified object-source relocation for this newly frozen Q1 campaign."""
import hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install(path):
    declaration=json.loads(Path(path).read_text())
    if declaration['scope']=='local_development_only':
        from relocated_environment import install as development_install
        return development_install(path)
    if declaration['scope']!='q1_object_asset_relocation_v1':raise ValueError('Unknown relocation scope')
    inventory=Path(declaration['source_inventory'])
    if sha(inventory)!=declaration['source_inventory_sha256']:raise ValueError('Changed source inventory')
    root=Path(declaration['local_prefix']).resolve()
    for rel,digest in json.loads(inventory.read_text())['files'].items():
        file=(root/rel).resolve()
        if not file.is_relative_to(root) or sha(file)!=digest:raise ValueError('Changed imported source: '+rel)
    old=declaration['original_prefix'];new=str(root)
    if not old.startswith('/') or old==new:raise ValueError('Invalid original prefix')
    def replace(value,source,target):
        if isinstance(value,str):return target+value[len(source):] if value==source or value.startswith(source+'/') else value
        if isinstance(value,dict):return {k:replace(v,source,target) for k,v in value.items()}
        if isinstance(value,list):return [replace(v,source,target) for v in value]
        if isinstance(value,tuple):return tuple(replace(v,source,target) for v in value)
        return value
    for asset in declaration['assets'].values():
        if sha(replace(asset['path'],old,new))!=asset['sha256']:raise ValueError('Changed bound asset')
    import viserdex.distillation.environment as environment
    load_original=environment.load_environment;contract_original=environment.environment_contract
    environment.load_environment=lambda p:replace(load_original(p),old,new)
    environment.environment_contract=lambda data:replace(contract_original(data),new,old)
    return declaration
