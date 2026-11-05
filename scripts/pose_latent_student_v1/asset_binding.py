"""Hash-preserving relocation for the frozen local RSLCube runtime assets."""
from pathlib import Path


OLD_PREFIX = '/home/brainco/agentic_rotation_newton/prepared_release_20261029'


def install(environment_yaml, local_prefix):
    import viserdex.distillation.environment as environment

    old = OLD_PREFIX
    new = str(Path(local_prefix).resolve())
    if old == new:
        raise ValueError('Relocation source and destination must differ')

    def replace(value, source, target):
        if isinstance(value, str):
            return target + value[len(source):] if value == source or value.startswith(source + '/') else value
        if isinstance(value, dict):
            return {key: replace(item, source, target) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item, source, target) for item in value]
        if isinstance(value, tuple):
            return tuple(replace(item, source, target) for item in value)
        return value

    original_loader = environment.load_environment
    original_contract = environment.environment_contract
    def relocated_loader(path):
        return replace(original_loader(path), old, new)

    def canonical_contract(data):
        return replace(original_contract(data), new, old)

    environment.load_environment = relocated_loader
    environment.environment_contract = canonical_contract
    return dict(source_prefix=old, local_prefix=new,
                environment_yaml_sha256=__import__('hashlib').sha256(Path(environment_yaml).read_bytes()).hexdigest())
