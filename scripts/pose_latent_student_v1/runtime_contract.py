"""Check a prospective runtime subset before the first reset/model request."""
import json
from pathlib import Path


def check_subset(expected, actual, prefix='runtime'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f'{prefix}: expected object')
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f'{prefix}.{key}: missing')
            check_subset(value, actual[key], prefix + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f'{prefix}: sequence mismatch')
        for index, (left, right) in enumerate(zip(expected, actual)):
            check_subset(left, right, f'{prefix}[{index}]')
    elif actual != expected:
        raise ValueError(f'{prefix}: expected {expected!r}, observed {actual!r}')


def verify(path, manifest):
    contract = json.loads(Path(path).read_text())
    if contract.get('format') != 'cube_runtime_subset_v1':
        raise ValueError('Unknown runtime contract')
    check_subset(contract['expected'], manifest)
