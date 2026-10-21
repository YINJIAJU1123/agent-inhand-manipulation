"""Export only this completed experiment's diagnostic JSON files from NAS."""
import base64
import gzip
import hashlib
import json
from pathlib import Path


root = Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_feedback_hold_20261022/paired_r4')
result = json.loads((root/'result.json').read_text())
assert result['status'] == 'completed' and result['cloud_calls'] == 0
source_bundle = gzip.compress(json.dumps(result).encode(), mtime=0)
expected = '371e2f025a7e3b7ec85258814193cf550f5404674a7dd800c71af212eff6397f'
assert hashlib.sha256(source_bundle).hexdigest() == expected
files = {}
for seed in ('117101', '117102'):
    for mode in ('none', 'hold', 'feedback'):
        for name in ('trace.json', 'retention_events.json', 'shared_state_audit.json', 'prefix_audit.json', 'protocol.json', 'report.json'):
            path = root/'cases'/seed/mode/name
            content = path.read_bytes()
            assert len(content) < 2_000_000
            files[str(path.relative_to(root))] = dict(text=content.decode(), sha256=hashlib.sha256(content).hexdigest())
payload = json.dumps(dict(status='completed', source_result_bundle_sha256=expected,
    files=files, cloud_calls=0, purpose='Read-only export of existing measured feedback traces and evaluator-only diagnostics.')).encode()
assert len(payload) < 6_000_000
blob = gzip.compress(payload, mtime=0)
assert len(blob) < 1_000_000
print('FEEDBACK_TRACES_SHA256 '+hashlib.sha256(blob).hexdigest(), flush=True)
print('FEEDBACK_TRACES_BEGIN', flush=True)
encoded = base64.b64encode(blob).decode()
for i in range(0, len(encoded), 64):
    print(encoded[i:i+64], flush=True)
print('FEEDBACK_TRACES_END', flush=True)
print(json.dumps(dict(status='completed', files=len(files), compressed_bytes=len(blob), cloud_calls=0)), flush=True)
