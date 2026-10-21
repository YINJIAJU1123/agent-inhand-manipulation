"""Retrieve the bounded trained RGB reader from its authenticated DLC job logs."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    command = ['/home/yin/.local/bin/bctrl', '--region', 'cn-wulanchabu', 'dlc', 'logs', args.job_id, '--no-follow', '--lines', '100000']
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    text = re.sub(r'\x1b\[[0-9;]*m', '', reply.stdout)
    begin, end = 'SYMBOL_MODEL_BEGIN', 'SYMBOL_MODEL_END'
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('Expected exactly one complete model artifact')
    encoded = ''.join(text.split(begin, 1)[1].split(end, 1)[0].split())
    if len(encoded) > 4000000:
        raise ValueError('Model artifact exceeds declared transfer bound')
    payload = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(payload).hexdigest()
    expected = re.search(r'SYMBOL_MODEL_SHA256\s+([a-f0-9]{64})', text)
    if len(payload) >= 3000000 or expected is None or digest != expected.group(1):
        raise ValueError('Model artifact checksum or size mismatch')
    allowed = {'head.pt', 'result.json', 'protocol.json', 'history.json'}
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(allowed) or {m.name for m in members} != allowed:
            raise ValueError('Unexpected model artifact contents')
        if any(not m.isfile() or m.size >= 3000000 for m in members) or sum(m.size for m in members) > 5000000:
            raise ValueError('Invalid artifact member type or size')
        for member in members:
            files[member.name] = archive.extractfile(member).read()
    result = json.loads(files['result.json'])
    if hashlib.sha256(files['head.pt']).hexdigest() != result['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from recorded result')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (args.output / name).write_bytes(value)
    (args.output / 'retrieval.json').write_text(json.dumps(dict(job_id=args.job_id, artifact_sha256=digest, bytes=len(payload), verified=True), indent=2) + '\n')
    print(json.dumps(dict(job_id=args.job_id, verified=True, offline_gate_passed=result['offline_gate_passed'], output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
