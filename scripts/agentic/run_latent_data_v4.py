"""Fresh simulator per collection seed, bounded watchdog, audited dataset merge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--protocol', default=str(ROOT / 'configs/agentic_two_track_v4.json'))
a = p.parse_args()
root = Path(a.output).resolve()
root.mkdir(parents=True, exist_ok=True)
protocol = json.loads(Path(a.protocol).read_text())
env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='4')
for split, seeds, steps in [('train', protocol['train_seeds'], 300), ('val', protocol['validation_seeds'], 180)]:
    shards = []
    for seed in seeds:
        out = root / 'shards' / f'{split}_{seed}'
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = out / 'manifest.json'
        if not manifest.exists():
            command = [sys.executable, str(ROOT / 'scripts/agentic/collect_latent_data_v4.py'), '--headless',
                       '--output', str(out), '--split', split, '--seeds', str(seed), '--max-steps', str(steps)]
            with (root / f'{split}_{seed}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        info = json.loads(manifest.read_text())
        assert info['status'] == 'completed' and info['seeds'] == [seed] and info['split'] == split
        for name, digest in info['source_sha256'].items():
            assert hashlib.sha256((ROOT / 'scripts/agentic' / name).read_bytes()).hexdigest() == digest
        shards.append(out)
        print(json.dumps({'completed_split': split, 'seed': seed, 'frames': info['frames']}), flush=True)
    dest = root / split
    dest.mkdir(exist_ok=True)
    rows = []
    for shard in shards:
        for line in (shard / 'frames.jsonl').read_text().splitlines():
            row = json.loads(line)
            assert len(row['label_usable']) == 6
            for name in [row['image'], row['history']]:
                if not (dest / name).exists():
                    os.link(shard / name, dest / name)
            rows.append(row)
    (dest / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    meta = json.loads((shards[0] / 'manifest.json').read_text())
    meta.update(seeds=seeds, frames=len(rows), shards=[str(x) for x in shards],
                frames_sha256=hashlib.sha256((dest / 'frames.jsonl').read_bytes()).hexdigest())
    (dest / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'merged': split, 'frames': len(rows)}), flush=True)
