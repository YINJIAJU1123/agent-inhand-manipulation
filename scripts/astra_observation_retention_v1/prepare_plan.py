"""Select observation waits using only commanded skill and frozen sensor evidence."""
import argparse, hashlib, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--replays', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
plan, audit, counts = [], [], {}
for shard in sorted(a.replays.glob('*/data.pt')):
    data = torch.load(shard, map_location='cpu', weights_only=True)
    manifest = json.loads((shard.parent / 'manifest.json').read_text())
    assert manifest['replay_max_error'] <= 1e-6
    source = Path(manifest['source'])
    report = json.loads((source / 'report.json').read_text())
    seen = set()
    for n, row in enumerate(data['rows']):
        case, color, step = row['case'], row['skill'], row['step']
        events = report['episodes'][case]['async_events']
        waits = [e for e in events if e.get('request_step', 10**9) <= step < e.get('delivery_step', -1)]
        if case in seen or row['state'] != 'inspect' or color is None or not waits:
            continue
        if row['probability'][color] < .7:
            continue
        seen.add(case)
        item = dict(source=str(source), case=case, step=step, goal=color)
        plan.append(item)
        # Labels are recorded only after selection, never used to select rows.
        audit.append(dict(**item, probability=row['probability'][color],
                          measured_orientation_error_for_audit=float(data['error'][n, color]),
                          request_step=waits[0]['request_step'], delivery_step=waits[0]['delivery_step'],
                          original_reason=report['episodes'][case]['reason']))
    counts[source.name] = dict(episodes=len(report['episodes']), selected=len(seen))
protocol = dict(episodes=sum(v['episodes'] for v in counts.values()), selected=len(plan),
                counts=counts, rates=[1., .25, 0.], horizon=180, cloud_calls=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selection='First sampled inspect skill with frozen pose probability >=0.7 while an original request is pending; at most one state per episode, all episodes included.',
                scope='Development intervention on observation poses, not semantic task success or live Astra responses.',
                replay_gate='Original physical prefix must match to 1e-6; all three intervention states must match to 1e-6. No tolerance relaxation.')
for name, value in [('plan.json', plan), ('audit.json', audit), ('protocol.json', protocol)]:
    (a.output / name).write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps(protocol, indent=2))
