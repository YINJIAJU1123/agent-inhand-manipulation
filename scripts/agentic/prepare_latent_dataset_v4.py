"""Calibrate visibility thresholds to the actual 7.5 mm black ink radius.

The collector projects an 18 mm outline. Its black ink has radius 7.5 mm;
normalize counted dark/neutral pixels by ink area, not the 9 mm white outline.
This fixed geometric correction was selected on collection probes, not tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
source = Path(a.input).resolve()
root = Path(a.output).resolve()
fractions = [.315, .78, .3825, .4375, .4, .3825]
for split in ['train', 'val']:
    src, dest = source / split, root / split
    manifest = json.loads((src / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'frames.jsonl').exists():
        raise RuntimeError('Refusing to overwrite prepared data')
    rows = []
    for line in (src / 'frames.jsonl').read_text().splitlines():
        row = json.loads(line)
        for color, symbol in enumerate(row['layout']):
            expected_ink = row['projected_area'][color] * fractions[symbol] * (.0075 / .009) ** 2
            pixels = row['symbol_pixels'][symbol]
            row['readable'][color] = row['facing'][color] >= .5 and pixels >= max(35, .65 * expected_ink)
            row['label_usable'][color] = row['readable'][color] or pixels <= 3 or row['facing'][color] <= 0
        for name in [row['image'], row['history']]:
            os.link(src / name, dest / name)
        rows.append(row)
    data = ''.join(json.dumps(row) + '\n' for row in rows)
    (dest / 'frames.jsonl').write_text(data)
    manifest.update(raw_frames_sha256=manifest['frames_sha256'], frames_sha256=hashlib.sha256(data.encode()).hexdigest(),
                    label_rule='facing >= .5 and visible black/white change pixels >= max(35, .65 * projected ink area); clearly absent <=3 pixels or backfacing; discard ambiguous',
                    label_preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    counts = {name: 0 for name in rows[0]['symbols']}
    for row in rows:
        for color, readable in enumerate(row['readable']):
            if readable: counts[row['symbols'][row['layout'][color]]] += 1
    print(json.dumps({'split': split, 'frames': len(rows), 'readable_targets': counts}), flush=True)
