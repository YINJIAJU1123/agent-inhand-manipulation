"""Cache six frozen policy queries, preserving dense sensors and separate labels."""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]/'semantic_completion_v2'))
from assets import ADAPTER, STUDENT, save, sha
from features import FrozenFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    encoder = FrozenFeatures(args.device)
    manifest = []
    for path in sorted(Path(args.data).glob('*/data.pt')):
        source = json.loads((path.parent/'manifest.json').read_text())
        digest = sha(path)
        if source['status'] != 'completed' or digest != source['data_sha256']:
            raise RuntimeError(f'Incomplete or changed data: {path}')
        if source['student_sha256'] != sha(STUDENT):
            raise RuntimeError('Student differs from collection')
        data = torch.load(path, map_location='cpu', weights_only=True)
        features = []
        for start in range(0, len(data['rows']), 32):
            features.append(encoder.candidates(
                data['features'][start:start+32].to(args.device),
                data['proprio'][start:start+32].to(args.device)).cpu())
        cache = {key: data[key] for key in (
            'roi_rgb', 'roi_depth', 'roi_box', 'roi_found', 'error', 'rotation_label', 'rows')}
        cache['x'] = torch.cat(features)
        target = out/(path.parent.name+'.pt')
        torch.save(cache, target)
        row = dict(shard=path.parent.name, frames=len(data['rows']), data_sha256=digest,
                   cache_sha256=sha(target))
        manifest.append(row)
        save(out/'progress.json', dict(status='caching', completed=len(manifest), current=row))
        print(json.dumps(row), flush=True)
    if len(manifest) != 10:
        raise RuntimeError('Expected eight train and two validation shards')
    save(out/'manifest.json', dict(status='completed', shards=manifest,
         student_sha256=sha(STUDENT), adapter_sha256=sha(ADAPTER),
         source_sha256=sha(__file__), feature_source_sha256=sha(Path(__file__).resolve().parents[1]/'semantic_completion_v2/features.py'),
         sensor_fields=['x', 'roi_rgb', 'roi_depth', 'roi_box', 'roi_found'],
         supervision_only_fields=['error', 'rotation_label'],
         split_contract='Eight new train seeds, calibration 96311, development audit 96312. No formal test seeds or cloud calls.'))
    save(out/'progress.json', dict(status='completed', completed=len(manifest)))


if __name__ == '__main__':
    main()
