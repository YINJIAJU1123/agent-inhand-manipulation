"""CPU-only release inspection. Does not establish simulator or task performance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    sys.path.insert(0, str(root / 'source/viserdex'))
    import torch
    from viserdex.supermimic.core import Config
    from viserdex.supermimic.networks import Actor

    manifest = json.loads((root / 'checkpoints/manifest.json').read_text())
    path = root / manifest['file']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['sha256'], 'Published checkpoint hash mismatch'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    cfg = Config(**payload['config'])
    actor = Actor(cfg).eval()
    actor.load_state_dict(payload['actor'], strict=True)
    torch.set_num_threads(2)
    with torch.inference_mode():
        output = actor(actor.obs_norm.mean.expand(2, cfg.history, 124).clone())
    assert output.shape == (2, 21) and output.isfinite().all()
    report = {
        'source_repository': 'https://github.com/KeZhang19/supermimic',
        'source_commit': subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': digest,
        'iteration': payload['iteration'],
        'payload_keys': list(payload),
        'config': payload['config'],
        'tensor_shapes': {k: list(v.shape) for k, v in payload['actor'].items()},
        'parameters': sum(p.numel() for p in actor.parameters()),
        'strict_load': True,
        'cpu_mean_input_forward_finite': True,
        'output_shape': list(output.shape),
        'direct_replacement_for_pose_latent_student_v1': False,
        'reasons': [
            'SuperMimic Actor state_dict differs from recurrent BeliefStudent.',
            'Input is history10 x obs124 including target and relative quaternions.',
            'Current latent student consumes proprio109, measured pose7, latent768.',
            'Released sphere runs at 20 Hz; frozen cube runtime enforces 30 Hz.',
            'Release includes no sphere RGB pose-estimator checkpoint.',
        ],
        'scope': 'Hash, strict CPU loading and mean-input network forward only. '
                 'No simulation, RGB perception validation or semantic success estimate.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'source_commit', 'checkpoint_sha256', 'strict_load',
        'cpu_mean_input_forward_finite', 'direct_replacement_for_pose_latent_student_v1')}, indent=2))


if __name__ == '__main__':
    main()
