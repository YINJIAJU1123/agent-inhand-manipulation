"""Frozen dense pose verifier, conditional on Astra's semantic hypothesis.

Research candidate only: its original offline gate remains failed. This adapter
never infers symbol-to-color binding and never consumes evaluator labels.
"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT/'scripts/semantic_completion_v3'),
                 str(ROOT/'scripts/semantic_completion_v2')])
from dense_model import load_model
from sensors import object_regions
from features import FrozenFeatures
from assets import STUDENT, ADAPTER, COLORS, sha


class DenseEvidenceRuntime:
    def __init__(self, path, device, n):
        self.model, self.pack = load_model(path, device)
        manifest = self.pack['feature_manifest']
        for path, key in ((STUDENT, 'student_sha256'), (ADAPTER, 'adapter_sha256')):
            if sha(path) != manifest[key]:
                raise RuntimeError('Frozen feature asset mismatch')
        if not self.pack['dense'] or self.pack['source'] != 'dense_completion_v3':
            raise RuntimeError('Expected the dense completion V3 checkpoint')
        if not self.pack.get('calibration'):
            raise RuntimeError('No calibration threshold exists')
        self.encoder = FrozenFeatures(device)
        self.threshold = float(self.pack['threshold'])
        self.last = None
        self.visits = torch.zeros(n, 6, device=device)

    @torch.inference_mode()
    def update(self, rgb, features, proprio, depth):
        regions = object_regions(rgb, depth)
        output = self.model(self.encoder.candidates(features, proprio),
            regions['roi_rgb'], regions['roi_depth'], regions['roi_box'], regions['roi_found'])
        self.last = output['ordinal'][..., 2].sigmoid()
        if not torch.isfinite(self.last).all():
            raise RuntimeError('Nonfinite completion prediction')

    def can_stop(self, case, color, symbol=None, instruction=None):
        # Semantic correctness remains Astra's responsibility and is scored
        # separately. Never consult a hidden layout or evaluator target here.
        if self.last is None or color is None or not 0 <= color < 6:
            return False
        return bool(self.last[case, color] >= self.threshold)

    def summary(self, case, verification, observation=False):
        if not verification or self.last is None:
            return {}
        return {'local_pose_verifier': {
            'colors': list(COLORS),
            'completion_probability_by_color': self.last[case].cpu().tolist(),
            'threshold': self.threshold,
            'scope': 'Pose only, conditional on your color choice. Does not verify symbol binding. Learned estimates can be wrong.'}}
