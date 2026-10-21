"""Cache the existing dense verifier's features without a second forward pass."""
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/astra_completion_v3'))
from completion_runtime import DenseEvidenceRuntime
from retention_runtime import measured_feature_bank


class CachedDenseEvidence(DenseEvidenceRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retention_features = None

    @torch.inference_mode()
    def update(self, rgb, features, proprio, depth):
        cache = {}
        def query_hook(module, inputs, output):
            cache['query'] = output
        def model_hook(module, inputs, output):
            cache['candidates'] = inputs[0]
            cache['output'] = output
        hooks = [self.model.query.register_forward_hook(query_hook),
                 self.model.register_forward_hook(model_hook)]
        try:
            super().update(rgb, features, proprio, depth)
        finally:
            for hook in hooks:
                hook.remove()
        self.retention_features = measured_feature_bank(cache['query'], cache['output'], cache['candidates'])
