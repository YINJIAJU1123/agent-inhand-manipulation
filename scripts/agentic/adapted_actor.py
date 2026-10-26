"""Published action-only component of the fixed paper student.

Feedback/stop outputs remain the caller's original model outputs. This component
reproduces proposal selection, not the paper's full asynchronous evaluator.
"""
import copy
import hashlib
from pathlib import Path

import torch

REFERENCE_SHA256 = '28f62f3e1f8590cd57479a89aff5d36f054b6c49d19e544a7820d72d10f8d6f7'
PAPER_ACTOR_SHA256 = 'b02d80830dcd4aebeec14c7d30e8645a47dc168a6ce059165d4744f683359862'


class AdaptedActionProposal:
    def __init__(self, reference, reference_pack, checkpoint):
        path = Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != PAPER_ACTOR_SHA256:
            raise ValueError('Expected the frozen paper action checkpoint')
        pack = torch.load(path, map_location='cpu', weights_only=True)
        metadata = pack['onpolicy_distillation']
        if metadata['format'] != 'teacher_onpolicy_fit_v1' or metadata['arm'] != 'mixed':
            raise ValueError('Unexpected action adaptation format')
        if metadata['source_student_sha256'] != REFERENCE_SHA256:
            raise ValueError('Unexpected reference student')
        for field in ('rgb_dim', 'language_dim', 'proprio_dim', 'action_dim', 'history',
                      'memory_mode', 'action_scale', 'freeze_id'):
            if pack[field] != reference_pack[field]:
                raise ValueError('Reference contract mismatch: '+field)
        actual = reference.state_dict()
        if actual.keys() != reference_pack['model'].keys() or actual.keys() != pack['model'].keys():
            raise ValueError('Model state keys differ')
        for name, tensor in actual.items():
            if not torch.equal(tensor.detach().cpu(), reference_pack['model'][name].cpu()):
                raise ValueError('Reference model was already modified: '+name)
            candidate = pack['model'][name]
            if candidate.shape != tensor.shape or not torch.isfinite(candidate).all():
                raise ValueError('Invalid candidate tensor: '+name)
            if name.startswith(('evidence_head.', 'stop_head.')) and not torch.equal(candidate, tensor.cpu()):
                raise ValueError('Candidate evidence/stop parameters changed: '+name)
        self.model = copy.deepcopy(reference).eval().requires_grad_(False)
        self.model.load_state_dict(pack['model'], strict=True)
        self.checkpoint_sha256 = PAPER_ACTOR_SHA256

    @torch.inference_mode()
    def action(self, original_output, batch, public_skill_delivered, active, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('enabled must be a Python bool')
        original = original_output['action']
        for mask in (public_skill_delivered, active):
            if mask.dtype != torch.bool or mask.shape != original.shape[:1] or mask.device != original.device:
                raise ValueError('Expected one boolean mask value per environment on the action device')
        candidate = self.model(batch)['action']
        if candidate.shape != original.shape or candidate.shape[-1] != 21:
            raise ValueError('Expected matching 21-dimensional proposals')
        if not all(torch.isfinite(t).all() and t.abs().max() <= 1 for t in (original, candidate)):
            raise ValueError('Invalid normalized action proposal')
        selected = public_skill_delivered & active & enabled
        return torch.where(selected[:, None], candidate, original)
