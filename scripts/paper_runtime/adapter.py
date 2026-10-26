"""Isolated fitted actors. The stop-evidence model never uses these weights."""
import copy
import json
from pathlib import Path
import torch

SOURCE_SHA = '28f62f3e1f8590cd57479a89aff5d36f054b6c49d19e544a7820d72d10f8d6f7'


class FittedActors:
    def __init__(self, source, original_pack, directory):
        import hashlib
        self.models = {}
        self.metadata = {}
        for mode in ('old_only','mixed'):
            asset = json.loads(Path(directory).read_text())['assets'][{'old_only':'paper_comparison_actor', 'mixed':'adapted_action_student'}[mode]]
            path = Path(__file__).resolve().parents[2]/asset['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256']
            pack = torch.load(path,map_location='cpu',weights_only=True)
            meta = pack['onpolicy_distillation']
            assert meta['format']=='teacher_onpolicy_fit_v1' and meta['arm']==mode
            assert meta['source_student_sha256']==SOURCE_SHA
            assert pack['model'].keys()==original_pack['model'].keys()
            changed = []
            for name,value in pack['model'].items():
                original = original_pack['model'][name].detach().cpu()
                assert value.shape==original.shape and torch.isfinite(value).all()
                if not torch.equal(value,original):
                    assert not name.startswith(('evidence_head.','stop_head.'))
                    changed.append(name)
            assert changed
            for name in ('history','rgb_dim','language_dim','proprio_dim','action_dim','memory_mode','action_scale','freeze_id'):
                assert pack[name]==original_pack[name],name
            model = copy.deepcopy(source)
            model.load_state_dict(pack['model'],strict=True)
            model.eval().requires_grad_(False)
            self.models[mode] = model
            self.metadata[mode] = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_student_sha256=SOURCE_SHA,selected_epoch=meta['selected_epoch'],changed_parameters=changed)

    @torch.inference_mode()
    def proposals(self, original_output, batch):
        # Same computation in every arm; the original source is never changed.
        values = {mode:model(batch)['action'] for mode,model in self.models.items()}
        values['original'] = original_output['action']
        assert all(torch.isfinite(v).all() and v.abs().max()<=1 for v in values.values())
        return values

    @staticmethod
    def select(values, mode, enabled):
        assert mode in values
        if torch.is_tensor(enabled):
            assert enabled.dtype==torch.bool and enabled.shape==values['original'].shape[:1]
            return torch.where(enabled[:,None],values[mode],values['original'])
        return values[mode] if enabled else values['original']
