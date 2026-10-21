"""Request-scoped action retention. No task scoring or semantic inference here."""
from pathlib import Path
import hashlib
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'scripts/astra_retention_v1'))
from value_model import RetentionValue, RATES
from window import RetentionWindow


def load_value_head(path, evidence_path, device, require_gate=True):
    path = Path(path)
    pack = torch.load(path, map_location=device, weights_only=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if pack['base_sha256'] != digest(evidence_path):
        raise RuntimeError('Retention and pose encoder checkpoints do not match')
    if tuple(pack['rates']) != RATES:
        raise RuntimeError('Retention action ordering does not match training')
    if require_gate:
        result = json.loads((path.parent / 'result.json').read_text())
        if not result['offline_gate_passed'] or result['checkpoint_sha256'] != digest(path):
            raise RuntimeError('Retention head has not passed its held-out local gate')
    model = RetentionValue().to(device).eval()
    model.load_state_dict(pack['model'])
    model.requires_grad_(False)
    return model


def measured_feature_bank(query_hidden, output, candidates):
    """Same 125 measured features as frozen_features(), for all six hypotheses."""
    bank = torch.cat((query_hidden, output['ordinal'],
                      output['log_error'][..., None], candidates[..., 533:554]), -1)
    if bank.ndim != 3 or bank.shape[1:] != (6, 125) or not torch.isfinite(bank).all():
        raise RuntimeError('Invalid retention sensor features')
    return bank


class RetentionRuntime:
    def __init__(self, cases, mode='none', model=None, threshold=.7, horizon=180):
        if mode not in ('none', 'hold', 'learned'):
            raise ValueError('Unsupported retention mode')
        if mode == 'learned' and model is None:
            raise ValueError('Learned retention requires a trained value head')
        self.mode, self.model, self.threshold = mode, model, threshold
        self.horizon = horizon
        self.windows = [RetentionWindow(horizon=horizon) for _ in range(cases)]
        self.requests = [None] * cases
        self.events = []

    @torch.inference_mode()
    def apply(self, *, step, states, colors, pending, delivered, request_ids, active,
              ready_probability, feature_bank, fresh, student_action, current, base_action):
        # Only commanded colors and measured sensors cross this boundary.
        n = len(self.windows)
        if any(len(x) != n for x in (states, colors, pending, delivered, request_ids, active)):
            raise ValueError('Retention batch size mismatch')
        if self.mode == 'none':
            return base_action.clone()
        action = base_action.clone()
        for i in range(n):
            window = self.windows[i]
            previous = window.rate
            eligible = (bool(active[i]) and bool(pending[i]) and
                        states[i] in ('execute', 'verify', 'stop') and
                        colors[i] is not None and 0 <= colors[i] < 6)
            request_changed = self.requests[i] != request_ids[i]
            goal_changed = window.goal is not None and window.goal != colors[i]
            if bool(delivered[i]) or request_changed or goal_changed or not eligible:
                self.windows[i] = window = RetentionWindow(horizon=self.horizon)
                self.requests[i] = request_ids[i]
                if previous is not None:
                    self.events.append(dict(case=i, step=step, event='retention_release',
                                            reason='reply' if bool(delivered[i]) else
                                            'new_request' if request_changed else
                                            'new_goal' if goal_changed else 'ineligible'))
                previous = None
            # A new reply must take effect for at least this control step even
            # if a subsequent request was already submitted in the same step.
            if not eligible or bool(delivered[i]):
                continue
            color = int(colors[i])
            ready = ready_probability is not None and bool(ready_probability[i, color] >= self.threshold)
            rate = 0.
            values = None
            if fresh and ready and window.start is None and not window.spent:
                if self.mode == 'learned':
                    if feature_bank is None:
                        raise RuntimeError('Fresh visual evidence lacks retention features')
                    values = self.model(feature_bank[i, color][None])['value'][0]
                    if not torch.isfinite(values).all():
                        raise RuntimeError('Nonfinite retention value prediction')
                    rate = RATES[int(values.argmax())]
            selected = window.update(step=step, pending=True, grounded_command=True,
                                     goal=color, ready=ready, fresh=fresh, selected_rate=rate)
            if selected is not None:
                action[i] = current[i] + selected * (student_action[i] - current[i])
            if previous != selected:
                self.events.append(dict(case=i, step=step, event='retention_choice',
                    request_id=int(request_ids[i]), color=color, rate=selected,
                    value=None if values is None else values.cpu().tolist()))
        return action
