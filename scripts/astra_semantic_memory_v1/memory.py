"""Retain measured high-confidence bindings for a single, unchanged object."""
import torch


class SemanticMemory:
    def __init__(self, cases, threshold, device='cpu'):
        if not 0 < threshold <= 1:
            raise ValueError('A calibrated acceptance threshold is required')
        self.threshold = float(threshold)
        self.symbol = torch.full((cases, 6), -1, dtype=torch.long, device=device)
        self.confidence = torch.zeros(cases, 6, device=device)
        self.observed_step = torch.full_like(self.symbol, -1)
        self.last_step = -1

    @torch.inference_mode()
    def update(self, probability, step):
        if probability.shape != (*self.symbol.shape, 7) or not torch.isfinite(probability).all():
            raise ValueError('Expected finite per-color symbol/unknown probabilities')
        if step <= self.last_step:
            raise ValueError('Frames must arrive in strictly increasing order')
        if (probability < 0).any() or (probability > 1).any() or not torch.allclose(probability.sum(-1), torch.ones_like(self.confidence), atol=1e-5):
            raise ValueError('Invalid probability distribution')
        confidence, symbol = probability[..., :6].max(-1)
        accepted = confidence >= self.threshold
        self.symbol[accepted] = symbol[accepted]
        self.confidence[accepted] = confidence[accepted]
        self.observed_step[accepted] = int(step)
        self.last_step = int(step)
        return dict(symbol=self.symbol.clone(), confidence=self.confidence.clone(),
                    observed_step=self.observed_step.clone(), fresh_accepted=accepted.clone())

    def candidates(self, case, symbol):
        if not 0 <= symbol < 6:
            raise ValueError('Unknown public query symbol')
        return (self.symbol[case] == symbol).nonzero().flatten().tolist()
