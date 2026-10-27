"""Causal per-goal history with the original cadence and reset semantics."""
import torch

class TemporalStopRuntime:
    """All inputs are measured/model-derived features. No scorer or pose labels."""
    def __init__(self,model):
        self.model=model.eval().requires_grad_(False)
        self.history=None;self.previous_step=None;self.last=None

    @torch.inference_mode()
    def update(self,step,feature_bank):
        if feature_bank.ndim!=3 or feature_bank.shape[1:]!=(6,125):raise ValueError('Expected B x 6 x 125 sensor bank')
        if not torch.isfinite(feature_bank).all():raise ValueError('Nonfinite measured features')
        if step%10:raise ValueError('Frozen readiness cadence is every ten control steps')
        if self.previous_step is not None and step<=self.previous_step:raise ValueError('Time must increase')
        if self.history is not None and self.history.shape[:2]!=feature_bank.shape[:2]:raise ValueError('Batch shape changed')
        if self.previous_step is None or step-self.previous_step!=10:
            self.history=feature_bank[:,:,None,:].repeat(1,1,3,1)
        else:
            self.history=torch.cat((self.history[:,:,1:],feature_bank[:,:,None,:]),2)
        self.previous_step=step
        self.last=self.model(self.history.flatten(-2)).sigmoid()
        if not torch.isfinite(self.last).all():raise RuntimeError('Nonfinite stop prediction')
        return self.last
