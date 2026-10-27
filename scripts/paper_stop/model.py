"""Bounded and unbounded residuals, retaining the original base logit."""
import torch
import rich as old

class StopCorrection(old.RichStop):
    def __init__(self, bounded):
        super().__init__()
        self.bounded=bool(bounded)
    def forward(self,features):
        if features.shape[-1]!=old.FRAME_DIM*old.WINDOW:
            raise ValueError('Invalid temporal sensor input')
        base=features[..., (old.WINDOW-1)*old.FRAME_DIM+old.BASE_LOGIT]
        delta=self.net(((features-self.mean)/self.scale).clamp(-8,8)).squeeze(-1)
        return base+(4*torch.tanh(delta/4) if self.bounded else delta)
