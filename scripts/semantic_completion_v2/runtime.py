"""Read-only learned assistance. Astra owns the semantic goal hypothesis."""
import torch
from assets import STUDENT,ADAPTER,sha,COLORS
from features import FrozenFeatures
from completion_model import load_model

class CompletionRuntime:
    def __init__(self,path,device):
        self.encoder=FrozenFeatures(device);self.model,self.pack=load_model(path,device)
        m=self.pack['feature_manifest']
        if sha(STUDENT)!=m['student_sha256'] or sha(ADAPTER)!=m['adapter_sha256']:raise RuntimeError('Frozen policy or language adapter mismatch')
        self.threshold=self.pack['threshold'];self.last=None
    @torch.inference_mode()
    def update(self,features,proprio):
        y=self.model(self.encoder.candidates(features,proprio))
        self.last=dict(ready=y['ordinal'][...,2].sigmoid(),progress=y['progress'].sigmoid(),estimated_error=y['log_error'].exp())
        return self.last
    def can_stop(self,case,color):
        if color is None or not self.pack.get('calibration') or self.last is None:return False
        return bool(self.last['ready'][case,color]>=self.threshold)
    def feedback(self,case,color):
        if color is None or self.last is None:return {}
        feedback=dict(local_completion_estimate=float(self.last['ready'][case,color]),
            queried_goal_color=COLORS[color],
            scope='Conditional on your semantic binding. Learned estimates from measured sensors; no ground-truth success sensor.')
        if self.pack.get('progress_enabled',False):
            feedback['local_progress_estimate']=float(self.last['progress'][case,color])
        return feedback
