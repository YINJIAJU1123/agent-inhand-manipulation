"""Compatibility fix for zero request ages; original experiment stays frozen."""
import importlib.util
from pathlib import Path
import torch
_source=Path(__file__).resolve().parents[1]/'astra_learning_v1/runtime.py'
_spec=importlib.util.spec_from_file_location('_astra_runtime_frozen_v1',_source)
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

class ControllerRuntime(_module.ControllerRuntime):
    def action(self,step,goal,prop,proposal,current,pending,known,delivered,request_age):
        # An all-idle batch produces tensor([0,...,0]) with int64 dtype in v1.
        # Even an empty indexed assignment requires matching source/dest dtype.
        request_age=torch.as_tensor(request_age,device=self.wait_age.device,dtype=self.wait_age.dtype)
        return super().action(step,goal,prop,proposal,current,pending,known,delivered,request_age)
