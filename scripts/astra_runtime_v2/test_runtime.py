"""Regression: the whole batch has no pending request and integer zero ages."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agentic'))
import torch
from runtime import ControllerRuntime,_module
from recovery_controller_v3 import RecoveryController

class RuntimeRegression(unittest.TestCase):
    def test_idle_integer_ages_equal_float_reference(self):
        torch.manual_seed(11)
        model=RecoveryController(616).eval().requires_grad_(False)
        fixed=ControllerRuntime(model,torch.zeros(6,21),False)
        original=_module.ControllerRuntime(model,torch.zeros(6,21),False)
        args=(101,torch.randn(6,768),torch.randn(6,128),torch.rand(6,21)*.2,torch.zeros(6,21),torch.zeros(6,dtype=torch.bool),torch.ones(6,dtype=torch.bool),[False]*6)
        with torch.inference_mode():
            with self.assertRaises(RuntimeError):original.action(*args,torch.zeros(6,dtype=torch.long))
            expected,diag=original.action(*args,torch.zeros(6))
            actual,newdiag=fixed.action(*args,torch.zeros(6,dtype=torch.long))
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(newdiag,diag)
    def test_delivered_integer_age_is_accepted(self):
        model=RecoveryController(616).eval().requires_grad_(False)
        fixed=ControllerRuntime(model,torch.zeros(1,21),False)
        with torch.inference_mode():
            action,diag=fixed.action(180,torch.randn(1,768),torch.zeros(1,128),torch.ones(1,21)*.2,torch.zeros(1,21),torch.tensor([False]),torch.tensor([True]),[True],torch.tensor([6]))
        self.assertEqual(fixed.wait_age.item(),6.);self.assertTrue(diag['raw_refreshed'][0]);self.assertTrue(torch.isfinite(action).all())
if __name__=='__main__':unittest.main()
