"""Target-conditioning and runtime sensor-contract checks."""
import unittest,torch
from pathlib import Path
from features import FrozenFeatures
from runtime import CompletionRuntime
class ContractTests(unittest.TestCase):
    def test_candidate_batch_matches_explicit_goal_queries(self):
        torch.set_num_threads(2);enc=FrozenFeatures('cpu')
        path=Path('outputs/semantic_completion_v1/corrected_repair_20261021/base/data/train_96111/data.pt')
        d=torch.load(path,weights_only=True,map_location='cpu');f=d['features'][:2];p=d['proprio'][:2]
        all_x=enc.candidates(f,p)
        single=torch.stack([enc(f,p,torch.full((2,),i,dtype=torch.long)) for i in range(6)],1)
        self.assertTrue(torch.allclose(all_x,single,atol=2e-5,rtol=1e-5))
        self.assertGreater(float((all_x[:,0]-all_x[:,1]).abs().mean()),.01)
        self.assertTrue(all(not p.requires_grad for p in enc.parameters()))
    def test_uncalibrated_gate_never_claims_completion(self):
        gate=CompletionRuntime.__new__(CompletionRuntime);gate.pack={'calibration':None};gate.threshold=1.
        gate.last={'ready':torch.ones(1,6)}
        self.assertFalse(gate.can_stop(0,0))
    def test_unvalidated_progress_is_not_agent_feedback(self):
        gate=CompletionRuntime.__new__(CompletionRuntime);gate.pack={}
        gate.last={'ready':torch.ones(1,6),'progress':torch.ones(1,6)}
        self.assertNotIn('local_progress_estimate',gate.feedback(0,0))
    def test_gate_uses_only_queried_goal(self):
        gate=CompletionRuntime.__new__(CompletionRuntime);gate.pack={'calibration':{'accepted':20}};gate.threshold=.9
        gate.last={'ready':torch.tensor([[.95,.1,.1,.1,.1,.1]])}
        self.assertTrue(gate.can_stop(0,0));self.assertFalse(gate.can_stop(0,1));self.assertFalse(gate.can_stop(0,None))
if __name__=='__main__':unittest.main()
