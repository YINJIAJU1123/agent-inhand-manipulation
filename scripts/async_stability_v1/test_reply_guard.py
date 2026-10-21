"""Behavioral invariants for the opt-in handoff modes."""
import unittest
import torch
from reply_guard import guard_action

class GuardTest(unittest.TestCase):
 def test_pending_and_unknown_remain_owned_by_original_controller(self):
  current=torch.zeros(4,21);proposal=torch.ones_like(current);controlled=torch.full_like(current,-.2)
  pending=torch.tensor([True,False,True,False]);known=torch.tensor([True,True,False,False])
  for mode in ('direct','ramp','cap'):
   result=guard_action(mode,proposal,current,controlled,pending,known,100,torch.full((4,),100))
   self.assertTrue(torch.equal(result[[0,2,3]],controlled[[0,2,3]]))
 def test_ramp_progresses_without_overshoot_and_ends_at_proposal(self):
  current=torch.full((2,21),-.8);proposal=torch.full_like(current,.8);controlled=current.clone()
  pending=torch.zeros(2,dtype=torch.bool);known=~pending;delivery=torch.tensor([100,100])
  values=[guard_action('ramp',proposal,current,controlled,pending,known,t,delivery) for t in (100,105,115,130)]
  self.assertTrue(all(torch.all((v>=current)&(v<=proposal)) for v in values))
  self.assertTrue(torch.all(values[1]>values[0]));self.assertTrue(torch.allclose(values[-1],proposal))
 def test_cap_bounds_per_joint_jump_and_preserves_direction(self):
  current=torch.linspace(-.9,.9,21)[None];controlled=-current;proposal=controlled.clone()
  result=guard_action('cap',proposal,current,controlled,torch.tensor([False]),torch.tensor([True]),3,torch.tensor([3]))
  self.assertLessEqual(float((result-current).abs().max()),.120001)
  self.assertTrue(torch.all((result-current)*(controlled-current)>=0))
if __name__=='__main__':unittest.main()
