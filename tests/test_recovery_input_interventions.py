import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/agentic'))
from recovery_input_interventions import controller_inputs,probe_actions,VARIANTS
from recovery_controller_v3 import RecoveryController

class InterventionTests(unittest.TestCase):
 def test_only_selected_fields_change_and_originals_remain_intact(self):
  obs=torch.randn(12,616);goal=torch.randn(12,768);visual=torch.randn(12,400);other=goal.roll(1,0)
  original=obs.clone();original_goal=goal.clone()
  for variant in VARIANTS:
   x,g=controller_inputs(obs,goal,visual,other,variant)
   self.assertTrue(torch.equal(x[:,400:],obs[:,400:]))
   self.assertTrue(torch.equal(x[:,:400],visual if 'frozen_visual' in variant else obs[:,:400]))
   self.assertTrue(torch.equal(g,other if 'shifted_goal' in variant else goal))
  self.assertTrue(torch.equal(obs,original));self.assertTrue(torch.equal(goal,original_goal))
 def test_request_frame_freeze_is_identical_and_probes_do_not_move_model(self):
  for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
   torch.manual_seed(15);model=RecoveryController(616).to(device).eval()
   # Nonzero actor avoids a vacuous zero-initialization equality test.
   with torch.no_grad():model.actor.weight.normal_(0,.01)
   x=torch.randn(12,616,device=device);g=torch.randn(12,768,device=device)
   p=torch.rand(12,21,device=device)*2-1;c=torch.rand_like(p)*2-1
   before={k:v.clone() for k,v in model.state_dict().items()}
   results=probe_actions(model,x,g,x[:,:400].clone(),g.roll(1,0),p,c,torch.zeros(12,device=device),torch.ones(12,dtype=torch.bool,device=device))
   self.assertEqual(results['frozen_visual']['action_l2_from_full'],[0.]*12)
   self.assertGreater(max(results['shifted_goal']['action_l2_from_full']),0)
   for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
 def test_rejects_invalid_intervention(self):
  with self.assertRaises(ValueError):controller_inputs(torch.zeros(2,10),torch.zeros(2,3),torch.zeros(2,4),torch.zeros(2,3),'unknown')

if __name__=='__main__':unittest.main()
