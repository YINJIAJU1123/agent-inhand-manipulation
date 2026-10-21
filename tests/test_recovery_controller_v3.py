"""Meaningful invariants for optimization isolation and baseline preservation."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/agentic'))
from recovery_controller_v3 import RecoveryController,make_optimizers,residual_command,update,OutcomeReplay

class RecoveryV3Tests(unittest.TestCase):
 def test_zero_mean_matches_strong_fixed_baseline_and_bounds(self):
  devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
  for device in devices:
   with self.subTest(device=device):
    torch.manual_seed(7);p=torch.rand(128,21,device=device)*2-1;c=torch.rand_like(p)*2-1
    pending=torch.arange(128,device=device)%2==0;frac=torch.linspace(0,1,128,device=device)
    action,_,_=residual_command(p,c,torch.zeros(128,22,device=device),frac,pending)
    self.assertTrue(torch.equal(action[pending],(c+.15*(p-c))[pending]))
    self.assertTrue(torch.equal(action[~pending],p[~pending]))
    for sign in [-1,1]:
     action,_,_=residual_command(p,c,torch.full((128,22),sign*1000.,device=device),frac,pending)
     self.assertTrue(torch.isfinite(action).all() and action.abs().max()<=1)
     self.assertTrue(torch.equal(action[-1],p[-1]))
 def test_critic_and_outcome_updates_cannot_move_policy(self):
  torch.manual_seed(3);m=RecoveryController(20,9,3);opts=make_optimizers(m)
  x=torch.randn(8,20);g=torch.randn(8,9);priv=torch.randn(8,8)
  old={name:p.detach().clone() for name,p in m.named_parameters() if id(p) in {id(v) for v in m.parameters_for('policy')}}
  for branch,loss in [('value',m.value(x,g,priv).square().mean()),('outcome',m.outcome_logit(x,g).square().mean())]:
   opts[branch].zero_grad();loss.backward();opts[branch].step()
  for name,p in m.named_parameters():
   if name in old:torch.testing.assert_close(p,old[name],rtol=0,atol=0)
 def test_kl_guard_rolls_back_excessive_policy_update(self):
  torch.manual_seed(9);m=RecoveryController(20,9,3);opts=make_optimizers(m)
  opts['policy'].param_groups[0]['lr']=1.
  old=[p.detach().clone() for p in m.parameters_for('policy')];roll=[]
  for t in range(4):
   with torch.no_grad():
    x=torch.randn(12,20);g=torch.randn(12,9);priv=torch.randn(12,8);d=m.distribution(x,g);a=d.sample()
    roll.append(dict(obs=x,goal=g,priv=priv,action=a,logprob=d.log_prob(a),mean=d.mean,std=d.base_dist.scale,
      value=m.value(x,g,priv),reward=torch.arange(12).float()*(t==3),valid=torch.ones(12,dtype=torch.bool),done=torch.full((12,),t==3)))
  result=update(m,opts,roll,torch.arange(12)%2==0,target_kl=.001)
  self.assertTrue(result['kl_rollback']);self.assertEqual(result['accepted_policy_steps'],0)
  self.assertLessEqual(result['final_exact_kl'],.001)
  for a,b in zip(old,m.parameters_for('policy')):torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test_outcome_split_is_by_whole_seed(self):
  replay=OutcomeReplay();sample=[dict(valid=torch.ones(2,dtype=torch.bool),obs=torch.randn(2,20),goal=torch.randn(2,9)) for _ in range(16)]
  replay.add(sample,torch.tensor([1,0]),920000);replay.add(sample,torch.tensor([0,1]),920001)
  self.assertEqual({c['seed'] for c in replay.validation},{920000})
  self.assertEqual({c['seed'] for c in replay.train},{920001})

if __name__=='__main__':unittest.main()
