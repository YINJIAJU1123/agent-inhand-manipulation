"""Verify independent factors, unchanged sample count and weighted PPO guard."""
import copy,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/agentic'),str(ROOT/'scripts/recovery_factorial')]
from design import ARMS,branch_for,merge_draws,weights_for
from recovery_controller_v3 import RecoveryController,make_optimizers,update as baseline_update
from weighted_ppo import update,weighted_mean


def rollout(model):
 rows=[]
 for t in range(4):
  with torch.no_grad():
   obs=torch.randn(12,20);g=torch.randn(12,9);z=torch.randn(12,8);d=model.distribution(obs,g);a=d.sample()
   rows.append(dict(obs=obs,goal=g,priv=z,action=a,mean=d.mean,std=d.base_dist.scale,logprob=d.log_prob(a),value=model.value(obs,g,z),
    reward=torch.arange(12).float()*(t==3),done=torch.full((12,),t==3),valid=torch.ones(12,dtype=torch.bool)))
 return rows

class FactorialTests(unittest.TestCase):
 def test_independent_factors_and_draw_identity(self):
  for arm,factors in ARMS.items():
   branches=[branch_for(arm,i) for i in range(6)]
   self.assertEqual(branches,list(range(6)) if factors['pairing'] else [0]*6)
   self.assertEqual(weights_for(arm,[1,2]*6),[1,2]*6 if factors['weighting'] else [1]*12)
   packs=[dict(seed=1,delay=6.,checkpoint_sha256='same',request_cache_sha256='same',training_arm=arm,draw_id=i,branch=branches[i],noise_seed=100+i,
    success=torch.ones(12,dtype=torch.bool),rollout=[dict(obs=torch.zeros(12,3),valid=torch.ones(12,dtype=torch.bool))]) for i in range(6)]
   merged,_=merge_draws(packs,arm);self.assertEqual(len(merged[0]['obs']),72)
   packs[0]['branch']=5 if not factors['pairing'] else 1
   with self.assertRaises(ValueError):merge_draws(packs,arm)
 def test_uniform_weights_are_exact_baseline_update(self):
  torch.set_num_threads(2);torch.manual_seed(7);m=RecoveryController(20,9,3);other=copy.deepcopy(m);data=rollout(m);success=torch.arange(12)%2==0
  torch.manual_seed(8);a=baseline_update(m,make_optimizers(m),data,success)
  torch.manual_seed(8);b=update(other,make_optimizers(other),data,success,[1]*12)
  for x,y in zip(m.parameters(),other.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
  self.assertEqual(a['samples'],b['unique_samples']);self.assertEqual(a['accepted_policy_steps'],b['accepted_policy_steps'])
 def test_weighting_does_not_duplicate_decisions_and_kl_rolls_back(self):
  torch.manual_seed(9);m=RecoveryController(20,9,3);data=rollout(m);opts=make_optimizers(m);opts['policy'].param_groups[0]['lr']=1.
  before=[p.detach().clone() for p in m.parameters_for('policy')]
  result=update(m,opts,data,torch.arange(12)%2==0,[2,1]*6,target_kl=.001)
  self.assertEqual(result['samples'],48);self.assertLess(result['effective_samples'],48)
  self.assertTrue(result['kl_rollback']);self.assertEqual(result['accepted_policy_steps'],0)
  self.assertLessEqual(result['final_exact_kl'],.001)
  for a,b in zip(before,m.parameters_for('policy')):torch.testing.assert_close(a,b,atol=0,rtol=0)
 def test_weighted_objective_matches_integer_replication_without_data_expansion(self):
  values=torch.tensor([2.,-3.,5.],requires_grad=True);w=torch.tensor([2.,1.,2.])
  explicit=weighted_mean(values,w);expanded=values[torch.tensor([0,0,1,2,2])].mean()
  torch.testing.assert_close(explicit,expanded)
  torch.testing.assert_close(torch.autograd.grad(explicit,values,retain_graph=True)[0],torch.autograd.grad(expanded,values)[0])

if __name__=='__main__':unittest.main()
