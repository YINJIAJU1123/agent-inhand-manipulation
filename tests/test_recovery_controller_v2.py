"""Check credit assignment, action boundaries and trainability without Isaac."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/agentic'))
from recovery_controller_v2 import RecoveryController, residual_command, fixed_command, advantages, update

class RecoveryTests(unittest.TestCase):
    def test_initial_policy_matches_slow_smooth_and_final_student(self):
        torch.manual_seed(1)
        proposed=torch.rand(6,21)*2-1;current=torch.rand(6,21)*2-1
        fraction=torch.linspace(0,1,6);raw=torch.zeros(6,22)
        action,_,_=residual_command(proposed,current,raw,fraction)
        expected=fixed_command('slow_smooth',proposed,current,current,torch.zeros(6,dtype=torch.bool),fraction)
        torch.testing.assert_close(action,expected)
        huge=torch.full((6,22),1000.)
        end,_,_=residual_command(proposed,current,huge,torch.ones(6))
        torch.testing.assert_close(end,proposed)
        action,_,_=residual_command(proposed,current,huge,torch.zeros(6))
        self.assertTrue(torch.isfinite(action).all() and action.abs().max()<=1.)

    def test_hold_and_rejoin_semantics(self):
        prop=torch.ones(2,21);cur=torch.zeros(2,21);anchor=torch.full((2,21),-.2)
        pending=torch.tensor([True,False]);fraction=torch.zeros(2)
        smooth=fixed_command('hold_smooth',prop,cur,anchor,pending,fraction)
        torch.testing.assert_close(smooth[0],anchor[0]);torch.testing.assert_close(smooth[1],cur[1])
        instant=fixed_command('hold',prop,cur,anchor,pending,fraction)
        torch.testing.assert_close(instant[1],prop[1])

    def test_outcome_credit_reaches_wait_but_not_dead_samples(self):
        reward=torch.tensor([[0.,-8.],[0.,0.],[12.,999.]])
        valid=torch.tensor([[True,True],[True,False],[True,False]])
        done=torch.tensor([[False,True],[False,True],[True,True]])
        result=advantages(reward,torch.zeros_like(reward),done,valid,gamma=1.,lam=1.)
        torch.testing.assert_close(result,torch.tensor([[12.,-8.],[12.,0.],[12.,0.]]))

    def test_actor_is_privileged_free_and_goal_sensitive_after_update(self):
        torch.manual_seed(8);model=RecoveryController(20,goal_dim=9,action_dim=3)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.randn(8,20);goal=torch.randn(8,9)
        roll=[];before=model.actor.weight.detach().clone()
        for step in range(4):
            with torch.no_grad():
                priv=torch.randn(8,8);dist=model.distribution(x,goal);act=dist.sample()
                roll.append(dict(obs=x,goal=goal,priv=priv,action=act,logprob=dist.log_prob(act),
                    value=model.value(x,goal,priv),reward=torch.arange(8).float()*(step==3),
                    done=torch.full((8,),step==3),valid=torch.ones(8,dtype=torch.bool)))
        stats=update(model,opt,roll,torch.arange(8)%2==0)
        self.assertTrue(torch.isfinite(torch.tensor(list(stats.values()))).all())
        self.assertFalse(torch.equal(before,model.actor.weight))
        with torch.no_grad():
            first=model.distribution(x,goal).mean
            _=model.value(x,goal,torch.full((8,8),999.))
            torch.testing.assert_close(first,model.distribution(x,goal).mean)
            self.assertFalse(torch.equal(first,model.distribution(x,-goal).mean))

if __name__=='__main__':unittest.main()
