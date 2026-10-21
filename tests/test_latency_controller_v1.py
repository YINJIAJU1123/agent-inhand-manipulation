"""Action semantics and trainability checks, without Isaac Sim."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/agentic'))
from latency_controller_v1 import command, Supervisor, ppo_update

class LatencyControllerTests(unittest.TestCase):
    def test_position_target_interpolation_and_hold_entry(self):
        current=torch.tensor([[.8,-.6]]).repeat(3,1)
        student=torch.tensor([[-.2,.4]]).repeat(3,1)
        anchor=torch.zeros_like(current)
        mode=torch.tensor([0,1,2]); previous=torch.tensor([0,0,0])
        action,latched=command(student,current,anchor,mode,previous)
        torch.testing.assert_close(action[0],student[0])
        torch.testing.assert_close(action[1],current[1]+.15*(student[1]-current[1]))
        torch.testing.assert_close(action[2],current[2])
        action2,_=command(student,current*.3,latched,mode,mode)
        torch.testing.assert_close(action2[2],current[2])

    def test_ppo_changes_actor_without_privileged_actor_input(self):
        torch.manual_seed(4)
        model=Supervisor(9); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        before=[v.detach().clone() for v in model.actor.parameters()]
        rollout=[]
        for step in range(4):
            obs=torch.randn(8,9); priv=torch.randn(8,8)
            with torch.no_grad():
                dist=model.distribution(obs); act=dist.sample()
                rollout.append(dict(obs=obs,priv=priv,action=act,logprob=dist.log_prob(act),
                    value=model.value(obs,priv),reward=(act==1).float(),
                    done=torch.full((8,),step==3),valid=torch.ones(8,dtype=torch.bool)))
        ppo_update(model,optimizer,rollout)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(before,model.actor.parameters())))
        self.assertEqual(model.actor[0].in_features,9)

if __name__=='__main__': unittest.main()
