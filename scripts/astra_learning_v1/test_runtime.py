"""Checks at the asynchronous response boundary, without a simulator or API."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/agentic'))
import torch
from model import LearnedController,load_controller
from runtime import ControllerRuntime
from recovery_controller_v2 import observation
from recovery_controller_v3 import residual_command

class Probe:
    def __init__(self):self.seen=[]
    def raw_action(self,x,g):
        self.seen.append((x.clone(),g.clone()))
        return torch.ones(len(x),22)*g[:,0,None]
    def diagnostic(self,x,g):return torch.ones(len(x),3)/3

class RuntimeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_zero_correction_and_safe_checkpoint(self):
        path=ROOT/'outputs/astra_learning_v1/train_20261019/controller_final.pt'
        trained,pack=load_controller(path,'cpu')
        m=LearnedController(pack['canonical']);x=torch.randn(6,616);g=pack['canonical']
        self.assertTrue(torch.equal(m.raw_action(x,g),m.reference.distribution(x,g).mean))
        self.assertTrue(torch.isfinite(trained.raw_action(x,g)).all())
        self.assertFalse(any(p.requires_grad for p in trained.parameters()))
        delta=trained.raw_action(x,g)-trained.reference.distribution(x,g).mean
        self.assertLessEqual(float(delta.abs().max()),.100001)
    def test_delivery_on_odd_step_and_handoff(self):
        p=Probe();r=ControllerRuntime(p,torch.zeros(1,21),True);r.update_visual(torch.zeros(1,100))
        prop=torch.zeros(1,128);proposal=torch.ones(1,21)*.2;current=torch.zeros(1,21);g=torch.ones(1,768)*.2
        r.action(60,g,prop,proposal,current,torch.tensor([True]),torch.tensor([True]),[False],torch.tensor([2.]))
        g2=g*2
        a,d=r.action(61,g2,prop,proposal,current,torch.tensor([False]),torch.tensor([True]),[True],torch.tensor([61/30]))
        self.assertTrue(d['raw_refreshed'][0]);self.assertEqual(len(p.seen),2)
        self.assertTrue(torch.equal(p.seen[-1][1],g2))
        self.assertAlmostEqual(float(p.seen[-1][0][0,-4]),61/30/20,places=6)
        self.assertEqual(float(p.seen[-1][0][0,-3]),0.)
        a,d=r.action(121,g2,prop,proposal,current,torch.tensor([False]),torch.tensor([True]),[False],torch.zeros(1))
        self.assertTrue(torch.equal(a,proposal));self.assertFalse(d['controller_active'][0])
    def test_unknown_goal_not_used_for_control(self):
        p=Probe();r=ControllerRuntime(p,torch.zeros(1,21),True);r.update_visual(torch.zeros(1,100))
        a,d=r.action(0,torch.zeros(1,768),torch.zeros(1,128),torch.zeros(1,21),torch.zeros(1,21),torch.tensor([True]),torch.tensor([False]),[False],torch.zeros(1))
        self.assertEqual(len(p.seen),0);self.assertFalse(d['controller_active'][0])
    def test_observation_matches_training_during_recovery(self):
        p=Probe();r=ControllerRuntime(p,torch.zeros(1,21),True)
        compact=torch.randn(1,100);r.update_visual(compact);proprio=torch.randn(1,128)
        proposal=torch.zeros(1,21);cur=torch.ones(1,21)*.1;g=torch.ones(1,768)*.1
        r.action(0,g,proprio,proposal,cur,torch.tensor([True]),torch.tensor([True]),[False],torch.tensor([0.]))
        r.commit(cur)
        r.action(180,g,proprio,proposal,cur,torch.tensor([False]),torch.tensor([True]),[True],torch.tensor([6.]))
        r.action(182,g,proprio,proposal,cur,torch.tensor([False]),torch.tensor([True]),[False],torch.tensor([0.]))
        expected=observation(r.history,proprio,proposal,cur,r.anchor,r.previous,torch.tensor([6.]),torch.tensor([False]),torch.tensor([2/30]))
        self.assertTrue(torch.equal(p.seen[-1][0],expected))
if __name__=='__main__':unittest.main()
