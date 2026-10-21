"""Check ordering and gradients of the proposed first-stop training objective."""
import math,unittest
import torch
from sequential import first_stop_loss,score

class FirstStopObjectiveTest(unittest.TestCase):
    def test_correct_first_stop_probability(self):
        shift=math.log(.85/.15)
        z=torch.tensor([math.log(.2/.8)+shift,math.log(.8/.2)+shift],requires_grad=True)
        loss=first_stop_loss(z,torch.tensor([False,True]))
        self.assertAlmostEqual(float(loss),-math.log(.8*.8),places=5)
        loss.backward()
        self.assertGreater(float(z.grad[0]),0.)
        self.assertLess(float(z.grad[1]),0.)
    def test_no_reach_rewards_no_stop(self):
        shift=math.log(.85/.15)
        z=torch.tensor([shift,shift],requires_grad=True)
        loss=first_stop_loss(z,torch.tensor([False,False]))
        self.assertAlmostEqual(float(loss),-math.log(.25),places=5)
        loss.backward();self.assertTrue(bool((z.grad>0).all()))
    def test_frame_accuracy_does_not_hide_first_error(self):
        data={'rows':[dict(seed=1,case=0,step=i,skill=0) for i in range(3)],'error':torch.tensor([[.3],[.1],[.1]])}
        r=score(torch.tensor([[.9],[.9],[.9]]),data,.85)
        self.assertEqual(r['false_first_stops'],1)
        self.assertEqual(r['correct'],0)

if __name__=='__main__':unittest.main()
