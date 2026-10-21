import unittest,copy
from unittest.mock import patch
from concurrent.futures import Future
import numpy as np
from async_control import AsyncAgent
from audit import ResponseAudit,state_drift
class Pool:
    def submit(self,fn):self.f=Future();return self.f
class Agent:
    budget=6;next_step=60
class Tests(unittest.TestCase):
    def test_delay_after_known_skill_only_and_no_early_delivery(self):
        pool=Pool();agent=Agent();w=AsyncAgent(agent,pool,'astra',extra_delay_s=6)
        with patch('async_control.time.monotonic',return_value=100.):self.assertTrue(w.submit(None,None,0))
        self.assertEqual(w.event['injected_delay_s'],0.)
        new=copy.deepcopy(w.snapshot);new['decision']['color']=0
        pool.f.set_result(dict(snapshot=new,finished=102.,error=None))
        self.assertFalse(w.poll(59));self.assertTrue(w.poll(60))
        self.assertEqual(w.due,120)
        with patch('async_control.time.monotonic',return_value=100.):self.assertTrue(w.submit(None,None,120))
        self.assertEqual(w.event['injected_delay_s'],6.)
        agent.next_step=180;pool.f.set_result(dict(snapshot=new,finished=102.,error=None))
        self.assertFalse(w.poll(359));self.assertTrue(w.poll(360));self.assertEqual(w.due,420)
    def test_late_reply_cannot_mutate_closed_episode(self):
        pool=Pool();w=AsyncAgent(Agent(),pool,'astra',extra_delay_s=6);w.snapshot['decision']['color']=0
        with patch('async_control.time.monotonic',return_value=100.):w.submit(None,None,0)
        before=copy.deepcopy(w.snapshot);new=copy.deepcopy(before);new['decision']['color']=2
        pool.f.set_result(dict(snapshot=new,finished=102.,error=None));w.close(1)
        self.assertEqual(w.snapshot,before);self.assertEqual(w.events[0]['status'],'discarded_terminal')
    def test_quaternion_sign_invariance(self):
        p=dict(object_root=[0,0,0,1,0,0,0],joint_pos=[0,0]);q=copy.deepcopy(p);q['object_root'][3]=-1
        self.assertEqual(state_drift(p,q)['object_rotation_rad'],0.)
        q['object_root'][3:]=[2**-.5,0,0,2**-.5]
        self.assertAlmostEqual(state_drift(p,q)['object_rotation_rad'],np.pi/2)
    def test_audit_followup_and_no_input_mutation(self):
        a=ResponseAudit();s=dict(visual=[0],proprio=[0],command=[0]);p=dict(object_root=[0,0,0,1,0,0,0],joint_pos=[0]);snap=dict(decision=dict(color=0));image=np.zeros((2,2,3),dtype=np.uint8)
        a.request(0,1,10,s,p,snap,image,10);a.response(0,40,s,p,snap,image,40,dict(injected_delay_s=0))
        a.physical_step(0,41,True,False);r=a.finish([dict(case=0,success=False,drop=True)])[0]
        self.assertTrue(r['drop_within_2s']);self.assertFalse(r['followup_censored']);self.assertEqual(r['request_image_age_s'],1.)
        self.assertEqual(s,dict(visual=[0],proprio=[0],command=[0]))
if __name__=='__main__':unittest.main()
