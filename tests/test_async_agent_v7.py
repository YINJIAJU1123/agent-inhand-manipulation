import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import unittest,time
spec=importlib.util.spec_from_file_location('async_agent',Path(__file__).resolve().parents[1]/'scripts/agentic/async_agent_v7.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FakeAgent:
    budget=2;goal=None;goal_symbol=None;goal_embedding=None;calls=0
    def __init__(self,release):self.release=release
    def tick(self,rgb,prop,step):
        self.release.wait(2);self.calls+=1;self.goal=3;self.goal_symbol='star';self.next_step=step+20
        return SimpleNamespace(color=3,state='execute',reason='fake')

class AsyncTests(unittest.TestCase):
    def test_physics_advances_and_no_early_goal(self):
        release=Event()
        with ThreadPoolExecutor(1) as pool:
            w=module.AsyncAgent(FakeAgent(release),pool,'astra')
            self.assertTrue(w.submit(None,None,0))
            for step in range(12):
                self.assertFalse(w.poll(step));w.advanced()
                self.assertIsNone(w.snapshot['goal'])
            release.set();w.future.result(timeout=2)
            result=w.future.result();result['finished']=w.request_wall+2.
            self.assertFalse(w.poll(59));self.assertIsNone(w.snapshot['goal'])
            self.assertTrue(w.poll(60));self.assertEqual(w.snapshot['goal'],3)
            self.assertEqual(w.due,80);self.assertEqual(w.events[0]['physics_steps'],12)
    def test_terminal_discards_late_response(self):
        release=Event()
        with ThreadPoolExecutor(1) as pool:
            w=module.AsyncAgent(FakeAgent(release),pool,'astra');w.submit(None,None,0)
            w.close(3);release.set();w.future.result(timeout=2);w.poll(100)
            self.assertIsNone(w.snapshot['goal']);self.assertEqual(w.events[0]['status'],'discarded_terminal')
            self.assertFalse(w.submit(None,None,110))
if __name__=='__main__':unittest.main()
