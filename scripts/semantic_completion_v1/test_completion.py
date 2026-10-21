"""Regression checks for evaluator/agent separation and autonomous stopping."""
import sys
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'scripts/agentic')]
from scoring import Score,service_anomaly
from agent import EvidenceRuntime,language_symbol,reject_stop,override_inspection
from types import SimpleNamespace


class CompletionTests(unittest.TestCase):
    def test_reach_does_not_end_episode(self):
        s=Score()
        self.assertFalse(s.update(10,True,.1,False))
        self.assertTrue(s.reached);self.assertFalse(s.success)
        self.assertTrue(s.update(30,True,.3,False,stop=True))
        self.assertTrue(s.false_stop);self.assertFalse(s.success)
    def test_current_state_and_binding_required(self):
        for grounded,error,dropped in [(False,.01,False),(True,.17,False),(True,.01,True)]:
            s=Score();self.assertTrue(s.update(20,grounded,error,dropped,stop=True));self.assertFalse(s.success)
        s=Score();self.assertTrue(s.update(20,True,.159,False,stop=True));self.assertTrue(s.success)
    def test_timeout_after_reach_is_not_success(self):
        s=Score();s.update(1,True,.02,False)
        row=s.finish(3600);self.assertTrue(row['reached']);self.assertFalse(row['success']);self.assertEqual(row['reason'],'horizon')
    def test_service_error_is_separate(self):
        self.assertFalse(service_anomaly([{'status':'delivered','wall_latency_s':8.}]))
        self.assertTrue(service_anomaly([{'status':'error','wall_latency_s':2.}]))
        self.assertTrue(service_anomaly([{'status':'discarded_terminal','wall_latency_s':131.}]))
    def test_text_parser_does_not_use_case_or_layout(self):
        self.assertEqual(language_symbol('Bring the sticker with five pointed tips up.'),0)
        self.assertEqual(language_symbol('Put the rhombus upward.'),4)
        self.assertIsNone(language_symbol('Turn a sticker.'))
    def test_verifier_requires_both_binding_and_fresh_completion(self):
        r=EvidenceRuntime.__new__(EvidenceRuntime)
        r.bank=torch.zeros(1,6,6);r.bank[0,2,0]=.9
        r.last={'ready':torch.full((1,6),-10.)};r.ready_threshold=.9;r.symbol_threshold=.7
        self.assertFalse(r.can_stop(0,2,'star','star up'))
        r.last['ready'][0,2]=10.
        self.assertTrue(r.can_stop(0,2,'star','star up'))
        self.assertFalse(r.can_stop(0,2,'circle','star up'))
        self.assertFalse(r.can_stop(0,1,'star','star up'))
    def test_rejection_does_not_rewind_calls_or_lose_history(self):
        agent=SimpleNamespace(calls=4,budget=6,declared_success=True,response_id='r',call_id='c')
        w=SimpleNamespace(future=None,snapshot={'decision':{'color':2,'state':'stop'}},agent=agent,submitted=4)
        reject_stop(w,100)
        self.assertEqual(w.submitted,4);self.assertEqual(agent.calls,4);self.assertEqual(agent.call_id,'c')
        self.assertFalse(agent.declared_success);self.assertEqual(w.snapshot['decision']['state'],'execute')
        override_inspection(w,1,140);self.assertEqual(w.due,230);self.assertEqual(w.snapshot['decision']['color'],1)


if __name__=='__main__':unittest.main()
