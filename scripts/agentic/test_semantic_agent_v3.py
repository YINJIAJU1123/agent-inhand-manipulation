"""Contract tests. Mocked VLM outputs are not evaluation evidence."""
import ast
import unittest
from pathlib import Path
import numpy as np
from semantic_agent_v3 import SemanticAgent, SemanticVLMClient, SYMBOLS
from semantic_protocol_v3 import make_cases


def response(symbol='circle', color='red', frame='a', direction='up'):
    return {'goal_symbol':symbol,'goal_direction':direction,'image_sha256':frame,
            'observations':[] if color is None else [{'color':color,'symbol':symbol}],
            'raw':'mock','parsed':{},'prompt':'mock'}


class MockClient:
    def __init__(self, responses): self.responses=iter(responses);self.requests=[];self.intent=responses[0] if responses else response()
    def parse_instruction(self, instruction):
        r=dict(self.intent);r['observations']=[];r['image_sha256']=None
        return r
    def ground(self, rgb, instruction, feedback=None, goal_symbol=None):
        self.requests.append((instruction,feedback))
        return next(self.responses)


class Tests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.zeros((32,32,3),dtype=np.uint8)
        self.proprio=np.zeros(8)
    def tick(self, agent, step, rgb=None):
        return agent.tick(self.rgb if rgb is None else rgb,self.proprio,step)
    def test_full_instruction_enters_actual_prompt(self):
        class Capture(SemanticVLMClient):
            def ask(self,rgb,prompt,max_tokens):
                self.sent=prompt
                return {'parsed':{'goal_symbol':'circle','goal_direction':'up','stickers':[{'color':'blue','symbol':'circle'}]},'raw':'mock','image_sha256':'x'}
        c=Capture();instruction='Bring the round emblem onto the top.'
        result=c.ground(self.rgb,instruction,goal_symbol='circle')
        self.assertIn(instruction,c.sent)
        self.assertEqual(result['goal_symbol'],'circle')
        self.assertEqual(result['observations'],[{'color':'blue','symbol':'circle'}])
        self.assertEqual(result['prompt'],c.sent)
    def test_same_inventory_different_language_goal_selects_different_skill(self):
        inventory=[{'color':'red','symbol':'star'},{'color':'blue','symbol':'circle'}]
        goals=[]
        for symbol in ('star','circle'):
            r=response(symbol);r['observations']=inventory
            a=SemanticAgent('feedback',MockClient([r]),f'{symbol} up')
            goals.append(self.tick(a,0).color)
        self.assertEqual(goals,[0,2])
    def test_unknown_and_unsupported_goals_cannot_drive_a_skill(self):
        for symbol,direction in [('unknown','up'),('star','unknown')]:
            a=SemanticAgent('feedback',MockClient([response(symbol,direction=direction)]),'unsupported')
            self.assertIsNone(self.tick(a,0).color)
            self.assertIsNone(a.goal)
    def test_no_unique_visual_evidence_does_not_bind_a_target(self):
        r=response();r['observations'].append({'color':'blue','symbol':'circle'})
        a=SemanticAgent('feedback',MockClient([r]),'circle up')
        self.assertEqual(self.tick(a,0).state,'inspect')
        self.assertIsNone(a.goal)
    def test_unknown_retains_goal_and_call_budget_is_hard(self):
        a=SemanticAgent('periodic',MockClient([response(),response(color=None,frame='b')]),'circle up',budget=3)
        for step in range(0,600,10):self.tick(a,step)
        self.assertEqual(a.calls,3);self.assertEqual(a.goal,0)
    def test_conflicting_semantic_parse_does_not_reinterpret_instruction(self):
        a=SemanticAgent('periodic',MockClient([response(),response('star','blue','b')]),'circle up')
        self.tick(a,0);self.tick(a,90)
        self.assertEqual(a.goal_symbol,'circle');self.assertEqual(a.goal,0)
        self.assertEqual(a.trace[-1]['binding'],'instruction_parse_conflict')
    def test_observed_execution_stall_triggers_bounded_reobservation(self):
        a=SemanticAgent('feedback',MockClient([response(),response(color=None,frame='b')]),'circle up',budget=3)
        for step in range(0,100,10):d=self.tick(a,step)
        self.assertEqual(a.trace[-1]['reason'],'execution_stall_reobserve')
        self.assertEqual(d.state,'recover_observe')
        self.assertEqual(self.tick(a,110).state,'execute')
        self.assertEqual(a.calls,3)
        self.assertTrue(a.recovery_used)
    def test_inspection_feedback_reacts_to_a_new_view(self):
        a=SemanticAgent('feedback',MockClient([response(color=None),response(frame='b')]),'circle up')
        self.tick(a,0)
        for step in range(10,80,10):self.tick(a,step,np.full_like(self.rgb,128))
        self.assertEqual(a.trace[-1]['reason'],'new_observation_after_motion')
        self.assertEqual(a.goal,0)
    def test_counterfactual_instructions_preserve_scene_and_change_targets(self):
        first=make_cases(31001,0);second=make_cases(31001,1)
        for x,y in zip(first,second):
            self.assertEqual(x['layout'],y['layout'])
            self.assertNotEqual(x['target_color_index'],y['target_color_index'])
            self.assertNotEqual(x['instruction'],y['instruction'])
        self.assertEqual({x['target_symbol'] for x in first},set(SYMBOLS))
    def test_agent_imports_no_simulator_or_scorer(self):
        source=(Path(__file__).parent/'semantic_agent_v3.py').read_text()
        for forbidden in ['target_face','goal_rot','object_rot','semantic_metrics','isaaclab','semantic_protocol_v3']:
            self.assertNotIn(forbidden,source)
    def test_non_finite_proprioception_is_rejected(self):
        a=SemanticAgent('feedback',MockClient([]),'circle up')
        with self.assertRaises(ValueError):a.tick(self.rgb,np.array([np.nan]),0)

if __name__=='__main__':unittest.main()
