"""Controller boundary tests; mocked responses are NOT experimental results."""
import ast,sys,unittest
from pathlib import Path
import numpy as np
from visual_agent import VisualAgent

class Responses:
    def __init__(self,items): self.items=iter(items)
    def ground(self,rgb,instruction):
        color,frame=next(self.items)
        return {'color':color,'visible':color!='unknown','image_sha256':frame}

    def verify(self,rgb,color):
        return {'color':color,'visible':True,'image_sha256':'verify','parsed':{'upward':'no'}}

class AgentTests(unittest.TestCase):
    def setUp(self): self.rgb=np.zeros((32,32,3),dtype=np.uint8)
    def tick(self,agent,step): return agent.tick(self.rgb,step)
    def test_unknown_retains_accepted_evidence(self):
        agent=VisualAgent('periodic',Responses([('red','a'),('unknown','b')]))
        self.tick(agent,0); self.tick(agent,90)
        self.assertEqual(agent.goal,0)
    def test_initial_binding_and_conflict_require_distinct_observations(self):
        agent=VisualAgent('event_memory',Responses([('red','a'),('red','b'),('blue','c'),('blue','c'),('blue','d')]))
        self.assertEqual(self.tick(agent,0).state,'inspect')
        self.assertIsNone(agent.goal)
        self.tick(agent,60); self.assertEqual(agent.goal,0)
        agent.last_mass=np.ones(6); self.tick(agent,120)
        self.assertEqual(agent.goal,0)
        self.tick(agent,180); self.assertEqual(agent.goal,0)
        self.tick(agent,240); self.assertEqual(agent.goal,2)
    def test_call_budget_is_hard_limit(self):
        agent=VisualAgent('periodic',Responses([('unknown','a')]),budget=1)
        for step in range(0,1000,10): self.tick(agent,step)
        self.assertEqual(agent.calls,1)
    def test_once_abstains_on_no_evidence(self):
        agent=VisualAgent('once',Responses([('unknown','a')]))
        self.assertIsNone(self.tick(agent,0).color)
    def test_recovery_is_bounded_and_returns_to_goal(self):
        agent=VisualAgent('event_memory',Responses([('red','a'),('red','b')]),budget=3)
        self.tick(agent,0); self.tick(agent,60); agent.stale=4
        self.assertEqual(self.tick(agent,120).state,'recover_inspect')
        self.assertEqual(self.tick(agent,130).state,'recover_inspect')
        self.assertEqual(self.tick(agent,140).color,0)
        self.assertTrue(agent.recovery_used)
        self.assertEqual(agent.trace[-1]['operation'],'verify')
    def test_frontend_matches_validated_source(self):
        root=Path(__file__).resolve().parents[2]
        def function(path):
            tree=ast.parse(path.read_text())
            return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_semantic_rgb_features'),include_attributes=False)
        self.assertEqual(function(root/'scripts/vision/evaluate_feature_student.py'),function(root/'scripts/agentic/student_features.py'))
    def test_agent_has_no_simulator_imports(self):
        source=(Path(__file__).parent/'visual_agent.py').read_text()
        for forbidden in ['target_face','goal_rot','object_rot','semantic_metrics','isaaclab']:
            self.assertNotIn(forbidden,source)

if __name__=='__main__': unittest.main()
