"""Interface and leakage tests; mocks are not Astra performance evidence."""
import ast
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from astra_agent_v4 import AstraAgent, AstraClient, tools_schema, validate_action
from latent_data_v4 import split_layouts
from latent_goal_v4 import GoalHead
from latent_supervisor_v4 import LatentAgent


class Tests(unittest.TestCase):
    def test_goal_head_outputs_continuous_differentiable_tensor(self):
        torch.manual_seed(1)
        prototypes = torch.nn.functional.normalize(torch.randn(6, 768), dim=-1)
        head = GoalHead(12, prototypes)
        x = torch.randn(2, 12, requires_grad=True)
        z = head(x)['goal_embedding']
        self.assertEqual(z.shape, (2, 768))
        self.assertFalse(torch.any(torch.cdist(z, prototypes) < 1e-5))
        z[:, 0].sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.)

    def test_layout_banks_are_disjoint(self):
        banks = {split: {tuple(x) for seed in range(100) for x in split_layouts(seed, split)}
                 for split in ['train', 'val', 'test']}
        self.assertFalse(banks['train'] & banks['val'])
        self.assertFalse(banks['train'] & banks['test'])
        self.assertFalse(banks['val'] & banks['test'])

    def test_astra_requires_configured_credential(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'no Astra evaluation'):
                AstraClient()

    def test_astra_rejects_invalid_or_extra_tool_arguments(self):
        for name, args in [('inspect_face', {'color': 'red', 'steps': 1000, 'evidence': ''}),
                           ('reorient_target', {'color': 'red', 'steps': 20, 'evidence': '', 'goal_symbol': 'unknown'}),
                           ('abstain', {'evidence': '', 'target_pose': [1, 0, 0, 0]})]:
            with self.assertRaises(ValueError):
                validate_action(name, args)

    def test_astra_tool_schema_has_no_privileged_inputs(self):
        encoded = json.dumps(tools_schema())
        for forbidden in ['object_rotation', 'target_face', 'layout', 'success_rate', 'orientation_error']:
            self.assertNotIn(forbidden, encoded)

    def test_astra_feedback_loop_and_budget(self):
        class Client:
            def __init__(self): self.requests = []
            def ask(self, instruction, rgb, feedback, response_id, call_id):
                self.requests.append((instruction, feedback, response_id, call_id))
                return {'tool': 'reorient_target', 'arguments': {'goal_symbol': 'cross', 'color': 'red', 'steps': 20, 'evidence': 'mock'},
                        'response_id': 'response1', 'call_id': 'call1'}
        client = Client()
        agent = AstraAgent(client, 'Place the cross on top.', budget=2)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        for step in range(0, 100, 10): agent.tick(rgb, np.zeros(8), step)
        self.assertEqual(agent.calls, 2)
        self.assertEqual(client.requests[1][2:], ('response1', 'call1'))
        self.assertEqual(client.requests[1][1]['previous_skill'], 0)
        self.assertEqual(client.requests[0][0], 'Place the cross on top.')

    def test_latent_supervisor_preserves_raw_tensor(self):
        z = np.linspace(-1, 1, 768, dtype=np.float32)
        class Predictor:
            def predict(self, rgb, instruction):
                return {'goal_embedding': z.copy(), 'skill': 3, 'accepted': True, 'readable_probability': .9}
        agent = LatentAgent(Predictor(), 'Put the triangle on top.', budget=1)
        agent.tick(np.zeros((32, 32, 3), dtype=np.uint8), np.zeros(8), 0)
        np.testing.assert_array_equal(agent.goal_embedding, z)
        self.assertEqual(agent.goal, 3)

    def test_runtime_agents_import_no_scorer(self):
        for name in ['astra_agent_v4.py', 'latent_goal_v4.py', 'latent_supervisor_v4.py']:
            tree = ast.parse((Path(__file__).parent / name).read_text())
            imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            self.assertNotIn('latent_data_v4', imports)
            self.assertNotIn('semantic_protocol_v3', imports)


if __name__ == '__main__':
    unittest.main()
