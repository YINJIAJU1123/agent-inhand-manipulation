"""Control handoff regressions, including reply and re-request in one tick."""
import unittest
import torch
from retention_runtime import RetentionRuntime

class SelectFromFeature(torch.nn.Module):
    def forward(self, x):
        return {'value': x[..., :3]}

class HandoffTests(unittest.TestCase):
    def inputs(self, **changes):
        bank = torch.zeros(2, 6, 125)
        bank[:, :, 2] = 2.  # Choose hold except for color 4, which chooses slow.
        bank[:, 4, 1] = 3.
        x = dict(step=10, states=['execute', 'verify'], colors=[2, 1],
                 pending=[True, True], delivered=[False, False], request_ids=[2, 5],
                 active=[True, True], ready_probability=torch.full((2, 6), .95),
                 feature_bank=bank, fresh=True, student_action=torch.ones(2, 21),
                 current=torch.zeros(2, 21), base_action=torch.full((2, 21), .7))
        x.update(changes)
        return x

    def test_reply_releases_even_if_next_request_is_already_pending(self):
        r = RetentionRuntime(2, 'learned', SelectFromFeature())
        r.apply(**self.inputs())
        x = self.inputs(step=20, delivered=[True, False], request_ids=[3, 5])
        out = r.apply(**x)
        torch.testing.assert_close(out[0], x['base_action'][0])
        torch.testing.assert_close(out[1], x['current'][1])
        self.assertIsNone(r.windows[0].start)

    def test_timeout_cannot_renew_in_the_same_request(self):
        r = RetentionRuntime(2, 'hold')
        r.apply(**self.inputs())
        for step in (190, 200):
            x = self.inputs(step=step)
            torch.testing.assert_close(r.apply(**x), x['base_action'])
        out = r.apply(**self.inputs(step=210, request_ids=[3, 6]))
        torch.testing.assert_close(out, torch.zeros(2, 21))

    def test_changed_goal_releases_and_uses_its_own_measured_features(self):
        r = RetentionRuntime(2, 'learned', SelectFromFeature())
        r.apply(**self.inputs())
        x = self.inputs(step=11, colors=[4, 1], fresh=False)
        torch.testing.assert_close(r.apply(**x)[0], x['base_action'][0])
        out = r.apply(**self.inputs(step=20, colors=[4, 1]))
        torch.testing.assert_close(out[0], torch.full((21,), .25))

    def test_inspection_unknown_and_inactive_cases_do_not_inherit_hold(self):
        variants = [dict(states=['inspect', 'verify']), dict(colors=[None, 1]),
                    dict(active=[False, True]), dict(pending=[False, True])]
        for variant in variants:
            r = RetentionRuntime(2, 'hold')
            r.apply(**self.inputs())
            x = self.inputs(step=11, **variant)
            torch.testing.assert_close(r.apply(**x)[0], x['base_action'][0])

    def test_none_is_bitwise_identical_and_does_not_infer(self):
        r = RetentionRuntime(2, 'none')
        x = self.inputs(feature_bank=None, ready_probability=None)
        self.assertTrue(torch.equal(r.apply(**x), x['base_action']))
        self.assertEqual(r.events, [])

    def test_slow_matches_physical_counterfactual_formula(self):
        r = RetentionRuntime(2, 'learned', SelectFromFeature())
        x = self.inputs(colors=[4, 4], current=torch.full((2, 21), -.3))
        out = r.apply(**x)
        torch.testing.assert_close(out, x['current'] + .25*(x['student_action']-x['current']))

if __name__ == '__main__':
    unittest.main()
