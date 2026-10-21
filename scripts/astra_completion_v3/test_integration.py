"""Contracts for conditional completion advice and matched service accounting."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from completion_runtime import DenseEvidenceRuntime
spec = importlib.util.spec_from_file_location('dense_eval_analysis', HERE/'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class Contracts(unittest.TestCase):
    def test_conditional_gate_and_baseline_feedback(self):
        runtime = DenseEvidenceRuntime.__new__(DenseEvidenceRuntime)
        runtime.last = torch.tensor([[.1, .8, .2, .3, .4, .5]])
        runtime.threshold = .7
        self.assertTrue(runtime.can_stop(0, 1))
        self.assertFalse(runtime.can_stop(0, 0))
        self.assertFalse(runtime.can_stop(0, None))
        self.assertEqual(runtime.summary(0, False), {})
        # Pose estimates cannot validate a symbol binding; scorer must do so.
        self.assertEqual(runtime.can_stop(0, 1, 'star', 'star up'),
                         runtime.can_stop(0, 1, 'bars', 'star up'))
        self.assertNotIn('progress', runtime.summary(0, True))

    def test_service_exclusion_is_paired_and_raw_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'requests').mkdir()
            (root/'protocol.json').write_text(json.dumps({'total_episodes': 288}))
            for arm, anomaly, success in [('baseline', True, False), ('verify', False, True)]:
                folder = root/('eval_'+arm)
                folder.mkdir()
                job = dict(repeat=1, seed=86101, template='canonical', arm=arm)
                row = dict(case=0, service_anomaly=anomaly, success=success,
                    reached=success, drop=False, wait_drop=False, cold_drop=False,
                    false_stop=False, stop_claimed=success, calls=1,
                    reason='success' if success else 'infrastructure_error', controller_events=[])
                (folder/'job.json').write_text(json.dumps(job))
                (folder/'initial.json').write_text(json.dumps({'joint_pos': [1, 2]}))
                (folder/'report.json').write_text(json.dumps({'episodes': [row]}))
            summary = analysis.summarize(root)
            self.assertEqual(summary['completed_episodes'], 2)
            self.assertEqual(summary['raw']['verify']['successes'], 1)
            self.assertEqual(summary['normal_service_matched']['verify']['episodes'], 0)
            self.assertEqual(summary['normal_service_matched']['baseline']['episodes'], 0)
            self.assertTrue(summary['initial_states_equal'])
            self.assertEqual(summary['paired_vs_baseline']['verify']['pairs'], 0)


if __name__ == '__main__':
    unittest.main()
