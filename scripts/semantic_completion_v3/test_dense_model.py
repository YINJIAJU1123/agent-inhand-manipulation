import unittest
import torch
from dense_model import DenseCompletion, quaternion_matrix, rotation_matrix


class DenseModelContract(unittest.TestCase):
    def test_pose_convention_and_antipodal_labels(self):
        q = torch.tensor([[1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5]])
        expected = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                 [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        torch.testing.assert_close(quaternion_matrix(q), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(quaternion_matrix(q), quaternion_matrix(-q))
        six = expected[:, :, :2].transpose(-1, -2).reshape(2, 6)
        torch.testing.assert_close(rotation_matrix(six), expected)

    def test_goal_queries_permute_but_physical_pose_does_not(self):
        torch.manual_seed(42)
        args = [torch.randn(2, 6, 731), torch.randint(0, 255, (2, 3, 96, 96), dtype=torch.uint8),
                torch.rand(2, 1, 96, 96), torch.rand(2, 4), torch.ones(2, dtype=torch.bool)]
        permutation = torch.tensor([3, 1, 5, 2, 4, 0])
        for dense in (False, True):
            model = DenseCompletion(dense=dense).eval()
            with torch.no_grad():
                original = model(*args)
                permuted = model(args[0][:, permutation], *args[1:])
            torch.testing.assert_close(original['rotation'], permuted['rotation'], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(original['ordinal'][:, permutation], permuted['ordinal'], atol=1e-6, rtol=1e-5)
            self.assertFalse(torch.allclose(original['ordinal'][:, 0], original['ordinal'][:, 1]))
            if not dense:
                alternate = model(args[0], torch.zeros_like(args[1]), torch.zeros_like(args[2]), args[3]*0, ~args[4])
                torch.testing.assert_close(original['ordinal'], alternate['ordinal'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
