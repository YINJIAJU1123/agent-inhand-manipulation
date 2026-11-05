"""Frozen released RGB CNN measurements for the pose-latent student.

Call after initializing the simulation runtime. Images must already be rectified
into the supplied pinhole calibration. No object state or goal is accepted here.
"""

import hashlib
from pathlib import Path

import torch


class ReleasedPoseEstimator:
    def __init__(self, checkpoint, keypoints, device, image_size=(120, 120)):
        from viserdex.tasks.manipulation.inhand.pose_estimator.networks import PoseEstimatorNetwork
        from viserdex.tasks.manipulation.inhand.pose_estimator.transforms import get_transforms

        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        checkpoint, keypoints = Path(checkpoint), Path(keypoints)
        self.network = PoseEstimatorNetwork(27, pretrained=False).to(self.device)
        self.network.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True), strict=True)
        self.network.eval().requires_grad_(False)
        # Opening is deterministic at p=1 and must remain active as in training.
        self.preprocess = get_transforms(['binary_opening'], str(self.device))
        points = torch.load(keypoints, map_location=self.device, weights_only=True)
        if points.shape != (8, 3) or not torch.isfinite(points).all():
            raise ValueError('Expected the released eight object keypoints')
        self.reference = torch.cat((points.new_zeros(1, 3), points)).double()
        self.metadata = {
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'keypoints_sha256': hashlib.sha256(keypoints.read_bytes()).hexdigest(),
            'image_size': list(self.image_size),
            'pose_frame': 'right_hand_base_link',
            'quaternion_order': 'xyzw',
            'position_unit': 'm',
            'cnn_preprocessing': ['binary_opening', 'released_network_resize_and_normalization'],
            'visual_noise_added_to_cnn_pose': False,
            'ground_truth_fallback': False,
        }

    @torch.inference_mode()
    def predict_rectified(
        self, pixels, intrinsics, camera_rotation_w, camera_position_w,
        base_rotation_w, base_position_w,
    ):
        """Decode float RGB [N,3,H,W] in [0,1] into measured object pose.

        Rotations map local coordinates to world coordinates. The camera is in
        OpenCV convention. Camera and robot positions share absolute world
        origins. A single calibrated 3x3 intrinsic matrix applies to the batch.
        """
        from isaaclab.utils.math import quat_from_matrix
        from viserdex.distillation.noise import quat_unique
        from viserdex.distillation.vision import keypoints_to_frame_pose
        from viserdex.utils.pose_metrics import fit_rigid_pose, unproject_uvd

        width, height = self.image_size
        if (pixels.ndim != 4 or pixels.shape[1:] != (3, height, width)
                or pixels.device != self.device or not pixels.is_floating_point()
                or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1):
            raise ValueError('Expected calibrated finite float RGB [N,3,H,W] in [0,1] on the estimator device')
        if intrinsics.shape != (3, 3) or intrinsics.device != self.device:
            raise ValueError('Expected one shared 3x3 pinhole intrinsic matrix on the estimator device')
        prediction = self.network(self.preprocess(pixels))
        if not torch.isfinite(prediction).all() or (prediction.reshape(-1, 9, 3)[..., 2] <= 0).any():
            raise FloatingPointError('Invalid CNN keypoints/depth; no synthetic or stale pose fallback')
        rotation, position = keypoints_to_frame_pose(
            prediction, intrinsics, self.reference, camera_rotation_w, camera_position_w,
            base_rotation_w, base_position_w, self.image_size,
        )
        quaternion = quat_unique(quat_from_matrix(rotation))
        points = unproject_uvd(prediction.double(), intrinsics.double(), self.image_size)
        fit_rotation, fit_position = fit_rigid_pose(self.reference, points)
        fitted = self.reference @ fit_rotation.transpose(-1, -2) + fit_position[:, None]
        return {
            'position': position,
            'quaternion': quaternion,
            'keypoints_uvd': prediction,
            'fit_error_m': (points - fitted).norm(dim=-1).mean(-1).to(position.dtype),
        }
