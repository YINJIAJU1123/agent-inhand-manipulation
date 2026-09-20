"""Validate the Revo3 VisERDex visual-language deployment contract offline."""

from __future__ import annotations

import numpy as np

from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.viserdex_input_builder import FRAME_DIM, HIST_LEN, OBS_FRAMES, VisERDexInputBuilder


def main() -> None:
    profile = Revo3Profile.load("config/revo3_right.yaml")
    builder = VisERDexInputBuilder(profile.joint_lower_policy, profile.joint_upper_policy, profile.policy_joint_order)
    inputs = builder.reset(
        profile.joint_lower_policy,
        np.array([0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        0.91,
        np.linspace(-1.0, 1.0, builder.language_dim, dtype=np.float32),
    )
    assert inputs["obs"].shape == (1, OBS_FRAMES * FRAME_DIM)
    assert inputs["proprio_hist"].shape == (1, HIST_LEN, 42)
    assert inputs["pose_hist"].shape == (1, HIST_LEN, 8)
    assert inputs["language"].shape == (1, builder.language_dim)
    assert np.isclose(np.linalg.norm(inputs["pose_hist"][0, 0, 3:7]), 1.0)
    target = builder.action_to_target(np.ones(21, dtype=np.float32))
    assert np.all(target <= profile.joint_upper_policy)
    print({key: tuple(value.shape) for key, value in inputs.items()})
    print("VisERDex Revo3 IO smoke passed")


if __name__ == "__main__":
    main()
