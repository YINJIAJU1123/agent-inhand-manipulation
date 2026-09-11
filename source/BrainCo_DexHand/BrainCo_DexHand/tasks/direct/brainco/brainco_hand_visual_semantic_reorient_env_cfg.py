"""Camera-enabled configuration for the semantic Revo3 reorientation task.

The state-teacher environment intentionally remains camera-free.  This config
adds a fixed, front-facing tiled RGB camera for collecting visual trajectories
and for the later vision-language student.  The camera output is exposed by
``VisualSemanticReorientEnv.capture_camera`` rather than concatenated into the
PPO observation, so existing state-teacher checkpoints stay compatible.
"""

from pathlib import Path

from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .brainco_hand_semantic_reorient_env_cfg import BrainCoHandSemanticReorientEnvCfg


@configclass
class BrainCoHandVisualSemanticReorientEnvCfg(BrainCoHandSemanticReorientEnvCfg):
    """Semantic teacher plus a fixed RGB camera.

    ``include_camera_in_policy`` is deliberately false for the first data
    collection stage.  Camera frames can be read from the sensor directly and
    fused by a visual policy with its own encoder.  Setting it true is reserved
    for a custom multimodal runner that updates ``observation_space``.
    """

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/SemanticCamera",
        # Camera looks from the front (negative world-y) at the hand/object.
        # Isaac cameras look along local -Z; +90 degrees about X points this
        # axis toward +Y and keeps world +Z approximately upright.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.72, 0.72),
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=0.72,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        width=128,
        height=128,
    )
    # Use the repository's six-color cube for visual collection.  The
    # collision mesh and inertial parameters are embedded in this URDF.  The
    # state teacher retains the Nucleus cube and is therefore checkpoint
    # compatible; only this camera-enabled data-collection task uses it.
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(
                Path(__file__).resolve().parents[6]
                / "assets"
                / "urdf"
                / "objects"
                / "cube_multicolor.urdf"
            ),
            fix_base=False,
            merge_fixed_joints=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.11, 0.56), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    # This keeps the observation interface equal to the semantic state
    # teacher.  A multimodal runner should fuse ``capture_camera()`` output.
    include_camera_in_policy: bool = False
    camera_frame_stack: int = 1
