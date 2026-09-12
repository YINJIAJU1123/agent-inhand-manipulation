"""Camera-enabled configuration for the semantic Revo3 reorientation task.

The state-teacher environment intentionally remains camera-free.  This config
adds a fixed, front-facing tiled RGB camera for collecting visual trajectories
and for the later vision-language student.  The camera output is exposed by
``VisualSemanticReorientEnv.capture_camera`` rather than concatenated into the
PPO observation, so existing state-teacher checkpoints stay compatible.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

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
        # Fixed eye-to-hand camera, mounted in front of and above the hand.
        # The object starts near (0, -0.11, 0.56).  From
        # (0, -0.72, 0.95), OpenGL -Z must point along (0, 0.61, -0.39).
        # This requires +57.4074 degrees about X in the OpenGL convention,
        # giving 32.5926 degrees of depression below the horizon.
        # Isaac Lab's "world" convention points along +X instead: using the
        # same quaternion in that convention would miss the object.
        # Image framing and occlusion still require a rendered validation.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.72, 0.95),
            rot=(0.877115072743617, 0.48028028188336, 0.0, 0.0),
            convention="opengl",
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
    # Use an Isaac primitive for the camera task.  This avoids depending on
    # the optional URDF importer extension in headless Isaac Sim workers.
    # A textured six-face USD/OBJ can replace this spawn later without
    # changing the camera or policy interfaces.
    object_cfg: RigidObjectCfg = BrainCoHandSemanticReorientEnvCfg.object_cfg.replace(
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.07),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.85)),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
        )
    )
    # This keeps the observation interface equal to the semantic state
    # teacher.  A multimodal runner should fuse ``capture_camera()`` output.
    include_camera_in_policy: bool = False
    camera_frame_stack: int = 1
