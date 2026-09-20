"""Isaac adapters for heterogeneous rigid objects and their marker frames."""
from __future__ import annotations

import torch
from pxr import UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg

from BrainCo_DexHand.algo.agentic.object_catalog import validate_balanced_slots


def spawn_surface_object(prim_path, cfg, translation=None, orientation=None, **kwargs):
    spec = cfg.object_spec
    stage = sim_utils.get_current_stage()
    root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    root.SetCustomDataByKey("search_object_id", spec["id"])
    mesh_path = f"{prim_path}/geometry/mesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(spec["vertices"])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in spec["faces"]])
    mesh.CreateFaceVertexIndicesAttr([v for f in spec["faces"] for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    sim_utils.define_collision_properties(mesh_path, cfg.collision_props)
    sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    sim_utils.define_mass_properties(prim_path, cfg.mass_props)
    for label, material, bind in (
        ("visual", cfg.visual_material, sim_utils.bind_visual_material),
        ("physics", cfg.physics_material, sim_utils.bind_physics_material),
    ):
        path = f"{prim_path}/{label}"
        material.func(path, material)
        bind(mesh_path, path)
    return root


@configclass
class SurfaceObjectCfg(RigidObjectSpawnerCfg):
    func = spawn_surface_object
    object_spec: dict = None
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.3, .3, .3))
    physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.)


def configure_objects(cfg, specs):
    validate_balanced_slots(cfg.scene.num_envs, len(specs))
    rigid = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, enable_gyroscopic_forces=True,
        solver_position_iteration_count=8, solver_velocity_iteration_count=0,
        sleep_threshold=.005, stabilization_threshold=.0025, max_depenetration_velocity=1000.)
    assets = [SurfaceObjectCfg(object_spec=spec, rigid_props=rigid,
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=spec["mass"])) for spec in specs]
    cfg.object_cfg = cfg.object_cfg.replace(
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=False))
    cfg.scene.replicate_physics = False
    cfg.object_specs = specs


def read_object_geometry(env):
    """Read actual PhysX view order, avoiding assumptions about USD path sorting."""
    specs = env.cfg.object_specs
    stage = sim_utils.get_current_stage()
    paths = env.object.root_physx_view.prim_paths
    names = [stage.GetPrimAtPath(path).GetCustomDataByKey("search_object_id") for path in paths]
    expected = [spec["id"] for spec in specs]
    if len(names) != env.num_envs or any(name not in expected for name in names):
        raise RuntimeError(f"Spawned object identity mismatch: {names}")
    counts = [names.count(name) for name in expected]
    if len(set(counts)) != 1:
        raise RuntimeError(f"Unbalanced spawned objects: {counts}")
    indices = [expected.index(name) for name in names]
    surfaces = [specs[i]["surfaces"] for i in indices]
    return {
        "names": names,
        "index": torch.tensor(indices, device=env.device, dtype=torch.long),
        **{key: torch.tensor([[s[key] for s in row] for row in surfaces], device=env.device)
           for key in ("normal", "center", "rotation", "size")},
    }
