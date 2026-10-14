"""Training-only visibility labels. Never import this module in an agent."""
from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn.functional as F


def split_layouts(seed, split, count=6):
    """Disjoint permutations, as well as disjoint episode seeds, across splits."""
    bank = np.asarray(list(itertools.permutations(range(6))))
    bank = bank[np.random.default_rng(31990).permutation(len(bank))]
    start, end = {'train': (0, 480), 'val': (480, 600), 'test': (600, 720)}[split]
    return bank[np.random.default_rng(seed).choice(np.arange(start, end), count, replace=False)].tolist()


def visibility_labels(raw, camera):
    """One-sided depth-tested symbol plane; conservative readability proxy, not human truth.

    Labels are privileged supervision only. No projected boxes enter model input.
    The 18 mm symbol is inset within a 28 mm sticker; its surface is 37.27 mm
    from the cube center. Camera depth is optical-axis depth in this environment.
    """
    from isaaclab.utils.math import quat_apply
    from BrainCo_DexHand.algo.agentic.search_protocol import project_points

    n = raw.num_envs
    uv = torch.linspace(-.0075, .0075, 7, device=raw.device)
    xx, yy = torch.meshgrid(uv, uv, indexing='ij')
    grid = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(49, device=raw.device)), -1)
    local = quat_apply(raw._face_marker_rots[:, None].expand(-1, 49, -1).reshape(-1, 4),
                       grid[None].expand(6, -1, -1).reshape(-1, 3)).reshape(6, 49, 3)
    local += raw._face_normals[:, None] * .03727
    world = quat_apply(raw.object_rot[:, None, None].expand(-1, 6, 49, -1).reshape(-1, 4),
                       local[None].expand(n, -1, -1, -1).reshape(-1, 3)).reshape(n, 294, 3)
    center = raw.object_pos + raw.scene.env_origins
    world += center[:, None]
    cam = raw.camera.data
    pixels, z = project_points(world, cam.pos_w, cam.quat_w_ros, cam.intrinsic_matrices)
    depth = camera['depth'].reshape(n, *camera['depth'].shape[1:3])
    h, w = depth.shape[1:]
    x, y = pixels.unbind(-1)
    inside = (z > .05) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    xi, yi = x.round().long().clamp(0, w - 1), y.round().long().clamp(0, h - 1)
    observed = depth[torch.arange(n, device=raw.device)[:, None], yi, xi]
    # Point-instanced visual decals are absent from this camera's depth buffer.
    # Require no closer occluder, and a nearby backing surface behind the decal.
    passed = inside & torch.isfinite(observed) & (observed >= z - .002) & (observed <= z + .03)
    normal = quat_apply(raw.object_rot[:, None].expand(-1, 6, -1).reshape(-1, 4),
                        raw._face_normals[None].expand(n, -1, -1).reshape(-1, 3)).reshape(n, 6, 3)
    facing = (normal * F.normalize(cam.pos_w - center, dim=-1)[:, None]).sum(-1)
    fraction = passed.reshape(n, 6, 49).float().mean(-1)
    fraction = torch.where(facing > 0, fraction, 0.)
    area = (.018 ** 2 * cam.intrinsic_matrices[:, 0, 0, None]
            * cam.intrinsic_matrices[:, 1, 1, None] * facing.clamp_min(0)
            / z.reshape(n, 6, 49).mean(-1).clamp_min(.05).square())
    readable = (fraction >= .8) & (facing >= .35) & (area >= 100)
    return {'visible_fraction': fraction.cpu().tolist(), 'facing': facing.cpu().tolist(),
            'projected_area': area.cpu().tolist(), 'readable': readable.cpu().tolist(),
            'depth_residual_median': (observed-z).reshape(n, 6, 49).median(-1).values.cpu().tolist(),
            'predicted_depth_median': z.reshape(n, 6, 49).median(-1).values.cpu().tolist(),
            'projected_points': pixels.reshape(n, 6, 49, 2).cpu().tolist()}
