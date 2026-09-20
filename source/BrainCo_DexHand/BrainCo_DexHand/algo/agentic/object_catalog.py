"""Versioned procedural object bank with six explicit planar marker surfaces.

All distances are metres. Object identity and surface frames are simulator
metadata, never policy inputs. Fixed per-slot assignment balances PPO sampling.
"""
from __future__ import annotations

import math

CATALOG_VERSION = "procedural_surfaces_v1"
PATCH_SIZE = .020
PATCH_THICKNESS = .001
PATCH_GAP = .0002
NORMALS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
           (0., -1., 0.), (0., 0., 1.), (0., 0., -1.))

# Object instances, rather than marker permutations, define these splits.
SPECS = (
    ("box_70", "train", "box", (.070, .070, .070), 400.),
    ("box_60_70_80", "train", "box", (.060, .070, .080), 400.),
    ("hex_36_70", "train", "hex", (.036, .070), 400.),
    ("hex_40_60", "train", "hex", (.040, .060), 400.),
    ("oct_38_70", "train", "oct", (.038, .070), 400.),
    ("oct_34_80", "train", "oct", (.034, .080), 400.),
    ("box_65_75_60", "val", "box", (.065, .075, .060), 400.),
    ("hex_38_65", "val", "hex", (.038, .065), 400.),
    ("oct_36_75", "val", "oct", (.036, .075), 400.),
    ("box_80_55_70", "test", "box", (.080, .055, .070), 400.),
    ("hex_34_80", "test", "hex", (.034, .080), 400.),
    ("oct_40_60", "test", "oct", (.040, .060), 400.),
)


def normal_rotation(normal):
    """Unit wxyz quaternion mapping local +Z onto an outward normal."""
    x, y, z = normal
    if z < -.999999:
        return [0., 1., 0., 0.]
    q = [1. + z, -y, x, 0.]
    length = math.sqrt(sum(v * v for v in q))
    return [v / length for v in q]


def object_spec(identifier):
    matches = [row for row in SPECS if row[0] == identifier]
    if not matches:
        raise ValueError(f"Unknown object {identifier!r}")
    identifier, split, family, dimensions, density = matches[0]
    if family == "box":
        a, b, c = (v / 2 for v in dimensions)
        vertices = [[-a, -b, -c], [a, -b, -c], [a, b, -c], [-a, b, -c],
                    [-a, -b, c], [a, -b, c], [a, b, c], [-a, b, c]]
        faces = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        normals = NORMALS
        offsets = [a, a, b, b, c, c]
        volume = 8 * a * b * c
    else:
        sides = 6 if family == "hex" else 8
        radius, height = dimensions
        apothem = radius * math.cos(math.pi / sides)
        ring = [(radius * math.cos((2 * i + 1) * math.pi / sides),
                 radius * math.sin((2 * i + 1) * math.pi / sides)) for i in range(sides)]
        vertices = [[x, y, z] for z in (-height / 2, height / 2) for x, y in ring]
        faces = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
        faces += [[i, (i + 1) % sides, (i + 1) % sides + sides, i + sides] for i in range(sides)]
        normals = ([(math.cos(2 * i * math.pi / sides), math.sin(2 * i * math.pi / sides), 0.)
                    for i in range(sides)] if family == "hex" else NORMALS)
        offsets = [apothem] * 6 if family == "hex" else [apothem] * 4 + [height / 2] * 2
        volume = sides * radius**2 * math.sin(2 * math.pi / sides) * height / 2
    surfaces = [{"normal": list(n), "center": [d * v for v in n],
                 "rotation": normal_rotation(n), "size": [PATCH_SIZE, PATCH_SIZE]}
                for n, d in zip(normals, offsets)]
    return {"id": identifier, "split": split, "family": family,
            "dimensions": list(dimensions), "density": density, "volume": volume,
            "mass": volume * density, "vertices": vertices, "faces": faces,
            "surfaces": surfaces, "catalog_version": CATALOG_VERSION}


def select_objects(split, identifiers=None):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown object split {split!r}")
    selected = identifiers or [row[0] for row in SPECS if row[1] == split]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Object selection must be nonempty and unique")
    specs = [object_spec(name) for name in selected]
    if any(spec["split"] != split for spec in specs):
        raise ValueError("Object selection crosses the declared split")
    return specs


def validate_balanced_slots(num_envs, num_objects):
    if num_envs < num_objects or num_envs % num_objects:
        raise ValueError(f"num_envs={num_envs} must be a positive multiple of {num_objects} objects")
