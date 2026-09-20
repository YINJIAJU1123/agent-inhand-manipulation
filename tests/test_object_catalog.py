"""Geometry and partition invariants independent of the simulator."""
import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).parents[1] / "source/BrainCo_DexHand/BrainCo_DexHand/algo/agentic/object_catalog.py"
spec = importlib.util.spec_from_file_location("object_catalog", path)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def rotate(q, v):
    return v + 2 * torch.cross(q[1:], torch.cross(q[1:], v, dim=-1) + q[0] * v, dim=-1)


def test_disjoint_balanced_object_splits():
    banks = [catalog.select_objects(split) for split in ("train", "val", "test")]
    assert list(map(len, banks)) == [6, 3, 3]
    assert len({obj["id"] for bank in banks for obj in bank}) == 12
    for bank in banks:
        assert {obj["family"] for obj in bank} == {"box", "hex", "oct"}
    with pytest.raises(ValueError):
        catalog.select_objects("train", ["hex_34_80"])
    with pytest.raises(ValueError):
        catalog.validate_balanced_slots(64, 6)
    catalog.validate_balanced_slots(96, 6)


@pytest.mark.parametrize("identifier", [row[0] for row in catalog.SPECS])
def test_closed_outward_mesh_and_surface_markers(identifier):
    obj = catalog.object_spec(identifier)
    vertices = torch.tensor(obj["vertices"], dtype=torch.float64)
    planes = []
    directed_edges = []
    volume = 0.
    for face in obj["faces"]:
        a, b, c = vertices[face[:3]]
        normal = torch.nn.functional.normalize(torch.cross(b - a, c - a, dim=-1), dim=0)
        offset = normal @ a
        assert offset > 0
        assert ((vertices @ normal) <= offset + 1e-9).all()
        planes.append((normal, offset))
        directed_edges.extend(zip(face, face[1:] + face[:1]))
        for i in range(1, len(face) - 1):
            volume += float(a @ torch.cross(vertices[face[i]], vertices[face[i+1]], dim=-1)) / 6
    assert all(directed_edges.count((b, a)) == 1 for a, b in directed_edges)
    assert volume == pytest.approx(obj["volume"])
    assert obj["mass"] == pytest.approx(volume * obj["density"])
    for surface in obj["surfaces"]:
        normal, center, rotation = (torch.tensor(surface[key], dtype=torch.float64)
                                     for key in ("normal", "center", "rotation"))
        assert torch.allclose(rotate(rotation, torch.tensor([0., 0., 1.], dtype=torch.float64)), normal)
        assert any(torch.allclose(normal, n) and abs(float(n @ center - d)) < 1e-9 for n, d in planes)
        half = catalog.PATCH_SIZE / 2
        for x in (-half, half):
            for y in (-half, half):
                corner = center + rotate(rotation, torch.tensor([x, y, 0.], dtype=torch.float64))
                assert all(n @ corner <= d + 1e-9 for n, d in planes)
