"""Critical protocol invariants, independent of Isaac Sim."""
import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).parents[1] / "source/BrainCo_DexHand/BrainCo_DexHand/algo/agentic/search_protocol.py"
spec = importlib.util.spec_from_file_location("search_protocol", path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_layout_splits_disjoint_and_semantic_id_is_not_fixed_face():
    banks = [protocol.layout_bank(s) for s in ("train", "val", "test")]
    sets = [set(map(tuple, bank.tolist())) for bank in banks]
    assert [len(s) for s in sets] == [576, 72, 72]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    for bank in banks:
        assert torch.equal(bank.sort(-1).values, torch.arange(6).expand_as(bank))
        for color in range(6):
            assert len(set((bank == color).long().argmax(-1).tolist())) == 6


def test_instruction_roundtrip_rejects_ambiguous_or_unsupported_language():
    for i in range(6):
        assert protocol.parse_instruction(protocol.instruction(i)) == i
    with pytest.raises(ValueError):
        protocol.parse_instruction("Show red and green")
    with pytest.raises(ValueError):
        protocol.parse_instruction("Show the triangle")


def test_occlusion_offscreen_and_behind_camera_do_not_count_as_visible():
    points = torch.tensor([[[0., 0., 1.], [.1, 0., 1.], [4., 0., 1.], [0., 0., -1.]]])
    k = torch.tensor([[[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]]])
    depth = torch.ones(1, 10, 10, 1)
    depth[0, 5, 6] = .8  # finger in front of the second sample
    visible, _, _ = protocol.sample_visibility(points, torch.zeros(1, 3), torch.tensor([[1., 0., 0., 0.]]), k, depth)
    assert visible.item() == .25


def test_dwell_requires_continuity_and_drop_invalidates_terminal_success():
    count = torch.zeros(1, dtype=torch.long)
    for _ in range(30):
        count, success = protocol.update_dwell(count, torch.tensor([True]), torch.tensor([False]), 1/30, 1.)
    assert success.item()
    count, success = protocol.update_dwell(count, torch.tensor([False]), torch.tensor([False]), 1/30, 1.)
    assert count.item() == 0 and not success.item()
    count, success = protocol.update_dwell(torch.tensor([100]), torch.tensor([True]), torch.tensor([True]), 1/30, 1.)
    assert not success.item()


def test_scan_recognizes_only_image_evidence_and_clears_episode_history():
    rgb = torch.zeros(1, 16, 16, 3, dtype=torch.uint8)
    rgb[:, 4:12, 4:12] = torch.tensor([217, 13, 13], dtype=torch.uint8)
    scan = protocol.ScanRecognizeHold(1, "cpu")
    for _ in range(3):
        goal = scan(rgb, torch.tensor([0]))
    assert scan.found.item() and goal.item() == 0
    scan.reset(torch.tensor([True]))
    assert not scan.found.item() and scan.tick.item() == 0
    for _ in range(4):
        scan(rgb, torch.tensor([2]))
    assert not scan.found.item()
