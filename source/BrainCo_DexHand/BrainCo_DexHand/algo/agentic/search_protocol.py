"""Simulator-independent semantic search protocol and image-only features.

The v0 grammar names six colors. Color IDs are never local face IDs; the
layout is independently permuted at reset. No open-vocabulary claim is made.
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

COLORS = ("red", "green", "blue", "yellow", "magenta", "cyan")
PALETTE = ((.85, .05, .05), (.05, .75, .15), (.05, .25, .9),
           (.95, .75, .05), (.8, .05, .75), (.05, .8, .85))
NORMALS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
           (0., -1., 0.), (0., 0., 1.), (0., 0., -1.))


def instruction(color: int) -> str:
    return f"Show the {COLORS[color]} marker."


def parse_instruction(text: str) -> int:
    words = text.lower().replace(".", "").split()
    found = [i for i, c in enumerate(COLORS) if c in words]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one supported color in {text!r}")
    return found[0]


def layout_bank(split: str, device="cpu") -> torch.Tensor:
    """Stable disjoint permutation splits: 576 train / 72 val / 72 test."""
    perms = torch.tensor(list(itertools.permutations(range(6))), device=device)
    order = torch.randperm(720, generator=torch.Generator().manual_seed(20260920)).to(device)
    ranges = {"train": (0, 576), "val": (576, 648), "test": (648, 720)}
    if split not in ranges:
        raise ValueError(f"Unknown layout split {split}")
    start, stop = ranges[split]
    return perms[order[start:stop]]


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scalar-first quaternion rotation with broadcastable leading axes."""
    q, v = torch.broadcast_tensors(q, F.pad(v, (0, 1)))
    v = v[..., :3]
    t = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def project_points(points, camera_pos, camera_quat_ros, intrinsics):
    """World points to ROS camera pixels and optical-axis depth."""
    inv = camera_quat_ros.clone()
    inv[..., 1:] *= -1
    cam = quat_rotate(inv[:, None], points - camera_pos[:, None])
    z = cam[..., 2]
    pixels = torch.stack((intrinsics[:, None, 0, 0] * cam[..., 0] / z.clamp_min(1e-6)
                          + intrinsics[:, None, 0, 2],
                          intrinsics[:, None, 1, 1] * cam[..., 1] / z.clamp_min(1e-6)
                          + intrinsics[:, None, 1, 2]), dim=-1)
    return pixels, z


def sample_visibility(points, camera_pos, camera_quat_ros, intrinsics, depth,
                      tolerance=.006):
    """Depth-test surface samples; offscreen and behind-camera samples fail.

    Requires optical-axis depth (`distance_to_image_plane`), not ray length.
    Ground-truth geometry is used for reward/evaluation only.
    """
    pixels, z = project_points(points, camera_pos, camera_quat_ros, intrinsics)
    n, h, w = depth.shape[:3]
    x, y = pixels.unbind(-1)
    inside = (z > .05) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    xi = x.round().long().clamp(0, w - 1)
    yi = y.round().long().clamp(0, h - 1)
    observed = depth.reshape(n, h, w)[torch.arange(n, device=depth.device)[:, None], yi, xi]
    visible = inside & torch.isfinite(observed) & ((observed - z).abs() <= tolerance)
    return visible.float().mean(-1), pixels, z


def image_color_masks(rgb):
    """Simple shared color detector; reads RGB only, no simulator semantics."""
    image = rgb[..., :3].float() / 255.
    chroma = image / image.sum(-1, keepdim=True).clamp_min(.05)
    palette = torch.tensor(PALETTE, device=rgb.device)
    palette = palette / palette.sum(-1, keepdim=True)
    distance = ((chroma[..., None, :] - palette) ** 2).sum(-1)
    best, label = distance.min(-1)
    saturation = image.max(-1).values - image.min(-1).values
    valid = (best < .035) & (saturation > .18) & (image.max(-1).values > .2)
    return F.one_hot(label, 6).bool() & valid[..., None]


def image_features(rgb, depth, side=16):
    """Fixed RGB-D spatial pooling plus image-only color centroids.

    A transparent engineering encoder for the initial recurrent PPO pilot;
    this is not a pretrained VLM or a learned vision representation.
    """
    rgb_small = F.adaptive_avg_pool2d(rgb[..., :3].permute(0, 3, 1, 2).float() / 255., side)
    d = torch.nan_to_num(depth.float(), nan=2., posinf=2., neginf=0.).clamp(0, 2)
    d_small = F.adaptive_avg_pool2d(d.permute(0, 3, 1, 2) / 2., side)
    masks = image_color_masks(rgb).float()
    n, h, w, _ = masks.shape
    count = masks.sum((1, 2))
    xx = torch.linspace(-1, 1, w, device=rgb.device)[None, None, :, None]
    yy = torch.linspace(-1, 1, h, device=rgb.device)[None, :, None, None]
    cx = (masks * xx).sum((1, 2)) / count.clamp_min(1)
    cy = (masks * yy).sum((1, 2)) / count.clamp_min(1)
    stats = torch.stack((count / (h * w), cx, cy), -1).flatten(1)
    return torch.cat((rgb_small.flatten(1), d_small.flatten(1), stats), -1)


def update_dwell(previous, qualified, dropped, step_dt, hold_s):
    steps = torch.where(qualified & ~dropped, previous + 1, torch.zeros_like(previous))
    return steps, (steps >= math.ceil(hold_s / step_dt)) & ~dropped


class ScanRecognizeHold:
    """Image-only scan schedule using a shared visible-goal controller.

    Every interval selects the next currently visible non-requested color in
    cyclic order. On seeing the requested color for consecutive frames, the
    same controller receives that goal. Reacquire after a sustained loss.
    This cannot guarantee surface coverage; that limitation is measured.
    """
    def __init__(self, num_envs, device, switch_steps=60, min_pixels=12):
        self.tick = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = torch.zeros_like(self.tick)
        self.selected = torch.zeros_like(self.tick)
        self.found = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.evidence = torch.zeros_like(self.tick)
        self.lost = torch.zeros_like(self.tick)
        self.switch_steps, self.min_pixels = switch_steps, min_pixels

    def reset(self, done):
        for value in (self.tick, self.cursor, self.selected, self.found, self.evidence, self.lost):
            value[done] = 0

    def __call__(self, rgb, requested):
        counts = image_color_masks(rgb).sum((1, 2))
        seen = counts.gather(1, requested[:, None]).squeeze(1) >= self.min_pixels
        self.evidence = torch.where(seen, self.evidence + 1, 0)
        self.lost = torch.where(seen, 0, self.lost + 1)
        self.found = (self.found | (self.evidence >= 3)) & (self.lost < 30)
        switch = (self.tick % self.switch_steps == 0) & ~self.found
        candidates = (self.cursor[:, None] + torch.arange(1, 7, device=rgb.device)) % 6
        available = counts.gather(1, candidates) >= self.min_pixels
        available &= candidates != requested[:, None]
        rank = torch.where(available, torch.arange(6, device=rgb.device), 7).argmin(1)
        next_color = candidates.gather(1, rank[:, None]).squeeze(1)
        next_color = torch.where(available.any(1), next_color, (self.cursor + 1) % 6)
        self.selected = torch.where(switch, next_color, self.selected)
        self.cursor = torch.where(switch, next_color, self.cursor)
        self.tick += 1
        return torch.where(self.found, requested, self.selected)
