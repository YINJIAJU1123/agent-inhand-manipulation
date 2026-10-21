"""Completion evidence from frozen policy memory and optional measured RGB-D.

The forward interface accepts sensor tensors only. Exact pose is a training
target, never an input. All six candidate goals remain separate queries.
"""
import torch
from torch import nn
from torch.nn import functional as F

TOLERANCES = (.08, .12, .16, .22, .32, .5, 1.)


def rotation_matrix(six):
    """Continuous six-dimensional orientation prediction, represented by columns."""
    first = F.normalize(six[..., :3], dim=-1, eps=1e-6)
    raw = six[..., 3:]
    second = F.normalize(raw - (raw * first).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def quaternion_matrix(quaternion):
    """Training/evaluation conversion for the simulator's scalar-first labels."""
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(
                            *quaternion.shape[:-1], 3, 3)


class DenseCompletion(nn.Module):
    def __init__(self, dense=True):
        super().__init__()
        self.dense = dense
        self.register_buffer('mean', torch.zeros(731))
        self.register_buffer('scale', torch.ones(731))
        # Mean over all six hypothetical goals is independent of the chosen goal.
        self.context = nn.Sequential(nn.Linear(731, 128), nn.LayerNorm(128), nn.SiLU())
        if dense:
            layers = []
            for n_in, n_out in ((4, 16), (16, 32), (32, 48), (48, 64)):
                layers.extend((nn.Conv2d(n_in, n_out, 3, stride=2, padding=1),
                               nn.GroupNorm(4, n_out), nn.SiLU()))
            layers.extend((nn.Flatten(), nn.Linear(64*6*6, 128), nn.SiLU()))
            self.visual = nn.Sequential(*layers)
        self.state = nn.Sequential(nn.Linear(128 + (133 if dense else 0), 128),
                                   nn.LayerNorm(128), nn.SiLU())
        self.pose = nn.Linear(128, 6)
        with torch.no_grad():
            self.pose.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.query = nn.Sequential(nn.Linear(731 + 128, 192), nn.LayerNorm(192),
                                   nn.SiLU(), nn.Dropout(.1), nn.Linear(192, 96), nn.SiLU())
        self.ordinal = nn.Linear(96, len(TOLERANCES))
        self.error = nn.Linear(96, 1)

    def forward(self, x, rgb, depth, box, found):
        if x.ndim != 3 or x.shape[1:] != (6, 731):
            raise ValueError('Expected all six frozen candidate features')
        normalized = ((x-self.mean)/self.scale).clamp(-8, 8)
        context = self.context(normalized.mean(1))
        if self.dense:
            if rgb.shape[1:] != (3, 96, 96) or depth.shape[1:] != (1, 96, 96):
                raise ValueError('Expected aligned 96x96 RGB-D sensors')
            pixels = rgb.float()/255.
            visual = self.visual(torch.cat((pixels, depth.float()), 1))
            context = torch.cat((context, visual, box.float(), found.float()[:, None]), -1)
        state = self.state(context)
        query = self.query(torch.cat((normalized, state[:, None].expand(-1, 6, -1)), -1))
        return dict(ordinal=self.ordinal(query), log_error=self.error(query).squeeze(-1),
                    rotation=rotation_matrix(self.pose(state)))


def load_model(path, device='cpu'):
    pack = torch.load(path, map_location='cpu', weights_only=True)
    model = DenseCompletion(dense=pack['dense']).to(device)
    model.load_state_dict(pack['model'])
    return model.eval().requires_grad_(False), pack
