"""Six fixed latent skills for the exact released RSLCube asset.

The normals below are in the rigid-body frame, after the mesh's +90 degree
X rotation. They must not be replaced by the historical synthetic cube mapping.
The legacy latent channel 'cyan' denotes the pale-blue H texture in this asset.
"""

import hashlib
import math
from pathlib import Path
import types

import torch

ASSET_SHA256 = '93d60bde1d00cf5398b5d31e2016663e793d8bb4cdee817421e324f47fccb4a8'
COLORS = ('red', 'green', 'blue', 'yellow', 'magenta', 'cyan')
SYMBOLS = ('T', 'E', 'R', 'L', 'S', 'H')
TEXTURES = (2, 1, 4, 6, 5, 3)
NORMALS = ((0, -1, 0), (-1, 0, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0))
S = math.sqrt(.5)
QUATERNIONS_XYZW = ((-S, 0, 0, S), (0, S, 0, S), (1, 0, 0, 0),
                    (0, 0, 0, 1), (S, 0, 0, S), (0, -S, 0, S))


def load_bank(cache_path, original_asset_path, device):
    asset = Path(original_asset_path)
    if hashlib.sha256(asset.read_bytes()).hexdigest() != ASSET_SHA256:
        raise ValueError('Semantic mapping is audited only for the original released exact-box asset')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    vectors = torch.stack([cache['embeddings'][f'show the {color} marker'] for color in COLORS]).float()
    if vectors.shape != (6, 768) or not torch.isfinite(vectors).all():
        raise ValueError('Expected six finite 768D cached SigLIP2 features')
    # The cache already stores normalized embeddings. Preserve its values.
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(6), atol=1e-5, rtol=0)
    quaternions = torch.tensor(QUATERNIONS_XYZW, dtype=torch.float32)
    normals = torch.tensor(NORMALS, dtype=torch.float32)
    qxyz = quaternions[:, :3]
    cross = 2 * torch.cross(qxyz, normals, dim=-1)
    rotated = normals + quaternions[:, 3:] * cross + torch.cross(qxyz, cross, dim=-1)
    torch.testing.assert_close(rotated, torch.tensor([0., 0., 1.]).expand_as(rotated), atol=1e-6, rtol=0)
    manifest = {
        'asset_sha256': ASSET_SHA256, 'cache_sha256': hashlib.sha256(Path(cache_path).read_bytes()).hexdigest(),
        'latent_source': cache.get('model_sha256'), 'colors': COLORS, 'symbols': SYMBOLS,
        'texture_numbers': TEXTURES, 'normals_body': NORMALS,
        'target_quaternions_world_xyzw': QUATERNIONS_XYZW,
        'cyan_channel_visual_name': 'pale blue H', 'magenta_channel_visual_name': 'purple S',
        'mapping_scope': 'fixed body-frame mapping in the exact released asset, not randomized face layouts',
    }
    return vectors.to(device), quaternions.to(device), manifest


def install_training_goals(command, quaternions):
    """Explicit training-only replacement of the goal distribution.

    Preserve the original resampling lifecycle and counters. This modification
    must be recorded separately from the unchanged physical environment binding.
    """
    ids = torch.randint(6, (command.num_envs,), device=command.device)
    original = command._resample_command

    def resample(self, env_ids):
        original(env_ids)
        selected = ids[env_ids]
        ids[env_ids] = torch.randint(6, selected.shape, device=ids.device)
        self.quat_command_w[env_ids] = quaternions[ids[env_ids]]

    command._resample_command = types.MethodType(resample, command)
    command.quat_command_w[:] = quaternions[ids]
    return ids
