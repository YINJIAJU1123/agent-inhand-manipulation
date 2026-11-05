"""Six fixed motor orientations shared by the four object-specific students.

These are physical skill latents. Visible symbols are grounded to these skills
by the upper model; changing surface symbols never changes this bank.
"""
import hashlib
import math
from pathlib import Path
import torch

S=math.sqrt(.5)
NORMALS=((-1,0,0),(0,0,-1),(0,1,0),(0,-1,0),(0,0,1),(1,0,0))
QUATERNIONS_XYZW=((0,S,0,S),(1,0,0,0),(S,0,0,S),(-S,0,0,S),(0,0,0,1),(0,-S,0,S))
AXES=('negative X','negative Z','positive Y','negative Y','positive Z','positive X')
PROMPTS=tuple(f'Reorient the object with its {axis} surface facing upward.' for axis in AXES)

def load_bank(path,device):
    data=torch.load(path,map_location='cpu',weights_only=True)
    vectors=data['vectors']
    if vectors.shape!=(6,768) or not torch.isfinite(vectors).all():raise ValueError('Bad goal vectors')
    if tuple(data['prompts'])!=PROMPTS:raise ValueError('Unexpected fixed motor prompt order')
    torch.testing.assert_close(vectors.norm(dim=-1),torch.ones(6),atol=1e-5,rtol=0)
    q=torch.tensor(QUATERNIONS_XYZW,dtype=torch.float32)
    n=torch.tensor(NORMALS,dtype=torch.float32)
    c=2*torch.cross(q[:,:3],n,dim=-1)
    rotated=n+q[:,3:]*c+torch.cross(q[:,:3],c,dim=-1)
    torch.testing.assert_close(rotated,torch.tensor([0.,0.,1.]).expand_as(rotated),atol=1e-6,rtol=0)
    return vectors.to(device),q.to(device),dict(normals_body=NORMALS,
        target_quaternions_world_xyzw=QUATERNIONS_XYZW,cache_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        mapping_status='physical motor orientations; no visible-label mapping encoded')
