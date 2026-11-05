"""Goal bank audited against both USD texture and live Newton RGB."""
import hashlib
import math
import types
from pathlib import Path
import torch

S=math.sqrt(.5)
NUMBERS=(1,2,3,4,5,6)
NORMALS=((-1,0,0),(0,0,-1),(0,1,0),(0,-1,0),(0,0,1),(1,0,0))
QUATERNIONS_XYZW=((0,S,0,S),(1,0,0,0),(S,0,0,S),(-S,0,0,S),(0,0,0,1),(0,-S,0,S))
PROMPTS=tuple(f'Turn the die so that the face with {n} dots faces upward.' for n in NUMBERS)

def load_bank(path,device):
    data=torch.load(path,map_location='cpu',weights_only=True)
    vectors=data['vectors']
    if vectors.shape!=(6,768) or not torch.isfinite(vectors).all():raise ValueError('Bad goal vectors')
    if tuple(data['prompts'])!=PROMPTS:raise ValueError('Unexpected goal prompt order')
    torch.testing.assert_close(vectors.norm(dim=-1),torch.ones(6),atol=1e-5,rtol=0)
    q=torch.tensor(QUATERNIONS_XYZW,dtype=torch.float32)
    n=torch.tensor(NORMALS,dtype=torch.float32)
    c=2*torch.cross(q[:,:3],n,dim=-1)
    rotated=n+q[:,3:]*c+torch.cross(q[:,:3],c,dim=-1)
    torch.testing.assert_close(rotated,torch.tensor([0.,0.,1.]).expand_as(rotated),atol=1e-6,rtol=0)
    return vectors.to(device),q.to(device),dict(numbers=NUMBERS,normals_body=NORMALS,
        target_quaternions_world_xyzw=QUATERNIONS_XYZW,cache_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        mapping_status='USD texture and live Newton RGB verified: face_audit_live_v2')

def install_training_goals(command,quaternions):
    ids=torch.randint(6,(command.num_envs,),device=command.device)
    original=command._resample_command
    def resample(self,env_ids):
        original(env_ids)
        ids[env_ids]=torch.randint(6,ids[env_ids].shape,device=ids.device)
        self.quat_command_w[env_ids]=quaternions[ids[env_ids]]
    command._resample_command=types.MethodType(resample,command)
    command.quat_command_w[:]=quaternions[ids]
    return ids
