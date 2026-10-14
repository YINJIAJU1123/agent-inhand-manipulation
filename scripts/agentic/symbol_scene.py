"""Benchmark renderer only; layouts and object poses never enter agent inputs."""
from pathlib import Path
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_mul

SYMBOLS=('star','circle','triangle','cross','diamond','bars')

class SymbolScene:
    def __init__(self, raw):
        self.raw=raw
        root=Path(__file__).resolve().parents[2]/'assets/usd/semantic_symbols'
        self.markers=VisualizationMarkers(VisualizationMarkersCfg(
            prim_path='/Visuals/semantic_symbols',
            markers={name:sim_utils.UsdFileCfg(usd_path=str(root/(name+'.usda'))) for name in SYMBOLS}))
        self.layout=torch.arange(6,device=raw.device).repeat(raw.num_envs,1)

    def set_layout(self, layout):
        value=torch.as_tensor(layout,device=self.raw.device,dtype=torch.long)
        if value.shape != self.layout.shape or not torch.all(value.sort(-1).values==torch.arange(6,device=value.device)):
            raise ValueError('each environment needs a permutation of all six symbols')
        self.layout=value.clone()

    def update(self):
        raw=self.raw
        rot=raw.object_rot
        normals=raw._face_normals.unsqueeze(0).expand(raw.num_envs,-1,-1)
        offset=quat_apply(rot[:,None].expand(-1,6,-1).reshape(-1,4),(normals*.03715).reshape(-1,3)).reshape(raw.num_envs,6,3)
        pos=raw.object_pos[:,None]+raw.scene.env_origins[:,None]+offset
        orientation=quat_mul(rot[:,None].expand(-1,6,-1),raw._face_marker_rots[None].expand(raw.num_envs,-1,-1))
        self.markers.visualize(pos.reshape(-1,3),orientation.reshape(-1,4),marker_indices=self.layout.reshape(-1))
