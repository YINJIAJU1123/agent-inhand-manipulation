"""Benchmark-only letter intervention on upper-model RGB, preserving pose CNN input.

Simulator pose/depth construct the image but are never sent to a model. The
fixed permutation changes which printed symbol names each physical target.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from goals import SYMBOLS

class UpperImageIntervention:
    PERMUTATION=(1,2,3,4,5,0)
    def __init__(self,mode,geometry_path,texture_dir,project_root,runtime):
        if mode not in ('identity','permuted','letter_masked'):raise ValueError('Unknown visual intervention')
        self.mode=mode;self.geometry=json.loads(Path(geometry_path).read_text())
        if self.geometry['asset_sha256']!=runtime.goal_manifest['asset_sha256']:raise ValueError('Image geometry asset mismatch')
        code=Path(project_root)/'scripts/pose_latent_studies_v2/visual_intervention'
        sys.path.insert(0,str(code))
        from render_letters import textures,render
        self.render=render;make,self.backgrounds=textures(texture_dir)
        self.textures=make(self.PERMUTATION if mode=='permuted' else range(6),masked=mode=='letter_masked')
        self.metadata={'mode':mode,'face_to_symbol_index':list(self.PERMUTATION) if mode=='permuted' else list(range(6)),
            'scope':'Upper-model image only. Lower pose CNN retains native image. Geometry/depth are benchmark construction inputs, never model inputs.',
            'geometry_sha256':hashlib.sha256(Path(geometry_path).read_bytes()).hexdigest(),
            'renderer_sha256':hashlib.sha256((code/'render_letters.py').read_bytes()).hexdigest(),
            'textures_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(texture_dir).glob('texture*.png'))},'supersampling':4}
    def symbol(self,physical_goal):
        return SYMBOLS[self.PERMUTATION[physical_goal] if self.mode=='permuted' else physical_goal]
    def image(self,runtime,rgb):
        hand=runtime.vision._hand_depth_camera
        actual={'camera_intrinsics':hand.data.intrinsic_matrices.torch[0].cpu().numpy(),
            'camera_position_env':(hand.data.pos_w.torch[0]-runtime.raw.scene.env_origins[0]).cpu().numpy(),
            'camera_quaternion_xyzw_ros':hand.data.quat_w_ros.torch[0].cpu().numpy()}
        for name,value in actual.items():
            # q and -q are the same calibrated camera rotation.
            expected=np.asarray(self.geometry[name]);matches=np.allclose(value,expected,atol=1e-5,rtol=0)
            if name=='camera_quaternion_xyzw_ros':matches |= np.allclose(-value,expected,atol=1e-5,rtol=0)
            if not matches:raise ValueError('Visual intervention camera mismatch: '+name)
        obj=runtime.raw.scene['object']
        pose=np.concatenate([(obj.data.root_pos_w.torch[0]-runtime.raw.scene.env_origins[0]).cpu().numpy(),obj.data.root_quat_w.torch[0].cpu().numpy()])
        depth=runtime.vision.hand_depth[0,0].cpu().numpy()
        before=hashlib.sha256(rgb.tobytes()).hexdigest()
        out,mask,diag=self.render(rgb,pose,depth,self.geometry,self.textures,self.backgrounds)
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=before:raise RuntimeError('Native image was mutated')
        diag.update(native_pixel_sha256=before,upper_pixel_sha256=hashlib.sha256(out.tobytes()).hexdigest())
        return out,diag
