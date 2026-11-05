"""Upper-model-only planar letter intervention with an explicit identity control.

Ground-truth geometry is used by this benchmark image constructor, never supplied
as a model input. The lower CNN continues to receive the unchanged native image.
This is not a full-scene rendering-generalization claim.
"""
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def textures(directory):
    from pathlib import Path
    original=[np.asarray(Image.open(Path(directory)/f'texture{i}.png').convert('RGB')) for i in (2,1,4,6,5,3)]
    backgrounds=[np.median(x.reshape(-1,3),axis=0).astype(np.uint8) for x in original]
    masks=[]
    for x,b in zip(original,backgrounds):
        # The outer black texture border is not part of a letter.
        m=np.max(np.abs(x.astype(float)-b),axis=-1)>45
        m[:3]=False;m[-3:]=False;m[:,:3]=False;m[:,-3:]=False
        masks.append(m)
    def make(permutation,masked=False):
        result=[]
        for face,symbol_source in enumerate(permutation):
            x=original[face].copy();x[masks[face]]=backgrounds[face]
            if not masked:x[masks[symbol_source]]=original[symbol_source][masks[symbol_source]]
            result.append(x)
        return result
    return make,backgrounds


def render(rgb,pose,hand_depth,geometry,texture_set,backgrounds,supersampling=4):
    """Return an upper-model RGB copy, touched mask and geometric diagnostics."""
    h,w=rgb.shape[:2]
    if supersampling>1:
        import copy
        scale=supersampling;g=copy.deepcopy(geometry)
        K=np.asarray(g['camera_intrinsics'],dtype=float)
        K[:2,:]*=scale
        K[0,2]+=(scale-1)/2;K[1,2]+=(scale-1)/2
        g['camera_intrinsics']=K.tolist()
        high=np.repeat(np.repeat(rgb,scale,axis=0),scale,axis=1)
        depth=np.repeat(np.repeat(hand_depth,scale,axis=0),scale,axis=1)
        out,mask,diagnostics=render(high,pose,depth,g,texture_set,backgrounds,supersampling=1)
        out=out.reshape(h,scale,w,scale,3).mean((1,3)).round().astype(np.uint8)
        mask=mask.reshape(h,scale,w,scale).any((1,3))
        diagnostics['supersampling']=scale
        diagnostics['changed_pixels']=int(np.any(out!=rgb,axis=-1).sum())
        return out,mask,diagnostics
    K=np.array(geometry['camera_intrinsics']);Rc=Rotation.from_quat(geometry['camera_quaternion_xyzw_ros']).as_matrix()
    cam=np.array(geometry['camera_position_env']);R=Rotation.from_quat(pose[3:]).as_matrix();pos=np.array(pose[:3])
    yy,xx=np.indices((h,w));ray=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    direction=ray@Rc.T@R;origin=(cam-pos)@R
    nearest=np.full((h,w),np.inf);face_id=np.full((h,w),-1);uvmap=np.zeros((h,w,2))
    for face in geometry['faces']:
        pts=np.array(face['points_body']);uv=np.array(face['uv']);normal=np.array(face['normal_body'])
        # Exclude backfaces. Ray parameter is camera-frame depth because ray.z=1.
        denom=direction@normal
        t=np.divide((pts[0]-origin)@normal,denom,out=np.full((h,w),np.inf),where=np.abs(denom)>1e-9)
        local=origin+t[...,None]*direction-pts[0]
        basis=np.stack([pts[1]-pts[0],pts[3]-pts[0]],axis=1)
        ab=local@np.linalg.pinv(basis).T
        valid=(denom<0)&(t>0)&(ab[...,0]>=0)&(ab[...,0]<=1)&(ab[...,1]>=0)&(ab[...,1]<=1)&(t<nearest)
        texuv=uv[0]+ab[...,0:1]*(uv[1]-uv[0])+ab[...,1:2]*(uv[3]-uv[0])
        nearest[valid]=t[valid];face_id[valid]=face['goal_index'];uvmap[valid]=texuv[valid]
    # Preserve the actual hand silhouette; no drawing through occluding fingers.
    visible=(face_id>=0)&(~np.isnan(hand_depth))&(hand_depth>=nearest-.001)
    visible &= np.max(rgb,axis=-1)>5
    out=rgb.copy();counts={}
    for goal,tex in enumerate(texture_set):
        mask=visible&(face_id==goal);counts[str(goal)]=int(mask.sum())
        if not mask.any():continue
        uv=uvmap[mask];tx=np.clip(uv[:,0]*(tex.shape[1]-1),0,tex.shape[1]-1);ty=np.clip((1-uv[:,1])*(tex.shape[0]-1),0,tex.shape[0]-1)
        x0=np.floor(tx).astype(int);y0=np.floor(ty).astype(int);x1=np.minimum(x0+1,tex.shape[1]-1);y1=np.minimum(y0+1,tex.shape[0]-1)
        wx=(tx-x0)[:,None];wy=(ty-y0)[:,None]
        sampled=(1-wy)*((1-wx)*tex[y0,x0]+wx*tex[y0,x1])+wy*((1-wx)*tex[y1,x0]+wx*tex[y1,x1])
        # One scalar brightness estimate from the original colored face pixels.
        pix=rgb[mask].astype(float);base=np.array(backgrounds[goal],float)
        chroma=np.max(pix,axis=1)-np.min(pix,axis=1)
        selection=chroma>25
        gain=float(np.median(pix[selection].max(1)/max(base.max(),1))) if selection.any() else 1.
        gain=np.clip(gain,.25,1.5)
        out[mask]=np.clip(sampled*gain,0,255).round().astype(np.uint8)
    return out,visible,dict(face_pixel_counts=counts,changed_pixels=int(np.any(out!=rgb,axis=-1).sum()))
