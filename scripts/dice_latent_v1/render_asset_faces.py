"""CPU orthographic textured-mesh renders for auditing the six body-frame faces."""
import argparse,json
from pathlib import Path
import numpy as np,trimesh
from PIL import Image,ImageDraw

DIRECTIONS=[('+X',(1,0,0)),('-X',(-1,0,0)),('+Y',(0,1,0)),('-Y',(0,-1,0)),('+Z',(0,0,1)),('-Z',(0,0,-1))]

def load_meshes(asset):
    """Use USD coordinates for physics assets, avoiding GLB axis conversion."""
    if asset.suffix.lower() not in ('.usd', '.usda', '.usdc'):
        scene=trimesh.load(asset,force='scene',process=False);meshes=[]
        for node in scene.graph.nodes_geometry:
            transform,name=scene.graph[node];mesh=scene.geometry[name].copy()
            mesh.apply_transform(transform);meshes.append(mesh)
        return meshes
    from pxr import Usd,UsdGeom,UsdShade
    stage=Usd.Stage.Open(str(asset.resolve()));meshes=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or '/Visual/' not in str(prim.GetPath()):
            continue
        geom=UsdGeom.Mesh(prim)
        counts=np.asarray(geom.GetFaceVertexCountsAttr().Get())
        if not np.all(counts==3):raise ValueError('Face audit requires triangular visual mesh')
        points=np.asarray(geom.GetPointsAttr().Get(),dtype=float)
        matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        points=(np.column_stack((points,np.ones(len(points))))@matrix)[:,:3]
        faces=np.asarray(geom.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        st=UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        if st.GetInterpolation()!='vertex':raise ValueError('Expected vertex UVs')
        uv=np.asarray(st.ComputeFlattened())
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        textures=[]
        for shader_prim in Usd.PrimRange(material.GetPrim()):
            if shader_prim.IsA(UsdShade.Shader):
                value=UsdShade.Shader(shader_prim).GetInput('file').Get()
                if value is not None:textures.append(value.resolvedPath)
        if len(textures)!=1 or not textures[0]:raise ValueError('Expected one resolved color texture')
        mesh=trimesh.Trimesh(vertices=points,faces=faces,process=False)
        mesh.visual=trimesh.visual.TextureVisuals(uv=uv,image=Image.open(textures[0]).convert('RGB'))
        meshes.append(mesh)
    if not meshes:raise ValueError('No visual mesh found in USD')
    return meshes


def render_meshes(meshes,direction,size=384):
    eye=np.asarray(direction,dtype=float)
    up=np.array([0.,0.,1.]) if abs(eye[2])<.9 else np.array([0.,1.,0.])
    right=np.cross(up,eye);right/=np.linalg.norm(right);up=np.cross(eye,right)
    camera=np.stack((right,up,eye),axis=1)
    bounds=np.concatenate([np.asarray(m.vertices) for m in meshes])
    extent=np.abs(bounds).max()*1.18
    out=np.full((size,size,3),235,dtype=np.uint8);zbuf=np.full((size,size),-np.inf)
    for mesh in meshes:
        xyz=np.asarray(mesh.vertices)@camera
        screen=np.stack(((xyz[:,0]/extent+1)*(size-1)/2,(1-xyz[:,1]/extent)*(size-1)/2),-1)
        uv=np.asarray(mesh.visual.uv) if hasattr(mesh.visual,'uv') else None
        material=mesh.visual.material if hasattr(mesh.visual,'material') else None
        texture=getattr(material,'baseColorTexture',None)
        if texture is None:texture=getattr(material,'image',None)
        tex=np.array(texture.convert('RGB')) if texture is not None else None
        for fi,tri in enumerate(mesh.faces):
            normal=mesh.face_normals[fi]
            if np.dot(normal,eye)<=0:continue
            p=screen[tri];z=xyz[tri,2]
            x0,y0=np.maximum(np.floor(p.min(0)).astype(int),0);x1,y1=np.minimum(np.ceil(p.max(0)).astype(int),size-1)
            if x0>x1 or y0>y1:continue
            den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
            if abs(den)<1e-9:continue
            xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
            a=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den
            b=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den;c=1-a-b
            depth=a*z[0]+b*z[1]+c*z[2]
            use=(a>=-1e-6)&(b>=-1e-6)&(c>=-1e-6)&(depth>zbuf[y0:y1+1,x0:x1+1])
            if not use.any():continue
            if tex is not None and uv is not None:
                coords=a[...,None]*uv[tri[0]]+b[...,None]*uv[tri[1]]+c[...,None]*uv[tri[2]]
                tx=np.clip(np.rint(coords[...,0]*(tex.shape[1]-1)),0,tex.shape[1]-1).astype(int)
                ty=np.clip(np.rint((1-coords[...,1])*(tex.shape[0]-1)),0,tex.shape[0]-1).astype(int)
                color=tex[ty,tx]
            else:color=np.full((*a.shape,3),180,dtype=np.uint8)
            shade=.6+.4*max(0,float(np.dot(normal,eye)))
            out[y0:y1+1,x0:x1+1][use]=(color[use]*shade).astype(np.uint8)
            zbuf[y0:y1+1,x0:x1+1][use]=depth[use]
    return Image.fromarray(out)


def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 meshes=load_meshes(a.asset)
 sheet=Image.new('RGB',(1152,824),'white')
 for i,(name,d) in enumerate(DIRECTIONS):
  im=render_meshes(meshes,d);im.save(a.output/f'face_{name[0]}{name[1]}.png')
  x=(i%3)*384;y=(i//3)*412;sheet.paste(im,(x,y+28));ImageDraw.Draw(sheet).text((x+170,y+5),name,fill='black')
 sheet.save(a.output/'face_audit.jpg',quality=90)
 bounds=np.concatenate([np.asarray(m.vertices) for m in meshes])
 print(json.dumps({'vertices':sum(len(m.vertices) for m in meshes),'bounds':[bounds.min(0).tolist(),bounds.max(0).tolist()],'asset':str(a.asset),'output':str(a.output/'face_audit.jpg')},indent=2))
if __name__=='__main__':main()
