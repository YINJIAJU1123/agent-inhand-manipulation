"""Surface-conforming visual decals with fixed physics and permuted letters.

Decals are visual-only meshes placed 0.15 mm above the original surface. The
identity and permuted conditions share their geometry, colors and placements.
Only glyph textures differ. No screen-space or pose-dependent image editing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from goals import NORMALS

LETTERS=tuple('ABCDEF')
COLORS=((.85,.08,.08),(.06,.6,.12),(.06,.2,.85),(.92,.65,.03),(.03,.65,.72),(.65,.06,.7))
COLOR_NAMES=('red','green','blue','yellow','cyan','purple')
PERMUTATION=('D','F','B','E','A','C')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def records(stage, nonvisual=False):
    out=[]
    for prim in stage.Traverse():
        if nonvisual and str(prim.GetPath()).startswith('/Object/Visual'):continue
        out.append((str(prim.GetPath()),prim.GetTypeName(),str(prim.GetMetadata('apiSchemas'))))
        for a in sorted(prim.GetAttributes(),key=lambda x:x.GetName()):
            if a.HasAuthoredValueOpinion() or a.GetConnections():
                out.append((str(a.GetPath()),str(a.GetTypeName()),str(a.Get()),list(map(str,a.GetConnections()))))
        for r in sorted(prim.GetRelationships(),key=lambda x:x.GetName()):
            if r.GetTargets():out.append((str(r.GetPath()),list(map(str,r.GetTargets()))))
    return hashlib.sha256(json.dumps(out,sort_keys=True).encode()).hexdigest()

def native_mesh(stage):
    vertices=[];faces=[];offset=0
    cache=UsdGeom.XformCache()
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith('/Object/Visual/') or not p.IsA(UsdGeom.Mesh):continue
        m=UsdGeom.Mesh(p)
        if set(m.GetFaceVertexCountsAttr().Get())!={3}:raise ValueError('Need triangular visual mesh')
        transform=cache.GetLocalToWorldTransform(p)
        v=np.array([transform.Transform(Gf.Vec3d(*map(float,x))) for x in m.GetPointsAttr().Get()])
        f=np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    if not vertices:raise ValueError('Missing source visual geometry')
    return trimesh.Trimesh(np.concatenate(vertices),np.concatenate(faces),process=False)

def decal(mesh,normal,diameter,resolution=40):
    normal=np.asarray(normal,dtype=float)
    auxiliary=np.array([0.,0.,1.]) if abs(normal[2])<.5 else np.array([0.,1.,0.])
    u=np.cross(auxiliary,normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
    # All six markers are attached using body coordinates, independent of pose.
    center=(mesh.bounds[0]+mesh.bounds[1])*.5
    grid=np.linspace(-diameter/2,diameter/2,resolution+1)
    uu,vv=np.meshgrid(grid,grid)
    plane=center+uu.reshape(-1,1)*u+vv.reshape(-1,1)*v
    origins=plane+normal*.3
    locations,rays,triangles=mesh.ray.intersects_location(origins,np.tile(-normal,(len(origins),1)),multiple_hits=True)
    points=np.full_like(origins,np.nan);score=np.full(len(origins),-np.inf)
    # Select the first surface seen from that physical direction.
    for hit,ray in zip(locations,rays):
        value=float(hit@normal)
        if value>score[ray]:points[ray]=hit+normal*.00015;score[ray]=value
    uv=np.stack(((uu/diameter+.5).ravel(),(vv/diameter+.5).ravel()),-1)
    faces=[]
    for j in range(resolution):
        for i in range(resolution):
            ids=[j*(resolution+1)+i,j*(resolution+1)+i+1,(j+1)*(resolution+1)+i+1,(j+1)*(resolution+1)+i]
            if not np.isfinite(points[ids]).all():continue
            if np.linalg.norm(uv[ids].mean(0)-.5)>.49:continue
            if np.ptp(points[ids]@normal)>diameter*.2:continue
            faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
    if len(faces)<resolution**2*.8:raise ValueError('Insufficient continuous surface for marker')
    used=np.unique(faces);mapping=np.full(len(points),-1,dtype=int);mapping[used]=np.arange(len(used))
    return points[used],mapping[np.asarray(faces)],uv[used]

def texture(path,color,letter):
    im=Image.new('RGB',(512,512),tuple(round(c*255) for c in color))
    d=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',340)
    d.text((256,256),letter,font=font,anchor='mm',fill='white',stroke_width=3,stroke_fill=(20,20,20))
    im.save(path)

def author_marker(stage,index,geometry,texture_path):
    p=f'/Object/Visual/SemanticMarkers/Marker{index+1}'
    points,faces,uv=geometry
    mesh=UsdGeom.Mesh.Define(stage,p)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(False)
    st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    mat=UsdShade.Material.Define(stage,p+'/Material')
    surface=UsdShade.Shader.Define(stage,p+'/Material/Surface');surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.85)
    reader=UsdShade.Shader.Define(stage,p+'/Material/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex=UsdShade.Shader.Define(stage,p+'/Material/Texture');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture_path.resolve()))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    surface.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

def build(source,output):
    output.mkdir(parents=True,exist_ok=False)
    stage=Usd.Stage.Open(str(source.resolve()));physical=records(stage,True)
    mesh=native_mesh(stage)
    diameter=min(.025,float(mesh.extents.min())*.36)
    geometries=[decal(mesh,n,diameter) for n in NORMALS]
    geometry_path=output/"surface_samples.npz"
    np.savez(geometry_path,**{f"{key}_{i}":array for i,g in enumerate(geometries) for key,array in zip(("points","faces","uv"),g)})
    manifest=dict(status='development_only_requires_live_RGB_and_physics_audits',source=str(source.resolve()),
        source_sha256=sha(source),nonvisual_sha256=physical,marker_diameter_m=diameter,
        normals_body=NORMALS,colors_by_skill=COLOR_NAMES,identity_letters_by_skill=LETTERS,
        intervention='Only glyph textures permuted; surface decals are present in both conditions',
        surface_samples=dict(path=str(geometry_path.resolve()),sha256=sha(geometry_path)),conditions={})
    for name,letters in [('identity',LETTERS),('permuted',PERMUTATION)]:
        dest=output/(name+'.usda');s=Usd.Stage.CreateNew(str(dest.resolve()))
        s.GetRootLayer().subLayerPaths=[str(source.resolve())]
        s.SetDefaultPrim(s.GetPrimAtPath('/Object'));UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.SetStageMetersPerUnit(s,1.)
        images=[]
        for i,(color,letter,geometry) in enumerate(zip(COLORS,letters,geometries)):
            tex=output/f'{name}_skill{i+1}_{letter}.png';texture(tex,color,letter);images.append(dict(path=str(tex.resolve()),sha256=sha(tex)))
            author_marker(s,i,geometry,tex)
        s.GetRootLayer().Save()
        if records(s,True)!=physical:raise ValueError('Nonvisual physics changed')
        manifest['conditions'][name]=dict(asset=str(dest.resolve()),sha256=sha(dest),letters_by_physical_skill=letters,textures=images)
    manifest['geometry_sha256']=hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for g in geometries for a in g)).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.output)
