"""Append decal triangles to one existing visual mesh, keeping shape count fixed.

Newton randomizes material values per imported shape. Separate decal meshes
would change the random draw count despite having no collisions. This variant
preserves all original mesh primitive paths and changes only visual mesh data.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from pxr import Usd,UsdGeom,UsdShade,Gf,Sdf,Vt
from build_marked_assets import native_mesh,decal,records,sha,NORMALS,LETTERS,COLORS,COLOR_NAMES,PERMUTATION

def atlas(stage,prim,letters,path):
    mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader=mat.ComputeSurfaceSource()[0];color=shader.GetInput('diffuseColor')
    connected=color.GetConnectedSource()
    if connected:
        asset=UsdShade.Shader(connected[0].GetPrim()).GetInput('file').Get()
        source=Path(asset.resolvedPath or asset.path)
        if not source.exists():raise ValueError('Unresolved original diffuse texture')
        original=Image.open(source).convert('RGB').resize((1024,1024))
        source_record=dict(path=str(source.resolve()),sha256=sha(source))
    else:
        linear=np.array(color.Get(),dtype=float)
        srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
        original=Image.new('RGB',(1024,1024),tuple(np.rint(srgb.clip(0,1)*255).astype(int)))
        source_record=dict(diffuse_linear=linear.tolist())
    canvas=Image.new('RGB',(2048,1536));canvas.paste(original,(0,0))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',340)
    for i,(color,letter) in enumerate(zip(COLORS,letters)):
        x=1024+(i%2)*512;y=(i//2)*512
        draw.rectangle((x,y,x+511,y+511),fill=tuple(round(c*255) for c in color))
        draw.text((x+256,y+256),letter,font=font,anchor='mm',fill='white',stroke_width=3,stroke_fill=(20,20,20))
    canvas.save(path)
    return source_record

def merge(stage,prim_path,geometries,texture_path):
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(prim_path));old_points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    old_faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    old_normals=np.asarray(mesh.GetNormalsAttr().Get(),dtype=float)
    if mesh.GetNormalsInterpolation()!='vertex' or old_normals.shape!=old_points.shape:
        raise ValueError('Unexpected source normal layout')
    st=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    if st and st.HasValue():
        if st.GetInterpolation()!='vertex':raise ValueError('Unexpected original UV layout')
        old_uv=np.asarray(st.ComputeFlattened(),dtype=float)
    else:old_uv=np.full((len(old_points),2),.5)
    old_uv=old_uv*np.array([.5,2/3])+np.array([0.,1/3])
    inverse=UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()).GetInverse()
    points=[old_points];faces=[old_faces];normals=[old_normals];uvs=[old_uv];offset=len(old_points)
    for i,(world,tri,uv) in enumerate(geometries):
        local=np.asarray([inverse.Transform(Gf.Vec3d(*map(float,v))) for v in world])
        points.append(local);faces.append(tri+offset);offset+=len(local)
        normals.append(trimesh.Trimesh(local,tri,process=False).vertex_normals)
        uvs.append(uv*np.array([.25,1/3])+np.array([.5+(i%2)*.25,1-(i//2+1)/3]))
    points=np.concatenate(points);faces=np.concatenate(faces);uvs=np.concatenate(uvs)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.concatenate(normals).astype(np.float32)))
    mesh.SetNormalsInterpolation('vertex');mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
    uvvar=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
    uvvar.Set(Vt.Vec2fArray.FromNumpy(uvs.astype(np.float32)))
    mpath='/Object/Visual/SemanticAtlasMaterial';mat=UsdShade.Material.Define(stage,mpath)
    surface=UsdShade.Shader.Define(stage,mpath+'/Surface');surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
    reader=UsdShade.Shader.Define(stage,mpath+'/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex=UsdShade.Shader.Define(stage,mpath+'/Texture');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture_path.resolve()))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    surface.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

def build(source,output):
    output.mkdir(parents=True,exist_ok=False)
    native=Usd.Stage.Open(str(source.resolve()));physical=records(native,True)
    mesh=native_mesh(native);diameter=min(.025,float(mesh.extents.min())*.36)
    geometries=[decal(mesh,n,diameter) for n in NORMALS]
    geometry_path=output/'surface_samples.npz'
    np.savez(geometry_path,**{f'{k}_{i}':a for i,g in enumerate(geometries) for k,a in zip(('points','faces','uv'),g)})
    original_meshes=[str(p.GetPath()) for p in native.Traverse() if p.IsA(UsdGeom.Mesh)]
    prim=next(p for p in native.Traverse() if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith('/Object/Visual/'))
    manifest=dict(status='development_only_requires_live_audit',source=str(source.resolve()),source_sha256=sha(source),
        nonvisual_sha256=physical,marker_diameter_m=diameter,normals_body=NORMALS,colors_by_skill=COLOR_NAMES,
        identity_letters_by_skill=LETTERS,surface_samples=dict(path=str(geometry_path.resolve()),sha256=sha(geometry_path)),
        intervention='Only glyphs differ between layouts; decal triangles are merged into an existing visual mesh to preserve shape count',
        source_mesh_paths=original_meshes,conditions={})
    for mode,letters in [('identity',LETTERS),('permuted',PERMUTATION)]:
        path=output/(mode+'.usda');stage=Usd.Stage.CreateNew(str(path.resolve()))
        stage.GetRootLayer().subLayerPaths=[str(source.resolve())];stage.SetDefaultPrim(stage.GetPrimAtPath('/Object'))
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        tex=output/(mode+'_atlas.png');source_record=atlas(native,prim,letters,tex)
        merge(stage,str(prim.GetPath()),geometries,tex);stage.GetRootLayer().Save()
        if records(stage,True)!=physical:raise ValueError('Changed nonvisual USD')
        if [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]!=original_meshes:raise ValueError('Changed mesh count/order')
        manifest['conditions'][mode]=dict(asset=str(path.resolve()),sha256=sha(path),letters_by_physical_skill=letters,
            textures=[dict(path=str(tex.resolve()),sha256=sha(tex))],original_diffuse=source_record)
    manifest['geometry_sha256']=hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for g in geometries for a in g)).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),mesh_count=len(original_meshes),status=manifest['status'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.output)
