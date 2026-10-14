"""Generate physical-scale, non-colliding USD face decals for the benchmark."""
import math
from pathlib import Path

def polygon_mesh(name, polygons, scale, z, material):
    points=[]; indices=[]; counts=[]
    for poly in polygons:
        start=len(points); points += [(x*scale,y*scale,z) for x,y in poly]
        # Fan triangulation works for these star-shaped polygons; cross uses rectangles.
        center=len(points); points.append((sum(x for x,y in poly)/len(poly)*scale,sum(y for x,y in poly)/len(poly)*scale,z))
        for i in range(len(poly)):
            indices += [center,start+i,start+(i+1)%len(poly)]; counts.append(3)
    return f'''    def Mesh "{name}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
        point3f[] points = {points!r}
        int[] faceVertexCounts = {counts!r}
        int[] faceVertexIndices = {indices!r}
        uniform token subdivisionScheme = "none"
        bool doubleSided = true
        rel material:binding = </Symbol/{material}>
    }}\n'''

def main():
    circle=[(math.cos(i*2*math.pi/32),math.sin(i*2*math.pi/32)) for i in range(32)]
    star=[((1 if i%2==0 else .43)*math.cos(math.pi/2+i*math.pi/5), (1 if i%2==0 else .43)*math.sin(math.pi/2+i*math.pi/5)) for i in range(10)]
    shapes={'star':[star], 'circle':[circle], 'triangle':[[(0,1),(-.9,-.7),(.9,-.7)]],
            'cross':[[(-.25,-1),(.25,-1),(.25,1),(-.25,1)],[(-1,-.25),(1,-.25),(1,.25),(-1,.25)]],
            'diamond':[[(0,1),(-.8,0),(0,-1),(.8,0)]],
            'bars':[[(-.85,-.65),(.85,-.65),(.85,-.2),(-.85,-.2)],[(-.85,.2),(.85,.2),(.85,.65),(-.85,.65)]]}
    root=Path(__file__).resolve().parents[2]/'assets/usd/semantic_symbols'
    root.mkdir(parents=True,exist_ok=True)
    for name,polys in shapes.items():
        s='#usda 1.0\n(defaultPrim = "Symbol"\n metersPerUnit = 1\n upAxis = "Z")\ndef Xform "Symbol" {\n'
        for mat,color in [('White',(1,1,1)),('Black',(.01,.01,.01))]:
            s+=f'''    def Material "{mat}" {{
        token outputs:surface.connect = </Symbol/{mat}/Shader.outputs:surface>
        def Shader "Shader" {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = {color}
            float inputs:roughness = 1
            token outputs:surface
        }}
    }}\n'''
        s+=polygon_mesh('Outline',polys,.009,0,'White')
        s+=polygon_mesh('Ink',polys,.0075,.00012,'Black')
        (root/(name+'.usda')).write_text(s+'}\n')
if __name__=='__main__': main()
