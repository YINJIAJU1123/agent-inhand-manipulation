"""Offline audit of authored collision meshes at measured rigid-body poses.

The convex hulls are geometry diagnostics, not a replacement for PhysX's cooked
collision shapes. No information from this audit is passed to the policy.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def collision_geometry(raw):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = raw.sim.stage
    xf = UsdGeom.XformCache(0)
    names = list(raw.hand.body_names)
    pieces = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'), Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != 'Mesh':
            continue
        parent = prim
        collision = False
        body = None
        while parent and str(parent.GetPath()) != '/World/envs/env_0':
            collision |= parent.HasAPI(UsdPhysics.CollisionAPI)
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                body = parent
                break
            parent = parent.GetParent()
        if not collision or body is None:
            continue
        path = str(body.GetPath())
        kind = 'object' if '/object' in path else 'hand'
        if kind == 'hand' and body.GetName() not in names:
            continue
        points = np.asarray(prim.GetAttribute('points').Get(), dtype=float)
        if len(points) < 4:
            continue
        matrix = np.asarray(xf.GetLocalToWorldTransform(prim) * xf.GetLocalToWorldTransform(body).GetInverse())
        local = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
        # Hull vertices reduce computation without changing the convex hull.
        local = local[ConvexHull(local).vertices]
        pieces.append(dict(kind=kind, name=str(prim.GetPath()), body=body.GetName(),
                           index=names.index(body.GetName()) if kind=='hand' else 0, points=local))
    if not any(p['kind']=='object' for p in pieces):
        raise RuntimeError('No object collision mesh found')
    return pieces


def rotation(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def audit(raw):
    pieces = collision_geometry(raw)
    obj = raw.object.data.root_state_w.cpu().numpy()
    hands = raw.hand.data.body_state_w.cpu().numpy()
    result=[]
    for i in range(len(obj)):
        clouds=[]
        for p in pieces:
            state=obj[i] if p['kind']=='object' else hands[i,p['index']]
            cloud=p['points']@rotation(state[3:7]).T+state[:3]
            clouds.append((p,cloud))
        object_points=np.concatenate([c for p,c in clouds if p['kind']=='object'])
        oe=ConvexHull(object_points).equations
        overlaps=[]
        for p,c in clouds:
            if p['kind']!='hand' or np.any(c.max(0)<object_points.min(0)) or np.any(c.min(0)>object_points.max(0)):
                continue
            he=ConvexHull(c).equations
            eq=np.concatenate([oe,he]);norm=np.linalg.norm(eq[:,:3],axis=1)
            # Largest inscribed sphere in the intersection, a lower bound on
            # overlap thickness. Positive radius confirms convex-hull overlap.
            fit=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],norm],b_ub=-eq[:,3],
                        bounds=[(None,None)]*3+[(0,None)],method='highs')
            if fit.success and fit.x[3]>1e-6:
                overlaps.append(dict(body=p['body'],mesh=p['name'],intersection_radius_m=float(fit.x[3])))
        result.append(dict(case=i,object_bounds=[object_points.min(0).tolist(),object_points.max(0).tolist()],
                           overlaps=overlaps))
    return dict(method='Convex hull of authored collision meshes transformed by measured body poses; cooked PhysX decomposition may differ',
                mesh_count=len(pieces),object_local_bounds=[np.concatenate([p['points'] for p in pieces if p['kind']=='object']).min(0).tolist(),np.concatenate([p['points'] for p in pieces if p['kind']=='object']).max(0).tolist()],cases=result)
