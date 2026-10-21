"""Versioned simulation reset correction using authored collision geometry.

Only initial placement changes. No pose labels enter the policy, and no failed
case is removed. Baseline and candidate must use the same reset version when
comparing controllers. Convex hulls conservatively approximate cooked shapes.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from geometry_v2 import collision_geometry,rotation

def minimum_vertical_offsets(raw, step=.0005, maximum=.02):
    pieces=collision_geometry(raw)
    obj=raw.object.data.root_state_w.cpu().numpy();hand=raw.hand.data.body_state_w.cpu().numpy()
    offsets=[];diagnostics=[]
    for i in range(len(obj)):
        clouds=[]
        for p in pieces:
            state=obj[i] if p['kind']=='object' else hand[i,p['index']]
            clouds.append((p,p['points']@rotation(state[3:7]).T+state[:3]))
        points=np.concatenate([v for p,v in clouds if p['kind']=='object']);oe=ConvexHull(points).equations
        low,high=points.min(0),points.max(0)
        colliders=[(p['body'],v.min(0),v.max(0),ConvexHull(v).equations) for p,v in clouds if p['kind']=='hand']
        def penetration(lift):
            shift=np.array([0,0,lift]);moving=oe.copy();moving[:,3]-=moving[:,2]*lift
            worst=0.
            for _,cl,ch,he in colliders:
                if np.any(ch<low+shift) or np.any(cl>high+shift):continue
                eq=np.concatenate((moving,he));norm=np.linalg.norm(eq[:,:3],axis=1)
                fit=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],norm],b_ub=-eq[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
                if fit.success:worst=max(worst,float(fit.x[3]))
            return worst
        before=penetration(0.);lift=0.
        if before>1e-6:
            for lift in np.arange(step,maximum+step/2,step):
                if penetration(lift)<=1e-6:break
            else:raise RuntimeError(f'Case {i}: no collision-free upward placement within {maximum}m; retain as setup failure')
            # One additional grid interval gives clearance from numerical contact.
            lift+=step
            if lift>maximum or penetration(lift)>1e-6:raise RuntimeError(f'Case {i}: clearance check failed')
        offsets.append(float(lift));diagnostics.append(dict(case=i,initial_intersection_radius_m=before,lift_m=float(lift),after_intersection_radius_m=penetration(lift)))
    return offsets,dict(version='collision-clearance-grid-v1',step_m=step,maximum_m=maximum,cases=diagnostics,
                        interpretation='Reset geometry correction, not learned control; all initial cases retained, no simulation step removed')
