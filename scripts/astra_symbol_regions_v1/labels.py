"""Privileged renderer supervision. Never imported by deployed inference."""
import torch
from latent_data_v4 import visibility_labels

def readable(raw,camera,rgb,without,layout):
    labels=visibility_labels(raw,camera)
    diff=(rgb.float()-without.float()).abs().mean(-1)>20
    changed=diff & ((rgb.max(-1).values-rgb.min(-1).values)<45) & ((rgb.max(-1).values<80)|(rgb.min(-1).values>190))
    fractions=[.315,.78,.3825,.4375,.4,.3825]
    result=torch.zeros(6,6,dtype=torch.bool)
    usable=torch.zeros(6,6,dtype=torch.bool)
    for i in range(6):
        for c,points in enumerate(labels['projected_points'][i]):
            xx,yy=zip(*points);x0,x1=max(0,int(min(xx))-2),min(512,int(max(xx))+3);y0,y1=max(0,int(min(yy))-2),min(512,int(max(yy))+3)
            pixels=int(changed[i,y0:y1,x0:x1].sum()) if x1>x0 and y1>y0 else 0
            expected=labels['projected_area'][i][c]*fractions[layout[i][c]]*(.0075/.009)**2
            result[i,c]=labels['facing'][i][c]>=.5 and pixels>=max(35,.65*expected)
            usable[i,c]=bool(result[i,c]) or pixels<=3 or labels['facing'][i][c]<=0
    return result,usable
