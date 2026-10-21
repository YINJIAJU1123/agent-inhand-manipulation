"""Higher-resolution RGB-D evidence, cropped using measured RGB only."""
import torch
import torch.nn.functional as F

@torch.inference_mode()
def object_regions(rgb, depth, size=96):
    if rgb.ndim!=4 or rgb.shape[-1]<3:raise ValueError('Expected batched HWC RGB')
    n,h,w,_=rgb.shape
    if depth.ndim==3:depth=depth[...,None]
    if depth.shape[:3]!=(n,h,w):raise ValueError('RGB and depth must be aligned')
    image=rgb[...,:3].float()/255.
    small=F.interpolate(image.permute(0,3,1,2),size=(128,128),mode='bilinear',align_corners=False).permute(0,2,3,1)
    colors=small.new_tensor([[.85,.05,.05],[.05,.75,.15],[.05,.25,.90],[.95,.75,.05],[.80,.05,.75],[.05,.80,.85]])
    distance=(small[:,None]-colors[None,:,None,None]).square().mean(-1).amin(1)
    visible=(distance<.035)&((small.amax(-1)-small.amin(-1))>.3)
    depth=torch.nan_to_num(depth.float(),nan=2.,posinf=2.,neginf=0.).clamp(0.,2.)/2.
    crops=[];dcrops=[];boxes=[];found=[]
    for i in range(n):
        yy,xx=visible[i].nonzero(as_tuple=True)
        valid=len(xx)>=4
        if valid:
            # Padding includes adjacent hand contacts and uncolored symbol ink.
            side=max(int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
            pad=max(4,int(side*.2))
            x0=max(0,(int(xx.min())-pad)*w//128);x1=min(w,(int(xx.max())+pad+1)*w//128)
            y0=max(0,(int(yy.min())-pad)*h//128);y1=min(h,(int(yy.max())+pad+1)*h//128)
        else:x0,y0,x1,y1=0,0,w,h
        patch=F.interpolate(image[i,y0:y1,x0:x1].permute(2,0,1)[None],size=(size,size),mode='bilinear',align_corners=False)[0]
        dp=F.interpolate(depth[i,y0:y1,x0:x1].permute(2,0,1)[None],size=(size,size),mode='bilinear',align_corners=False)[0]
        crops.append((patch*255).round().clamp(0,255).to(torch.uint8));dcrops.append(dp.to(torch.float16))
        boxes.append([x0/w,y0/h,x1/w,y1/h]);found.append(valid)
    return dict(roi_rgb=torch.stack(crops),roi_depth=torch.stack(dcrops),roi_box=image.new_tensor(boxes),roi_found=torch.tensor(found,device=rgb.device))
