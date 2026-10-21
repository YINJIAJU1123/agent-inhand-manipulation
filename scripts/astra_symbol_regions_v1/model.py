"""Learn six sticker regions from RGB, then read their symbols at native resolution."""
import torch
from torch import nn
from torch.nn import functional as F


def crop_regions(rgb, boxes, size=64):
    """Differentiable measured-image crops. Boxes are normalized cx, cy, w, h."""
    n, q = boxes.shape[:2]
    axis = (torch.arange(size, device=rgb.device, dtype=rgb.dtype)+.5)/size-.5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    offset = torch.stack((xx, yy), -1)[None,None]
    grid = boxes[:,:,:2][:,:,None,None,:]+offset*boxes[:,:,2:][:,:,None,None,:]
    image = rgb[:,None].expand(-1,q,-1,-1,-1).reshape(n*q,*rgb.shape[1:])
    return F.grid_sample(image,grid.reshape(n*q,size,size,2)*2-1,mode='bilinear',padding_mode='zeros',align_corners=False)


class StickerReader(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b,stride in [(3,24,2),(24,32,2),(32,48,1),(48,64,1)]:
            layers.extend([nn.Conv2d(a,b,3,stride=stride,padding=1),nn.GroupNorm(4,b),nn.SiLU()])
        self.backbone=nn.Sequential(*layers)
        self.locations=nn.Conv2d(64,6,1)
        self.extent=nn.Sequential(nn.Linear(70,64),nn.SiLU(),nn.Linear(64,2))
        self.visible=nn.Sequential(nn.Linear(70,48),nn.SiLU(),nn.Linear(48,1))
        layers=[]
        for a,b in [(3,16),(16,32),(32,48)]:
            layers.extend([nn.Conv2d(a,b,3,stride=2,padding=1),nn.GroupNorm(4,b),nn.SiLU()])
        self.glyph=nn.Sequential(*layers,nn.Flatten(),nn.Linear(48*8*8,128),nn.SiLU(),nn.Dropout(.1))
        self.symbol=nn.Linear(134,6)

    def classify(self,crops):
        n=len(crops)//6
        colors=F.one_hot(torch.arange(6,device=crops.device),6).float().repeat(n,1)
        return self.symbol(torch.cat((self.glyph(crops),colors),-1)).reshape(n,6,6)

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3:
            raise ValueError('Expected measured NCHW RGB')
        rgb=rgb.float()/255. if rgb.dtype==torch.uint8 else rgb
        features=self.backbone(F.interpolate(rgb,(256,256),mode='bilinear',align_corners=False))
        n,_,h,w=features.shape
        logits=self.locations(features)
        weights=logits.flatten(2).softmax(-1)
        yy,xx=torch.meshgrid((torch.arange(h,device=rgb.device)+.5)/h,(torch.arange(w,device=rgb.device)+.5)/w,indexing='ij')
        coords=torch.stack((xx,yy),-1).reshape(-1,2)
        center=weights@coords
        pooled=weights@features.flatten(2).transpose(1,2)
        color=F.one_hot(torch.arange(6,device=rgb.device),6).float()[None].expand(n,-1,-1)
        z=torch.cat((pooled,color),-1)
        extent=.02+.48*self.extent(z).sigmoid()
        boxes=torch.cat((center,extent),-1)
        visible=self.visible(z).squeeze(-1)
        crops=crop_regions(rgb,boxes)
        symbol=self.classify(crops)
        probability=torch.cat((visible.sigmoid()[...,None]*symbol.softmax(-1),(1-visible.sigmoid())[...,None]),-1)
        return dict(boxes=boxes,location_logits=logits,visible_logit=visible,symbol_logits=symbol,probability=probability,crops=crops)


def losses(model,rgb,boxes,labels,output,predicted_weight=.2):
    known=(labels>=0)&(labels<6);usable=labels>=0
    if not known.any() or not usable.any():
        raise ValueError('Training batch lacks readable or usable supervision')
    h,w=output['location_logits'].shape[-2:]
    yy,xx=torch.meshgrid((torch.arange(h,device=rgb.device)+.5)/h,(torch.arange(w,device=rgb.device)+.5)/w,indexing='ij')
    coords=torch.stack((xx,yy),-1)[None,None]
    distances=(coords-boxes[:,:,:2][:,:,None,None,:]).square().sum(-1)
    target=(-distances/(2*.015**2)).flatten(2).softmax(-1)
    location=-(target*output['location_logits'].flatten(2).log_softmax(-1)).sum(-1)[known].mean()
    extent=F.smooth_l1_loss(output['boxes'][known][:,2:].log(),boxes[known][:,2:].clamp_min(.002).log())
    presence=F.binary_cross_entropy_with_logits(output['visible_logit'][usable],known[usable].float(),pos_weight=torch.tensor(2.,device=rgb.device))
    # Privileged boxes appear only in the supervision branch, never forward().
    actual=rgb.float()/255. if rgb.dtype==torch.uint8 else rgb
    gt_logits=model.classify(crop_regions(actual,boxes))
    glyph=F.cross_entropy(gt_logits[known],labels[known])
    predicted=F.cross_entropy(output['symbol_logits'][known],labels[known])
    loss=location+extent+presence+glyph+predicted_weight*predicted
    return loss,dict(location=location,extent=extent,presence=presence,glyph=glyph,predicted_glyph=predicted)
