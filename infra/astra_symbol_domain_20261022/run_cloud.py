"""Frozen-weight image-domain diagnosis on the stored calibration split only."""
import base64,hashlib,io,json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFilter
from model import StickerReader
out=Path(os.environ['DOMAIN_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
root=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_symbol_regions_20261022/learning_r1')
torch.set_num_threads(2);pack=torch.load(root/'train/head.pt',map_location='cpu',weights_only=True)
model=StickerReader().eval();model.load_state_dict(pack['model'])
data=torch.load(root/'data/val_96211/data.pt',map_location='cpu',weights_only=True)
known=(data['labels']>=0)&(data['labels']<6);usable=data['labels']>=0
metrics={};start=time.time()
def transform(x,kind):
 if kind=='native':return x
 images=[]
 for value in x:
  im=Image.fromarray(value.permute(1,2,0).numpy())
  if kind.startswith('jpeg'):
   stream=io.BytesIO();im.save(stream,format='JPEG',quality=int(kind[4:]));stream.seek(0);im=Image.open(stream).convert('RGB')
  elif kind=='blur0.7':im=im.filter(ImageFilter.GaussianBlur(.7))
  elif kind=='half_resolution':im=im.resize((256,256),Image.Resampling.BILINEAR).resize((512,512),Image.Resampling.BILINEAR)
  images.append(torch.from_numpy(np.array(im,copy=True)).permute(2,0,1))
 return torch.stack(images)
with torch.inference_mode():
 for kind in ['native','jpeg75','jpeg50','blur0.7','half_resolution']:
  probabilities=[]
  for begin in range(0,len(data['rgb']),4):probabilities.append(model(transform(data['rgb'][begin:begin+4],kind))['probability'])
  p=torch.cat(probabilities);confidence,pred=p[:,:,:6].max(-1);accepted=(confidence>=pack['threshold'])&usable
  macro=[float((p.argmax(-1)[data['labels']==c]==c).float().mean())for c in range(6)]
  metrics[kind]=dict(accepted=int(accepted.sum()),precision=float((pred[accepted]==data['labels'][accepted]).float().mean())if accepted.any()else None,macro_readable=sum(macro)/6,per_class=macro)
  print('SYMBOL_DOMAIN_PROGRESS '+json.dumps(dict(domain=kind,metrics=metrics[kind],elapsed_s=time.time()-start)),flush=True)
 canvas=Image.new('RGB',(256*4,300*3),'white');index=0;symbols=('star','circle','triangle','cross','diamond','bars');colors=('red','green','blue','yellow','magenta','cyan')
 for symbol in range(6):
  candidates=torch.nonzero(data['labels']==symbol)
  for row,color in candidates[::max(1,len(candidates)//2)][:2].tolist():
   output=model(data['rgb'][row:row+1]);crop=output['crops'][color];im=Image.fromarray((crop.permute(1,2,0).clamp(0,1).numpy()*255).astype('uint8')).resize((256,256))
   x,y=(index%4)*256,(index//4)*300;canvas.paste(im,(x,y));d=ImageDraw.Draw(canvas);d.text((x+3,y+258),colors[color]+' '+symbols[symbol],fill='black');d.text((x+3,y+274),str(data['rows'][row]),fill='black');index+=1
 canvas.save(out/'calibration_crops.png')
result=dict(metrics=metrics,checkpoint_sha256=hashlib.sha256((root/'train/head.pt').read_bytes()).hexdigest(),cloud_calls=0,note='Frozen calibration-only input perturbation diagnosis. No fit, threshold change, audit selection or deployment.')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('SYMBOL_DOMAIN_RESULT '+json.dumps(result),flush=True)
image=(out/'calibration_crops.png').read_bytes();print('SYMBOL_REGION_PREVIEW_SHA256 '+hashlib.sha256(image).hexdigest(),flush=True);print('SYMBOL_REGION_PREVIEW_BEGIN',flush=True)
s=base64.b64encode(image).decode()
for i in range(0,len(s),64):print(s[i:i+64],flush=True)
print('SYMBOL_REGION_PREVIEW_END',flush=True)
