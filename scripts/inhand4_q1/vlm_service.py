"""Local dice VLM inference, with either learned latent or frozen text rules."""
import argparse,base64,hashlib,io,json,time,sys,os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
from PIL import Image,ImageOps
import torch
from semantic import VLM_PROMPT,RULE_SUFFIX
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agentic'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multilingual_eval_v1'))
from agent_observation_v4 import rgb_crop_box
from latent_goal_v4 import load_backbone
from protocol import RULE_PROMPT_ZH,RULE_SCHEMA_ZH

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=('vlm','rule'),required=True)
 p.add_argument('--model',required=True);p.add_argument('--checkpoint');p.add_argument('--port',type=int,required=True)
 a=p.parse_args();torch.set_num_threads(4)
 if a.kind=='vlm':
  if not a.checkpoint:p.error('Latent inference requires checkpoint')
  from grounding import DiceLatentPredictor
  predictor=DiceLatentPredictor(a.checkpoint,a.model)
  meta=dict(checkpoint_sha256=predictor.checkpoint_sha256,adapter_sha256=predictor.adapter_sha256)
 else:
  if a.checkpoint:p.error('Rule baseline cannot receive fine-tuned weights')
  model,processor=load_backbone(a.model);model.eval();model.requires_grad_(False)
  meta=dict(checkpoint_sha256=None,adapter_sha256=None,base_model=str(Path(a.model).resolve()))
 meta.update(kind=a.kind,backend='inhand4_marked_q1_v1',semantic_sha256=hashlib.sha256(Path(__file__).with_name('semantic.py').read_bytes()).hexdigest(),frame_size=512,extra_control_modules=False)
 @torch.inference_mode()
 def predict(rgb,instruction,language='en'):
  if a.kind=='vlm':
   result=predictor.predict(rgb,instruction);result['goal_embedding']=result['goal_embedding'].tolist();result['language']=language;return result
  frame=Image.fromarray(rgb);box=rgb_crop_box(rgb);crop=ImageOps.pad(frame.crop(box),(336,336),color=(32,32,32))
  if language=='zh':
   object_name=os.environ.get('VISERDEX_OBJECT_NAME','duck')
   if object_name not in RULE_PROMPT_ZH:raise ValueError('Missing localized prompt for object: '+object_name)
   motor_contract=('\n固定的颜色到动作技能编号映射为：1 红色，2 绿色，3 蓝色，4 黄色，5 青色，6 紫色。'
    '每个技能将对应颜色的表面区域转到规范的朝上姿态。字母可能位于与训练时不同的颜色上，'
    '请从图像中读取实际的字母与颜色关系，不要把字母顺序当作动作编号。仅支持朝上目标。')
   prompt=RULE_PROMPT_ZH[object_name]+motor_contract+'\n用户指令：'+instruction+'\n'+RULE_SCHEMA_ZH[object_name]
  elif language=='en':prompt=VLM_PROMPT.format(instruction=instruction)+RULE_SUFFIX
  else:raise ValueError('Unsupported language')
  content=[dict(type='image',image=frame),dict(type='image',image=crop),dict(type='text',text=prompt)]
  inputs=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:0')
  tokens=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
  raw=processor.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
  parsed=None;decoder=json.JSONDecoder()
  for i,c in enumerate(raw):
   if c=='{':
    try:parsed=decoder.raw_decode(raw[i:])[0];break
    except json.JSONDecodeError:pass
  valid=isinstance(parsed,dict) and type(parsed.get('visible')) is bool and (parsed.get('number') is None or type(parsed.get('number')) is int and 1<=parsed['number']<=6)
  accepted=bool(valid and parsed['visible'] and parsed['number'] is not None)
  return dict(accepted=accepted,skill=parsed['number']-1 if accepted else None,raw=raw,parsed=parsed,format_valid=valid,crop_box=box,image_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),language=language,actual_prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,code,data):
   body=json.dumps(data,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def do_GET(self):self.send(200,meta)
  def do_POST(self):
   try:
    if self.path!='/predict':raise ValueError('Wrong route')
    length=int(self.headers.get('Content-Length','0'))
    if not 0<length<4000000:raise ValueError('Request size')
    data=json.loads(self.rfile.read(length));rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(data['image'],validate=True))).convert('RGB'))
    if rgb.shape!=(512,512,3) or not isinstance(data['instruction'],str):raise ValueError('Invalid image/instruction')
    language=data.get('language','en')
    if language not in ('en','zh'):raise ValueError('Unsupported language')
    start=time.monotonic();result=predict(rgb,data['instruction'],language);result['inference_s']=time.monotonic()-start;self.send(200,result)
   except Exception as e:self.send(500,dict(error=type(e).__name__+': '+str(e)))
 print(json.dumps(dict(status='ready',port=a.port,**meta)),flush=True)
 HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
