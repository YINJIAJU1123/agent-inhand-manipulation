"""Local pixel/text-only VLM service; no simulator state or file names in requests."""
import argparse,base64,io,json,time
from http.server import BaseHTTPRequestHandler,HTTPServer
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText,AutoProcessor,BitsAndBytesConfig

p=argparse.ArgumentParser()
p.add_argument('--model',required=True); p.add_argument('--port',type=int,default=18741)
p.add_argument('--quantize',action='store_true'); a=p.parse_args()
torch.set_num_threads(4)
kwargs=dict(dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa')
if a.quantize:
    kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
model=AutoModelForImageTextToText.from_pretrained(a.model,**kwargs).eval()
processor=AutoProcessor.from_pretrained(a.model,min_pixels=128*28*28,max_pixels=384*384)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'ready':True,'model':a.model,'quantize':a.quantize}).encode())
    def do_POST(self):
        start=time.monotonic()
        try:
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            # Strict allow-list excludes target ids, layout metadata, and simulator metrics.
            if set(request)-{'images','prompt','max_tokens'}: raise ValueError('unsupported request fields')
            images=[Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB') for b in request['images']]
            content=[{'type':'image','image':im} for im in images]+[{'type':'text','text':request['prompt']}]
            inputs=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,
                        add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                tokens=model.generate(**inputs,max_new_tokens=min(int(request.get('max_tokens',96)),192),do_sample=False)
            text=processor.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
            result={'text':text,'elapsed_s':time.monotonic()-start}
            self.send_response(200)
        except Exception as e:
            result={'error':str(e),'elapsed_s':time.monotonic()-start}; self.send_response(500)
        self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps(result).encode())
    def log_message(self,*args): pass
print(json.dumps({'ready':True,'port':a.port,'model':a.model,'quantize':a.quantize}),flush=True)
HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
