"""Separate GPU process for fine-tuned latent inference; loopback HTTP only."""
import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import requests
from PIL import Image

class RemotePredictor:
    def __init__(self, url):
        if not url.startswith('http://127.0.0.1:'): raise ValueError('Loopback required')
        self.url=url
        self.metadata=requests.get(url+'/health',timeout=5).json()
        self.checkpoint_sha256=self.metadata['checkpoint_sha256']
        self.adapter_sha256=self.metadata['adapter_sha256']
    def predict(self,rgb,instruction):
        buf=io.BytesIO();Image.fromarray(rgb).save(buf,format='PNG')
        r=requests.post(self.url+'/predict',json=dict(image=base64.b64encode(buf.getvalue()).decode(),instruction=instruction),timeout=(10,180))
        r.raise_for_status();return r.json()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,default=18772)
    p.add_argument('--backend',choices=('legacy','rslcube'),default='legacy')
    p.add_argument('--model_path',help='Local backbone path for a relocated RSLCube checkpoint; weights are not rewritten.')
    a=p.parse_args()
    if a.backend=='rslcube':
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pose_latent_student_v1'))
        from grounding import RSLCubeLatentPredictor as LatentPredictor
    else:
        from latent_goal_v4 import LatentPredictor
    if a.model_path and a.backend!='rslcube':raise ValueError('Backbone relocation is only supported for RSLCube')
    predictor=LatentPredictor(a.checkpoint,model_path=a.model_path) if a.model_path else LatentPredictor(a.checkpoint)
    frame_size=120 if a.backend=='rslcube' else 512
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,code,data):
            body=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.send(200,dict(checkpoint_sha256=predictor.checkpoint_sha256,adapter_sha256=predictor.adapter_sha256,queue='single GPU serialized requests',backend=a.backend,frame_size=frame_size))
        def do_POST(self):
            try:
                length=int(self.headers['Content-Length'])
                if not 0<length<4000000: raise ValueError('request size')
                data=json.loads(self.rfile.read(length));rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(data['image']))).convert('RGB'))
                if rgb.shape!=(frame_size,frame_size,3):raise ValueError('image shape')
                start=time.monotonic();result=predictor.predict(rgb,data['instruction']);result['goal_embedding']=result['goal_embedding'].tolist();result['inference_s']=time.monotonic()-start
                self.send(200,result)
            except Exception as e:self.send(500,dict(error=type(e).__name__+': '+str(e)))
    print('latent_service_ready',flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
