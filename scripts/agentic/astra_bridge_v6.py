"""Loopback-only, run-scoped Astra bridge for SSH-tunneled simulator workers.

OAuth credentials remain on the user's local machine. A separate random bridge
key authorizes only this bounded experiment; it is not an OpenAI credential.
"""
import argparse
import base64
import hmac
import io
import json
from pathlib import Path
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import requests
from PIL import Image
from astra_subscription_v6 import AstraSubscriptionClient, preflight
from visual_agent import encode_png
from transport_retry_v7 import ask_with_retry


class BridgeClient:
    def __init__(self, model='gpt-6-astra', audit_dir=None, max_calls=360, max_tokens=3000000):
        import os
        self.url=os.environ['ASTRA_BRIDGE_URL']
        if not self.url.startswith('http://127.0.0.1:'):raise ValueError('Bridge must use SSH loopback')
        self.key=Path(os.environ['ASTRA_BRIDGE_KEY_FILE']).read_text().strip()
        self.session=uuid.uuid4().hex
        self.model=model
    def ask(self,instruction,rgb,feedback,previous_response_id=None,previous_call_id=None):
        payload=dict(session=self.session,instruction=instruction,image=encode_png(Image.fromarray(np.asarray(rgb,dtype=np.uint8))),
                     feedback=feedback,previous_response_id=previous_response_id,previous_call_id=previous_call_id)
        r=requests.post(self.url+'/ask',headers={'Authorization':'Bearer '+self.key},json=payload,timeout=(10,300),allow_redirects=False)
        if r.status_code!=200:raise RuntimeError(f'bridge_http_{r.status_code}: '+r.json().get('error','request_failed'))
        return r.json()


def bridge_health():
    import os
    url=os.environ['ASTRA_BRIDGE_URL']
    key=Path(os.environ['ASTRA_BRIDGE_KEY_FILE']).read_text().strip()
    r=requests.get(url+'/health',headers={'Authorization':'Bearer '+key},timeout=15,allow_redirects=False)
    if r.status_code!=200 or r.json().get('model')!='gpt-6-astra':raise RuntimeError('bridge_unavailable')


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18761);p.add_argument('--key-file',required=True)
    p.add_argument('--audit-dir',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-calls',type=int,default=360);p.add_argument('--max-tokens',type=int,default=3000000);p.add_argument('--max-sessions',type=int,default=60)
    p.add_argument('--transport-attempts',type=int,choices=[1,2],default=1)
    a=p.parse_args();preflight();key=Path(a.key_file).read_text().strip()
    clients={};locks={};registry=threading.Lock();slots=threading.Semaphore(a.workers)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,data):
            body=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+key)
        def do_GET(self):
            if not self.authorized():self.send(403,{'error':'unauthorized'});return
            self.send(200,{'model':'gpt-6-astra','auth_method':'sign_in_with_chatgpt','workers':a.workers})
        def do_POST(self):
            if not self.authorized():self.send(403,{'error':'unauthorized'});return
            if self.path!='/ask':self.send(404,{'error':'not_found'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<4000000:raise ValueError('invalid_request_size')
                d=json.loads(self.rfile.read(length)); session=d['session']
                if not isinstance(session,str) or len(session)!=32:raise ValueError('invalid_session')
                with registry:
                    if session not in clients:
                        if len(clients)>=a.max_sessions:raise ValueError('session_cap_reached')
                        clients[session]=AstraSubscriptionClient(audit_dir=a.audit_dir,max_calls=a.max_calls,max_tokens=a.max_tokens)
                        locks[session]=threading.Lock()
                im=Image.open(io.BytesIO(base64.b64decode(d['image'],validate=True)))
                if im.size!=(512,512):raise ValueError('invalid_frame_size')
                with locks[session],slots:
                    result=ask_with_retry(clients[session],d['instruction'],np.asarray(im.convert('RGB')),d['feedback'],d.get('previous_response_id'),d.get('previous_call_id'),attempts=a.transport_attempts)
                self.send(200,result)
            except Exception as e:
                reason=str(e) if type(e) in (RuntimeError,ValueError) else type(e).__name__
                self.send(503,{'error':reason})
    print(json.dumps({'status':'ready','port':a.port,'workers':a.workers}),flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
