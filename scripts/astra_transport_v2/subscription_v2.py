"""SIWC Astra supervisor transport. Isolated histories, bounded concurrent usage.

Uses only the app's separately consented OAuth record and public Responses API.
No Platform key fallback; no simulator labels are accepted by ask().
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

import numpy as np
import requests
from PIL import Image, ImageOps
from astra_agent_v4 import SYSTEM, tools_schema, validate_action
from visual_agent import encode_png
from agent_observation_v4 import rgb_crop_box

API = 'https://api.openai.com/v1'
AUTH_DIR = Path.home()/'.config/agent-inhand-siwc'


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f: json.dump(data, f, indent=2); f.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def safe_code(error):
    value = error.get('code') if isinstance(error, dict) else None
    return value if isinstance(value, str) and re.fullmatch(r'[a-z_]{1,100}', value) else 'api_error'


def token():
    # A flock per refresh transaction serializes rotating refresh tokens across workers.
    with (AUTH_DIR/'refresh.lock').open('a') as lock:
        os.chmod(AUTH_DIR/'refresh.lock', 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = AUTH_DIR/'credentials.json'; record = json.loads(path.read_text())
        if 'chatgpt.tokens.use.direct' not in record['scopes']: raise RuntimeError('plan_usage_not_authorized')
        if record['expires_at'] < time.time()+120:
            if not record.get('refresh_token'): raise RuntimeError('refresh_token_missing')
            response = requests.post('https://auth.openai.com/api/accounts/oauth/token', data={
                'grant_type':'refresh_token', 'client_id':record['client_id'],
                'refresh_token':record['refresh_token'], 'resource':API}, timeout=30, allow_redirects=False)
            if response.status_code != 200: raise RuntimeError(f'oauth_refresh_http_{response.status_code}')
            update = response.json()
            # Save rotated credentials even if plan scope has been removed.
            record.update(access_token=update['access_token'], refresh_token=update['refresh_token'],
                          expires_at=time.time()+update['expires_in'], scopes=update.get('scope',' '.join(record['scopes'])).split())
            if update.get('id_token'): record['id_token'] = update['id_token']
            save(path, record)
            if 'chatgpt.tokens.use.direct' not in record['scopes']: raise RuntimeError('plan_usage_not_authorized')
        return record['access_token']


def preflight(model='gpt-6-astra'):
    response=requests.get(API+'/models',headers={'Authorization':'Bearer '+token()},timeout=30,allow_redirects=False)
    if response.status_code != 200: raise RuntimeError(f'model_list_http_{response.status_code}')
    if model not in [x['slug'] for x in response.json().get('models',[]) if x.get('visibility')=='list']:
        raise RuntimeError('requested_model_unavailable')


class UsageLedger:
    def __init__(self, directory, max_calls=360, max_tokens=3000000):
        self.root=Path(directory); self.root.mkdir(parents=True, exist_ok=True)
        self.max_calls,self.max_tokens=max_calls,max_tokens
    def reserve(self):
        with (self.root/'usage.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            logs=[json.loads(p.read_text()) for p in self.root.glob('request_*.json')]
            if len(logs)>=self.max_calls: raise RuntimeError('experiment_call_cap_reached')
            if sum(x.get('usage',{}).get('total_tokens',0) for x in logs)>=self.max_tokens:
                raise RuntimeError('experiment_token_cap_reached')
            path=self.root/('request_'+uuid.uuid4().hex+'.json')
            save(path,{'status':'in_flight','started_at':time.time()})
            return path


class AstraSubscriptionClient:
    def __init__(self, model='gpt-6-astra', audit_dir=None, max_calls=360, max_tokens=3000000,
                 system_prompt=SYSTEM, tool_schema=tools_schema, action_validator=validate_action):
        if audit_dir is None: raise ValueError('An audit directory is required')
        self.model=model
        self.system_prompt=system_prompt
        self.tool_schema=tool_schema
        self.action_validator=action_validator
        self.http=requests.Session()
        self.ledger=UsageLedger(audit_dir,max_calls,max_tokens)
        self.history=[]; self.last_response=None; self.last_call=None

    def ask(self,instruction,rgb,feedback,previous_response_id=None,previous_call_id=None):
        if previous_response_id != self.last_response or previous_call_id != self.last_call:
            raise ValueError('History belongs to a different episode')
        frame=Image.fromarray(np.asarray(rgb,dtype=np.uint8)); box=rgb_crop_box(rgb)
        crop=ImageOps.pad(frame.crop(box),(336,336),color=(32,32,32))
        messages=list(self.history)
        if previous_call_id:
            messages.append({'type':'function_call_output','call_id':previous_call_id,
                             'output':json.dumps({'measured_feedback':feedback,'new_observation_follows':True})})
        prompt=json.dumps({'instruction':instruction,'feedback':feedback},ensure_ascii=False)
        content=[{'type':'input_text','text':prompt}]
        content += [{'type':'input_image','image_url':'data:image/png;base64,'+encode_png(im),'detail':'high'} for im in (frame,crop)]
        messages.append({'role':'user','content':content})
        payload={'model':self.model,'instructions':self.system_prompt,'input':messages,
                 'tools':[{'type':'namespace','name':'robot','description':'Bounded dexterous hand observation and reorientation skills.','tools':self.tool_schema()}],
                 'tool_choice':'required','parallel_tool_calls':False,'reasoning':{'effort':'low'},
                 'include':['reasoning.encrypted_content'],'store':False,'stream':True}
        bearer=token()
        path=self.ledger.reserve(); start=time.monotonic(); usage={}
        try:
            terminal=None; items={}; data=[]
            with self.http.post(API+'/responses',headers={'Authorization':'Bearer '+bearer},json=payload,
                               stream=True,timeout=(15,120),allow_redirects=False) as response:
                if response.status_code!=200:
                    try: code=safe_code(response.json().get('error'))
                    except Exception: code='api_error'
                    raise RuntimeError(f'http_{response.status_code}_{code}')
                for line in response.iter_lines():
                    line=line.decode('utf-8')
                    if line.startswith('data:'): data.append(line[5:].lstrip())
                    elif not line and data:
                        raw,data='\n'.join(data),[]
                        if raw=='[DONE]':continue
                        event=json.loads(raw); kind=event.get('type')
                        if kind=='response.output_item.done':items[event['output_index']]=event['item']
                        elif kind=='response.completed':
                            terminal=event['response'];usage=terminal.get('usage') or {};break
                        elif kind in ('response.failed','response.incomplete','error'):
                            raise RuntimeError(kind+'_'+safe_code(event.get('response',{}).get('error',event.get('error'))))
            if terminal is None or terminal.get('status')!='completed': raise RuntimeError('stream_not_completed')
            output=terminal.get('output') or [items[k] for k in sorted(items)]
            calls=[x for x in output if x.get('type')=='function_call']
            if len(calls)!=1: raise RuntimeError('expected_one_robot_tool_call')
            call=calls[0]
            if call.get('namespace') not in (None,'robot'):raise RuntimeError('unexpected_tool_namespace')
            args=self.action_validator(call['name'],json.loads(call['arguments']))
            # Returned reasoning and tool calls stay local in this episode only.
            self.history=messages+output;self.last_response=terminal['id'];self.last_call=call['call_id']
            result={'tool':call['name'],'arguments':args,'call_id':call['call_id'],'response_id':terminal['id'],
                    'model':terminal.get('model'),'usage':usage,'elapsed_s':time.monotonic()-start,
                    'instruction':instruction,'prompt':prompt,'system_prompt':self.system_prompt,
                    'image_sha256':hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),'crop_box':box,
                    'raw_tool_call':call,'auth_method':'sign_in_with_chatgpt','history_items':len(messages),
                    'request_audit_file':path.name}
            save(path,{'status':'completed','finished_at':time.time(),**result})
            return result
        except Exception as error:
            reason=str(error) if isinstance(error,(RuntimeError,ValueError)) else type(error).__name__
            save(path,{'status':'failed','reason':reason,'usage':usage,'elapsed_s':time.monotonic()-start})
            raise RuntimeError(reason) from None
