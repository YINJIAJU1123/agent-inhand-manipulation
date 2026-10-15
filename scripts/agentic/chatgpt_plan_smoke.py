"""Explicit SIWC login and one Astra smoke request; never reads Codex credentials.

Docs: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
Requires requests and PyJWT[crypto]. Tokens live outside the repository, mode 0600.
Run with --login, then open the printed loopback page and authorize in the browser.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
import requests

AUTH = 'https://auth.openai.com'
API = 'https://api.openai.com/v1'
SCOPES = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct'


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checked(response):
    if response.status_code != 200:
        # Never emit response bodies: authentication errors may echo credentials.
        raise RuntimeError(f'HTTP_{response.status_code}')
    return response


def smoke(record, model):
    headers = {'Authorization': 'Bearer ' + record['access_token']}
    if 'chatgpt.tokens.use.direct' not in record['scopes']:
        raise RuntimeError('plan_usage_scope_not_granted')
    if time.time() >= record['expires_at']:
        raise RuntimeError('token_expired_sign_in_again')
    catalog = checked(requests.get(API + '/models', headers=headers, timeout=30, allow_redirects=False)).json()
    models = [{'slug': x['slug'], 'display_name': x.get('display_name')} for x in catalog.get('models', []) if x.get('visibility') == 'list']
    result = {'auth_method': 'sign_in_with_chatgpt', 'models': models, 'requested_model': model}
    if model not in [x['slug'] for x in models]:
        result['status'] = 'requested_model_unavailable'
        return result
    payload = {'model': model, 'input': [{'role': 'user', 'content': 'Reply with exactly: ASTRA_SUBSCRIPTION_OK'}], 'store': False, 'stream': True}
    started = time.monotonic()
    with requests.post(API + '/responses', headers=headers, json=payload, stream=True, timeout=(15, 120), allow_redirects=False) as response:
        checked(response)
        data = []
        text_parts = {}
        for line in response.iter_lines(decode_unicode=True):
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if line.startswith('data:'):
                data.append(line[5:].lstrip())
            elif not line and data:
                raw, data = '\n'.join(data), []
                if raw == '[DONE]':
                    continue
                event = json.loads(raw)
                kind = event.get('type')
                key = (event.get('output_index', 0), event.get('content_index', 0))
                if kind == 'response.output_text.delta':
                    text_parts[key] = text_parts.get(key, '') + event.get('delta', '')
                elif kind == 'response.output_text.done':
                    text_parts[key] = event['text']
                if kind == 'response.completed':
                    r = event['response']
                    result.update(status='completed', model=r.get('model'), response_id=r.get('id'), usage=r.get('usage'), elapsed_s=time.monotonic()-started,
                                  output_text=''.join(c.get('text', '') for o in r.get('output', []) if o.get('type') == 'message' for c in o.get('content', []) if c.get('type') == 'output_text'))
                    if not result['output_text']:
                        result['output_text'] = ''.join(text_parts[k] for k in sorted(text_parts))
                    result['expected_text_matched'] = result['output_text'].strip() == 'ASTRA_SUBSCRIPTION_OK'
                    if not result['expected_text_matched']:
                        result['status'] = 'completed_unexpected_text'
                    return result
                if kind in ('response.failed', 'response.incomplete', 'error'):
                    code = (event.get('response', {}).get('error') or {}).get('code', '')
                    known = {'subscription_sharing_usage_limit_exceeded', 'subscription_sharing_usage_unavailable', 'subscription_sharing_user_not_eligible'}
                    raise RuntimeError(code if code in known else kind)
    raise RuntimeError('stream_ended_without_completion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--login', action='store_true')
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--port', type=int, default=18759)
    parser.add_argument('--auth-dir', type=Path, default=Path.home()/'.config/agent-inhand-siwc')
    parser.add_argument('--output', type=Path, default=Path('outputs/agentic_v4/astra_subscription_smoke/result.json'))
    args = parser.parse_args()
    os.umask(0o077)
    args.auth_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    host_file, record_file = args.auth_dir/'host.json', args.auth_dir/'credentials.json'
    if not host_file.exists():
        save(host_file, {'ext_agent_host_id': 'urn:uuid:' + str(uuid.uuid4())})
    host_id = json.loads(host_file.read_text())['ext_agent_host_id']
    def status(value):
        save(args.output, dict(checked_at=time.time(), **value))
        print(json.dumps(value), flush=True)
    if not args.login:
        status(smoke(json.loads(record_file.read_text()), args.model))
        return
    existing = json.loads(record_file.read_text()) if record_file.exists() else None
    pending_file = args.auth_dir/'pending_registration.json'
    pending = json.loads(pending_file.read_text()) if pending_file.exists() else None
    client_id = existing['client_id'] if existing else (pending['client_id'] if pending else 'dynamic_agent_client')
    state, nonce, verifier = [secrets.token_urlsafe(32) for _ in range(3)]
    callback = f'http://127.0.0.1:{args.port}/auth/callback'
    params = dict(client_id=client_id, ext_agent_host_id=host_id, response_type='code', redirect_uri=callback, scope=SCOPES, resource=API,
                  state=state, nonce=nonce, code_challenge_method='S256', code_challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode())
    if client_id == 'dynamic_agent_client':
        params['agent_name_hint'] = 'Agent Inhand Manipulation'
    elif existing and existing.get('id_token'):
        params['id_token_hint'] = existing['id_token']
    authorize = AUTH + '/api/accounts/authorize?' + urlencode(params)
    done = False
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def page(self, body, code=200):
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(('<!doctype html><meta charset="utf-8"><title>Agent Inhand — ChatGPT login</title><main style="font:18px sans-serif;max-width:720px;margin:80px auto">' + body + '</main>').encode())
        def do_GET(self):
            nonlocal done
            if self.headers.get('Host') != f'127.0.0.1:{args.port}':
                self.page('Invalid host', 400)
                return
            u = urlsplit(self.path)
            if u.path == '/':
                self.page('<h1>Agent Inhand Manipulation</h1><p>Authorize ChatGPT plan usage, then run one short Astra connectivity test.</p><p>No Platform API key or paid API fallback is used.</p><a href="/signin">Continue with ChatGPT</a>')
                return
            if u.path == '/signin':
                self.send_response(302)
                self.send_header('Location', authorize)
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                return
            if u.path != '/auth/callback':
                self.page('Not found', 404)
                return
            q = parse_qs(u.query)
            if not secrets.compare_digest(q.get('state', [''])[0], state):
                self.page('Invalid OAuth state.', 400)
                return
            done = True
            replied = False
            try:
                if 'error' in q:
                    raise RuntimeError('authorization_declined_or_failed')
                issued = q.get('client_id', [client_id])[0]
                if not issued or issued == 'dynamic_agent_client' or (client_id != 'dynamic_agent_client' and issued != client_id):
                    raise RuntimeError('invalid_issued_client_id')
                save(pending_file, {'client_id': issued})
                token = checked(requests.post(AUTH+'/api/accounts/oauth/token', data=dict(grant_type='authorization_code', client_id=issued,
                     code=q['code'][0], code_verifier=verifier, redirect_uri=callback, resource=API), timeout=30, allow_redirects=False)).json()
                discovery = checked(requests.get(AUTH+'/.well-known/openid-configuration', timeout=20, allow_redirects=False)).json()
                if discovery['issuer'] != AUTH or not discovery['jwks_uri'].startswith(AUTH+'/'):
                    raise RuntimeError('invalid_discovery')
                key = jwt.PyJWKClient(discovery['jwks_uri']).get_signing_key_from_jwt(token['id_token'])
                identity = jwt.decode(token['id_token'], key.key, algorithms=['RS256'], audience=issued, issuer=AUTH,
                                      options={'require': ['exp', 'iss', 'aud', 'sub', 'nonce']})
                if not secrets.compare_digest(identity['nonce'], nonce):
                    raise RuntimeError('invalid_oidc_nonce')
                if existing and identity['sub'] != existing['subject']:
                    raise RuntimeError('account_identity_changed')
                record = dict(client_id=issued, subject=identity['sub'], issuer=AUTH, email=identity.get('email'), ext_agent_host_id=host_id,
                              id_token=token['id_token'], access_token=token['access_token'], refresh_token=token.get('refresh_token'),
                              scopes=token.get('scope', '').split(), token_type=token.get('token_type'), expires_at=time.time()+token['expires_in'])
                save(record_file, record)
                pending_file.unlink(missing_ok=True)
                status({'status': 'authorized_testing'})
                self.page('ChatGPT authorization verified. The Astra test is running; return to Codex for its result.')
                replied = True
                status(smoke(record, args.model))
            except Exception as e:
                # Only our controlled RuntimeError messages are safe to show.
                reason = str(e) if type(e) is RuntimeError else type(e).__name__
                status({'status': 'failed', 'reason': reason})
                if not replied:
                    self.page('Authorization or connection test failed. Return to Codex for the error category.', 400)
    with HTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.timeout = 1
        status({'status': 'awaiting_browser_authorization', 'login_url': f'http://127.0.0.1:{args.port}/'})
        deadline = time.monotonic()+1800
        while not done and time.monotonic() < deadline:
            server.handle_request()
        if not done:
            status({'status': 'authorization_timed_out'})


if __name__ == '__main__':
    main()
