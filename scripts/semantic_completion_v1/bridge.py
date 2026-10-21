"""Use only the newly consented profile; never fall back to the old account."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/astra_transport_v2'),str(ROOT/'scripts/agentic')]
import subscription_v2
import bridge as frozen_bridge

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--auth-dir',required=True)
    p.add_argument('--auth-result',required=True)
    p.add_argument('--identity-sha256',required=True)
    a,remaining=p.parse_known_args()
    path=Path(a.auth_dir).resolve()
    if path==Path.home()/'.config/agent-inhand-siwc':
        raise RuntimeError('new_account_profile_required')
    result=json.loads(Path(a.auth_result).read_text())
    if result.get('status')!='completed' or not result.get('expected_text_matched'):
        raise RuntimeError('new_account_smoke_not_completed')
    record=json.loads((path/'credentials.json').read_text())
    digest=hashlib.sha256(record['subject'].encode()).hexdigest()
    if digest!=a.identity_sha256:raise RuntimeError('account_identity_changed')
    subscription_v2.AUTH_DIR=path
    original=subscription_v2.token
    def pinned_token():
        fresh=json.loads((path/'credentials.json').read_text())
        if hashlib.sha256(fresh['subject'].encode()).hexdigest()!=digest:
            raise RuntimeError('account_identity_changed')
        return original()
    subscription_v2.token=pinned_token
    sys.argv=[sys.argv[0]]+remaining
    frozen_bridge.main()
