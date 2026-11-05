"""Start a loopback semantic bridge using the colleague's own API key."""
import argparse,os,secrets,sys
from pathlib import Path
from functools import partial
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',choices=['cube','die','duck','hedgehog','pig','milk_carton'],required=True)
    p.add_argument('--port',type=int,default=19080);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('OPENAI_API_KEY'):raise SystemExit('Set your own OPENAI_API_KEY; account/OAuth profiles are not part of this handoff.')
    family='pose_latent_student_v1' if a.object=='cube' else 'dice_latent_v1' if a.object=='die' else 'inhand4_q1'
    sys.path[:0]=[str(ROOT/'scripts'/family),str(ROOT/'scripts/astra_transport_v2'),str(ROOT/'scripts/agentic')]
    from semantic import SYSTEM,tools_schema,validate_action
    import bridge
    from subscription_v2 import AstraSubscriptionClient
    a.run_dir.mkdir(parents=True,exist_ok=False);key=a.run_dir/'bridge.key';key.write_text(secrets.token_urlsafe(32)+'\n');key.chmod(0o600)
    bridge.AstraSubscriptionClient=partial(AstraSubscriptionClient,system_prompt=SYSTEM,tool_schema=tools_schema,action_validator=validate_action)
    sys.argv=[sys.argv[0],'--port',str(a.port),'--key-file',str(key),'--audit-dir',str(a.run_dir/'audit'),'--workers','1',
        '--frame-size','120' if a.object=='cube' else '512','--max-calls','360','--max-tokens','3000000','--max-sessions','60','--transport-attempts','1']
    bridge.main()
if __name__=='__main__':main()
