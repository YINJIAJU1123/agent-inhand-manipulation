"""Transport contract tests; no credentials, network, or simulator required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agentic'))
from astra_subscription_v6 import AstraSubscriptionClient,UsageLedger

class Reply:
    status_code=200
    def __init__(self,events):self.events=events
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_lines(self):
        for event in self.events:
            yield ('data: '+json.dumps(event)).encode();yield b''

class Contract(unittest.TestCase):
    def test_stream_tools_and_episode_history(self):
        with tempfile.TemporaryDirectory() as directory:
            c=AstraSubscriptionClient(audit_dir=directory)
            call={'type':'function_call','namespace':'robot','name':'inspect_face','arguments':json.dumps({'color':'red','steps':30,'evidence':'Inspect the visible sticker.'}),'call_id':'call1'}
            events=[{'type':'response.output_item.done','output_index':0,'item':call},
                    {'type':'response.completed','response':{'status':'completed','id':'r1','output':[],'usage':{'total_tokens':20}}}]
            with patch('astra_subscription_v6.token',return_value='test-only'),patch('astra_subscription_v6.requests.post',return_value=Reply(events)) as post:
                result=c.ask('Bring star up',np.zeros((512,512,3),dtype=np.uint8),{'step':0})
                payload=post.call_args.kwargs['json']
                self.assertFalse(payload['store']);self.assertTrue(payload['stream'])
                self.assertNotIn('previous_response_id',payload);self.assertNotIn('max_output_tokens',payload)
                self.assertEqual(payload['tools'][0]['type'],'namespace')
                self.assertEqual(result['tool'],'inspect_face')
                other=AstraSubscriptionClient(audit_dir=directory)
                with self.assertRaisesRegex(ValueError,'different episode'):
                    other.ask('Bring star up',np.zeros((512,512,3),dtype=np.uint8),{},'r1','call1')
                self.assertEqual(len(other.history),0)
            self.assertEqual(len(list(Path(directory).glob('request_*.json'))),1)

    def test_failed_stream_never_becomes_success(self):
        with tempfile.TemporaryDirectory() as directory:
            c=AstraSubscriptionClient(audit_dir=directory)
            events=[{'type':'response.failed','response':{'error':{'code':'subscription_sharing_usage_limit_exceeded'}}}]
            with patch('astra_subscription_v6.token',return_value='test-only'),patch('astra_subscription_v6.requests.post',return_value=Reply(events)):
                with self.assertRaisesRegex(RuntimeError,'usage_limit_exceeded'):
                    c.ask('Bring star up',np.zeros((512,512,3),dtype=np.uint8),{})
            self.assertFalse(c.history)
            log=json.loads(next(Path(directory).glob('request_*.json')).read_text())
            self.assertEqual(log['status'],'failed')

    def test_concurrent_call_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            def reserve(_):
                try:UsageLedger(directory,max_calls=2).reserve();return True
                except RuntimeError:return False
            with ThreadPoolExecutor(max_workers=6) as pool:
                accepted=list(pool.map(reserve,range(6)))
            self.assertEqual(sum(accepted),2)

if __name__=='__main__':unittest.main()
