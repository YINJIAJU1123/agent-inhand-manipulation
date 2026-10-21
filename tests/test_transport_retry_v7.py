import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/agentic'))
from transport_retry_v7 import ask_with_retry
class Client:
    def __init__(self,errors):self.errors=iter(errors);self.calls=0
    def ask(self,*args):
        self.calls+=1;error=next(self.errors,None)
        if error:raise RuntimeError(error)
        return {'tool':'inspect_face'}
class RetryTests(unittest.TestCase):
    def test_transient_and_budgeted_attempts(self):
        client=Client(['SSLError',None]);r=ask_with_retry(client,attempts=2,backoff=0)
        self.assertEqual(client.calls,2);self.assertEqual(r['transport_retry_errors'],['SSLError'])
    def test_bounded_failure(self):
        client=Client(['SSLError']*3)
        with self.assertRaisesRegex(RuntimeError,'SSLError'):ask_with_retry(client,attempts=2,backoff=0)
        self.assertEqual(client.calls,2)
    def test_budget_auth_and_bad_output_not_retried(self):
        for reason in ['experiment_call_cap_reached','http_401_api_error','expected_one_robot_tool_call']:
            client=Client([reason])
            with self.assertRaisesRegex(RuntimeError,reason):ask_with_retry(client,attempts=2,backoff=0)
            self.assertEqual(client.calls,1)
if __name__=='__main__':unittest.main()
