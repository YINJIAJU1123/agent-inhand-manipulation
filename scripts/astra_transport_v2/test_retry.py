import unittest
from retry import ask_with_retry
class Client:
    def __init__(self,failures):self.failures=failures;self.calls=0
    def ask(self,*args):
        self.calls+=1
        if self.calls<=len(self.failures):raise RuntimeError(self.failures[self.calls-1])
        return dict(action='unchanged')
class Tests(unittest.TestCase):
    def test_proxy_and_tls_retry_once(self):
        for reason in ['ProxyError','SSLError','ReadTimeout']:
            c=Client([reason]);r=ask_with_retry(c,attempts=2,backoff=0)
            self.assertEqual(c.calls,2);self.assertEqual(r['transport_retry_errors'],[reason]);self.assertEqual(r['action'],'unchanged')
    def test_does_not_retry_auth_budget_or_model_errors(self):
        for reason in ['http_401_invalid_token','experiment_call_cap_reached','expected_one_robot_tool_call']:
            c=Client([reason])
            with self.assertRaises(RuntimeError):ask_with_retry(c,attempts=2,backoff=0)
            self.assertEqual(c.calls,1)
    def test_exhaustion_is_reported(self):
        c=Client(['SSLError']*3)
        with self.assertRaises(RuntimeError):ask_with_retry(c,attempts=2,backoff=0)
        self.assertEqual(c.calls,2)
if __name__=='__main__':unittest.main()
