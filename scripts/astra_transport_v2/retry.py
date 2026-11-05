"""At most two audited attempts, restricted to transport failures.

The caller measures the entire operation, including failed attempts/backoff.
No changes to TLS verification, proxy settings, credentials or robot history.
"""
import time
TRANSPORT_REASONS=frozenset({'SSLError','ProxyError','ConnectionError','ConnectTimeout','ReadTimeout','Timeout','ChunkedEncodingError'})
def ask_with_retry(client,*args,attempts=2,backoff=1.):
    if not 1<=attempts<=2:raise ValueError('At most two attempts')
    failures=[]
    for i in range(attempts):
        try:
            r=client.ask(*args);r['transport_attempts']=i+1;r['transport_retry_errors']=failures
            return r
        except RuntimeError as e:
            if str(e) not in TRANSPORT_REASONS or i+1>=attempts:raise
            failures.append(str(e));time.sleep(backoff)
