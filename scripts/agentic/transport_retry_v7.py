"""Bounded retries of explicitly reported transport failures only.

Each retry calls the original client's audited/budgeted ask. No TLS verification
or authentication settings are changed; high-level state changes only on success.
"""
import time
TRANSPORT_REASONS=frozenset({'SSLError','ConnectionError','ConnectTimeout','ReadTimeout','Timeout','ChunkedEncodingError'})
def ask_with_retry(client, *args, attempts=1, backoff=1.0):
    if not 1<=attempts<=2:raise ValueError('At most two transport attempts')
    failures=[]
    for index in range(attempts):
        try:
            result=client.ask(*args)
            result['transport_attempts']=index+1
            result['transport_retry_errors']=failures
            return result
        except RuntimeError as error:
            if str(error) not in TRANSPORT_REASONS or index+1>=attempts:raise
            failures.append(str(error));time.sleep(backoff)
