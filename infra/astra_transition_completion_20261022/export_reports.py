"""Read-only export of this run's JSON reports; no images, weights or credentials."""
import base64,gzip,hashlib,json,time
from pathlib import Path
root=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_transition_completion_20261022/adaptation_r2')
files={}
for relative in ['result.json','protocol.json','training/result.json','training/protocol.json','training/history.json','cache/manifest.json']:
 content=(root/relative).read_bytes();assert len(content)<2_000_000
 files[relative]=dict(text=content.decode(),sha256=hashlib.sha256(content).hexdigest())
blob=gzip.compress(json.dumps(dict(files=files,source=str(root))).encode(),mtime=0);encoded=base64.b64encode(blob).decode()
chunks=[encoded[i:i+1024] for i in range(0,len(encoded),1024)]
for i,chunk in enumerate(chunks):
 print(json.dumps(dict(kind='transition_report_chunk',index=i,count=len(chunks),transport_sha256=hashlib.sha256(blob).hexdigest(),chunk_sha256=hashlib.sha256(chunk.encode()).hexdigest(),data=chunk)),flush=True);time.sleep(.1)
print(json.dumps(dict(status='completed',files=len(files),compressed_bytes=len(blob),cloud_calls=0)),flush=True)
