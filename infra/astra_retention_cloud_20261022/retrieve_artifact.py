"""Retrieve this experiment's bounded model bundle from authenticated DLC logs."""
import argparse,base64,hashlib,io,json,re,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--output',required=True);a=p.parse_args()
output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
cmd=['/home/yin/.local/bin/bctrl','--region','cn-wulanchabu','dlc','logs',a.job_id,'--no-follow','--lines','5000']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=60,check=True)
text=re.sub(r'\x1b\[[0-9;]*m','',r.stdout)
start='RETENTION_ARTIFACT_BEGIN';end='RETENTION_ARTIFACT_END'
assert text.count(start)==1 and text.count(end)==1,'Expected one completed training artifact'
encoded=text.split(start,1)[1].split(end,1)[0]
encoded=''.join(encoded.split());data=base64.b64decode(encoded,validate=True)
assert len(data)<2000000
expected=re.search(r'RETENTION_ARTIFACT_SHA256\s+([a-f0-9]{64})',text)
assert expected and hashlib.sha256(data).hexdigest()==expected.group(1),'Artifact checksum mismatch'
allowed={'head.pt','result.json','protocol.json','history.json'}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as archive:
 members=archive.getmembers()
 assert {m.name for m in members}==allowed
 assert all(m.isfile()and m.size<2000000 for m in members)
 for m in members:(output/m.name).write_bytes(archive.extractfile(m).read())
result=json.loads((output/'result.json').read_text())
assert hashlib.sha256((output/'head.pt').read_bytes()).hexdigest()==result['checkpoint_sha256']
(output/'retrieval.json').write_text(json.dumps(dict(job_id=a.job_id,artifact_sha256=expected.group(1),bytes=len(data),verified=True),indent=2)+'\n')
print(json.dumps(dict(job_id=a.job_id,verified=True,offline_gate_passed=result['offline_gate_passed'],output=str(output.resolve())),indent=2))
