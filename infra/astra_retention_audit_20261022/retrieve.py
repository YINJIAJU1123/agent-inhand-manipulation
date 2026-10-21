"""Download bounded, checksummed diagnostic artifacts from the owned DLC jobs."""
import argparse,base64,gzip,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--kind',choices=['features','preview'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
marker={'features':'RETENTION_FEATURES','preview':'SYMBOL_REGION_PREVIEW'}[a.kind]
r=subprocess.run(['/home/yin/.local/bin/bctrl','--region','cn-wulanchabu','dlc','logs',a.job_id,'--no-follow','--lines','10000'],capture_output=True,text=True,timeout=60,check=True)
text=re.sub(r'\x1b\[[0-9;]*m','',r.stdout)
assert text.count(marker+'_BEGIN')==1 and text.count(marker+'_END')==1,'Artifact has not completed'
encoded=''.join(text.split(marker+'_BEGIN',1)[1].split(marker+'_END',1)[0].split())
payload=base64.b64decode(encoded,validate=True);assert len(payload)<2000000
expected=re.search(marker+r'_SHA256\s+([a-f0-9]{64})',text)
assert expected and hashlib.sha256(payload).hexdigest()==expected.group(1),'Checksum mismatch'
data=gzip.decompress(payload)if a.kind=='features'else payload
assert len(data)<2000000
if a.kind=='preview':assert data.startswith(b'\x89PNG\r\n\x1a\n')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(data)
a.output.with_suffix(a.output.suffix+'.retrieval.json').write_text(json.dumps(dict(job_id=a.job_id,kind=a.kind,sha256=hashlib.sha256(data).hexdigest(),transport_sha256=expected.group(1),bytes=len(data)),indent=2)+'\n')
print(json.dumps(dict(verified=True,path=str(a.output.resolve()),bytes=len(data))))
