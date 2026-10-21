"""Read one finite probe's NAS reports without truncating long DLC log lines."""
import base64,gzip,hashlib,json,os,time
from pathlib import Path
source=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_observation_control_20261022/two_seed_r1')
out=Path(os.environ['REPORT_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
start=time.time();reported=set()
while not(source/'result.json').exists():
 if(source/'failure.json').exists():raise RuntimeError((source/'failure.json').read_text())
 for path in sorted(source.glob('*/report.json')):
  if path.parent.name in reported:continue
  row=json.loads(path.read_text());reported.add(path.parent.name)
  print('OBS_REPORT_RUN '+json.dumps({k:v for k,v in row.items()if k!='episodes'}),flush=True)
 if time.time()-start>1200:raise TimeoutError('Report wait ended without inferring job termination')
 time.sleep(10)
result=json.loads((source/'result.json').read_text())
events={path.parent.name:json.loads(path.read_text())for path in source.glob('*/retention_events.json')}
for row in result['runs']:
 print('OBS_REPORT_RUN '+json.dumps({k:v for k,v in row.items()if k!='episodes'}),flush=True)
full=dict(result=result,events=events,cloud_calls=0,source=str(source))
(out/'report.json').write_text(json.dumps(full,indent=2)+'\n');payload=gzip.compress(json.dumps(full).encode())
print('OBS_REPORT_SHA256 '+hashlib.sha256(payload).hexdigest(),flush=True);print('OBS_REPORT_BEGIN',flush=True)
s=base64.b64encode(payload).decode()
for i in range(0,len(s),64):print(s[i:i+64],flush=True)
print('OBS_REPORT_END',flush=True)
