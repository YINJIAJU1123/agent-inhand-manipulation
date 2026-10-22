"""Retry the unchanged frozen DLC package with bounded resumable download."""
import hashlib
import json
from pathlib import Path
import shlex
from bctrl.storage.config import ConfigStore
from bctrl.clients.dlc import DlcClient
from alibabacloud_pai_dlc20201203.models import CreateJobRequest
c=DlcClient(ConfigStore().load_required().model_copy(update={'region':'cn-wulanchabu'}),debug=False)
old=c.get_job('dlcd5ubyjup5tiz1')
assert old['Status']=='Failed'
parts=shlex.split(old['UserCommand']);assert parts[:2]==['bash','-lc'] and len(parts)==3
inner=parts[2]
start=inner.index('if command -v curl ');end=inner.index('fi',start)+2
fragment=inner[start:end]
tokens=shlex.split(fragment);i=tokens.index('curl',tokens.index('then')+1)
assert tokens[i:i+3]==['curl','-fsSL','-o']
target=tokens[i+3];url=tokens[i+4].rstrip(';')
# Keep the originally authorized owned OSS URL and package, without printing it.
assert url.startswith('http') and 'oss-' in url
retry=("download_ok=0; for attempt in 1 2 3 4 5 6; do "
       "echo PACKAGE_DOWNLOAD_ATTEMPT_$attempt; "
       f"if curl -fsSL --connect-timeout 20 --max-time 180 --speed-time 45 --speed-limit 1024 -C - -o {shlex.quote(target)} {shlex.quote(url)}; "
       "then download_ok=1; break; fi; sleep 3; done; "
       "test \"$download_ok\" = 1; echo PACKAGE_DOWNLOAD_COMPLETED")
inner=inner[:start]+'( '+retry+' )'+inner[end:]
allowed=['JobType','WorkspaceId','ResourceId','Priority','SchedulingStrategy','JobMaxRunningTimeMinutes','Envs','DataSources','JobSpecs']
request={k:old[k] for k in allowed if k in old}
request['DisplayName']='astra-transition-completion-r2-20261022'
request['Envs']=dict(old['Envs'],TRANSITION_COMPLETION_OUTPUT='/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_transition_completion_20261022/adaptation_r2')
request['UserCommand']='bash -lc '+shlex.quote(inner)
model=CreateJobRequest().from_map(request)
response=c.client.create_job(model).body.to_map()
jid=response['JobId']
record=dict(JobId=jid,Status='Submitted',previous_job='dlcd5ubyjup5tiz1',previous_failure='curl 18 incomplete package before entrypoint',package_unchanged=True,download_max_attempts=6,download_attempt_max_s=180,download_resume=True,output=request['Envs']['TRANSITION_COMPLETION_OUTPUT'],command_sha256=hashlib.sha256(request['UserCommand'].encode()).hexdigest())
Path('outputs/astra_transition_completion_v1/cloud_job_retry_20261022.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
