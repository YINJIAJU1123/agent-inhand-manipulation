import base64,hashlib,json,shlex
from pathlib import Path
from bctrl.storage.config import ConfigStore
from bctrl.clients.dlc import DlcClient
from alibabacloud_pai_dlc20201203.models import CreateJobRequest
c=DlcClient(ConfigStore().load_required().model_copy(update={'region':'cn-wulanchabu'}),debug=False)
old=c.get_job('dlcp3fojv1xxvzd2');assert old['Status']=='Succeeded'
inner=shlex.split(old['UserCommand'])[2];assert inner.endswith('bash entrypoint.sh')
source=Path(__file__).with_name('recalibrate.py').read_bytes();payload=base64.b64encode(source).decode()
# Reuse only the same owned code package, mounts, and one GPU. No new API credentials.
install="python3 -c "+shlex.quote("import base64,pathlib;pathlib.Path('run_cloud.py').write_bytes(base64.b64decode('"+payload+"'))")
inner=inner[:-len('bash entrypoint.sh')]+install+' && bash entrypoint.sh'
keys=['JobType','WorkspaceId','ResourceId','Priority','SchedulingStrategy','Envs','DataSources','JobSpecs']
r={k:old[k] for k in keys if k in old};r.update(DisplayName='astra-transition-recalibration-r2-20261022',JobMaxRunningTimeMinutes=25,UserCommand='bash -lc '+shlex.quote(inner))
r['Envs']={k:v for k,v in r['Envs'].items() if k!='TRANSITION_COMPLETION_OUTPUT'}
jid=c.client.create_job(CreateJobRequest().from_map(r)).body.to_map()['JobId']
record=dict(job_id=jid,status='Submitted',parent_job='dlcp3fojv1xxvzd2',source_sha256=hashlib.sha256(source).hexdigest(),gpu=1,max_running_minutes=25,cloud_calls=0)
Path('outputs/astra_transition_completion_v1/recalibration_job_retry_20261022.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
