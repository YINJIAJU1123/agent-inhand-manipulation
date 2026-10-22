import base64,hashlib,json,shlex
from pathlib import Path
from bctrl.storage.config import ConfigStore
from bctrl.clients.dlc import DlcClient
from alibabacloud_pai_dlc20201203.models import CreateJobRequest
root=Path(__file__).resolve().parents[2]
smoke=json.loads((root/'outputs/astra_goal_maintenance_v1/smoke_20261022/train/result.json').read_text());assert smoke['status']=='completed' and smoke['final']['accepted_policy_steps']>0
checks=json.loads((root/'outputs/astra_goal_maintenance_v1/smoke_20261022/replay_check.json').read_text());assert checks['max_difference']==0
c=DlcClient(ConfigStore().load_required().model_copy(update={'region':'cn-wulanchabu'}),debug=False)
old=c.get_job('dlcp3fojv1xxvzd2');assert old['Status']=='Succeeded';inner=shlex.split(old['UserCommand'])[2];assert inner.endswith('bash entrypoint.sh')
files={'run_cloud.py':Path(__file__).with_name('run_cloud.py')}
for name in ['train.py','model.py']:files['scripts/astra_goal_maintenance_v1/'+name]=root/'scripts/astra_goal_maintenance_v1'/name
payload={name:base64.b64encode(path.read_bytes()).decode() for name,path in files.items()}
installer='import pathlib,base64,json;data=json.loads('+repr(json.dumps(payload))+');[(pathlib.Path(k).parent.mkdir(parents=True,exist_ok=True),pathlib.Path(k).write_bytes(base64.b64decode(v))) for k,v in data.items()]'
inner=inner[:-len('bash entrypoint.sh')]+'python3 -c '+shlex.quote(installer)+' && bash entrypoint.sh'
keys=['JobType','WorkspaceId','ResourceId','Priority','SchedulingStrategy','Envs','DataSources','JobSpecs'];r={k:old[k] for k in keys if k in old}
r.update(DisplayName='astra-goal-maintenance-r1-20261022',JobMaxRunningTimeMinutes=80,UserCommand='bash -lc '+shlex.quote(inner))
r['Envs']={k:v for k,v in r['Envs'].items() if k!='TRANSITION_COMPLETION_OUTPUT'}
jid=c.client.create_job(CreateJobRequest().from_map(r)).body.to_map()['JobId']
record=dict(job_id=jid,status='Submitted',gpu=1,max_running_minutes=80,source_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},cloud_calls=0)
(root/'outputs/astra_goal_maintenance_v1/cloud_job_20261022.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
