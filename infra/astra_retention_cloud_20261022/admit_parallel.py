"""Finite admission and submission tied to completed diagnostics; no Astra calls."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
out=ROOT/'outputs/astra_retention_cloud_20261022/parallel_submission'
out.mkdir(parents=True,exist_ok=False)
bctrl='/home/yin/.local/bin/bctrl';jobs=[]
state=dict(status='waiting_for_diagnostics',pid=os.getpid(),jobs=jobs,cloud_calls=0)
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state);print(json.dumps(state),flush=True)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try:
 start=time.time();dep=ROOT/'outputs/astra_retention_v1/counterfactual_single_20261022'
 while True:
  d=json.loads((dep/'progress.json').read_text())
  if d['status']=='completed':break
  if d['status']=='failed':raise RuntimeError('Local diagnostic failed')
  os.kill(1549939,0)
  if b'continue_local.py'not in Path('/proc/1549939/cmdline').read_bytes():raise RuntimeError('Dependency identity changed')
  progress(dependency_pid=1549939,dependency_status=d['status'])
  if time.time()-start>3600:raise TimeoutError('Diagnostic wait exceeded one hour')
  time.sleep(10)
 summary_path=dep/'branches/summary.json';summary=json.loads(summary_path.read_text());rows=summary['counterfactual_training_labels']
 gain=sum(r['best_quality']-r['continue_quality']for r in rows)/max(1,len(rows))
 exact=all(r['max_abs_error']<=1e-6 for r in summary['paired_state_audits'])
 if len(rows)<8 or gain<=.03 or not exact:
  progress(status='not_started',reason='Complete local diagnostic gate failed');sys.exit(0)
 compat_path=ROOT/'outputs/astra_retention_cloud_20261022/cloud_compatibility_result.json';compat=json.loads(compat_path.read_text())
 observed=json.loads(subprocess.run([bctrl,'--region','cn-wulanchabu','dlc','show',compat['verified_job_id'],'--json'],capture_output=True,text=True,timeout=45,check=True).stdout)
 assert observed['Status']=='Succeeded'and compat['matched_states']==12 and compat['max_state_error']==0 and compat['cloud_calls']==0
 gate=dict(full_local_diagnostic_passed=True,cloud_compatibility_passed=True,local_states=len(rows),local_mean_oracle_gain=gain,local_diagnostic_sha256=digest(summary_path),cloud_job_id=compat['verified_job_id'],cloud_result_sha256=digest(compat_path),cloud_calls=0,scope='Local counterfactual learning only. No new Astra budget or calls authorized by this gate.')
 stage=HERE/'code_train';save(stage/'launch_gate.json',gate);save(out/'launch_gate.json',gate)
 files={str(p.relative_to(stage)):digest(p)for p in stage.rglob('*')if p.is_file()and p.name!='package_manifest.json'and '__pycache__'not in p.parts}
 save(stage/'package_manifest.json',dict(sha256=files,cloud_calls=0,scope='Independent retention data and local integration only'))
 save(out/'package_manifest.json',json.loads((stage/'package_manifest.json').read_text()))
 for rank in range(3):
  progress(status='submitting',rank=rank)
  cmd=[bctrl,'--region','cn-wulanchabu','dlc','submit',str(HERE/f'train-p{rank}.yaml'),'--yes']
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
  (out/f'submit-p{rank}.log').write_text(result.stdout+'\n'+result.stderr)
  found=re.search(r'<(dlc[a-z0-9]+)>',result.stdout)
  if found:
   jobs.append(dict(rank=rank,job_id=found.group(1)));progress(jobs=jobs)
  if result.returncode or not found:raise RuntimeError('Cloud submit did not complete for partition '+str(rank))
 progress(status='submitted',rank=None,jobs=jobs)
except SystemExit:
 raise
except BaseException as e:
 progress(status='failed',error=str(e));save(out/'failure.json',dict(error=traceback.format_exc()))
 # A partial group cannot finish. Stop only this launch's recorded job IDs.
 for job in jobs:
  result=subprocess.run([bctrl,'--region','cn-wulanchabu','dlc','stop',job['job_id']],capture_output=True,text=True,timeout=45)
  save(out/('cleanup-'+job['job_id']+'.json'),dict(returncode=result.returncode))
 raise
