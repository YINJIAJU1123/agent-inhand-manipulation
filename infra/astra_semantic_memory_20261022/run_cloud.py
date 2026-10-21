"""Two fresh frozen-weight sequence audits; no API or online deployment."""
import base64,gzip,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
out=Path(os.environ['MEMORY_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
for name in ('tmp','logs','rollouts','evaluation'):(out/name).mkdir()
manifest=json.loads((ROOT/'package_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Package mismatch '+name)
checkpoint=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_symbol_trajectory_20261022/verified_train_r1/train/head.pt')
digest='ff61173f56b3ba0c26481e1b9037f8e1a1307241077488415cae74dbd82a1f87'
if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen verified checkpoint mismatch')
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',TMPDIR=str(out/'tmp'),
    PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),str(ROOT/'outputs/learning_dependencies'),os.environ.get('PYTHONPATH','')]))
state=dict(status='starting',completed=[],cloud_calls=0);child=None

def progress(**kw):
    state.update(kw,updated=time.time());save(out/'progress.json',state)
    print('MEMORY_PROGRESS '+json.dumps(state),flush=True)

def run(name,command,target):
    global child
    save(out/'logs'/(name+'_command.json'),command)
    with (out/'logs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    start=time.monotonic()
    while child.poll() is None:
        status=target.parent/'progress.json'
        progress(status='running',current=name,child_pid=child.pid,child_progress=json.loads(status.read_text()) if status.exists() else None,elapsed_s=time.monotonic()-start)
        if time.monotonic()-start>1800:raise TimeoutError(name+' exceeded 30 minutes')
        if (target.parent/'failure.json').exists():raise RuntimeError((target.parent/'failure.json').read_text())
        time.sleep(15)
    if child.returncode or not target.exists():raise RuntimeError((out/'logs'/(name+'.log')).read_text()[-8000:])
    progress(completed=state['completed']+[name],child_pid=None)

try:
    save(out/'protocol.json',dict(seeds=[106401,106402],permutations='Fixed shuffle 31990, indices 600:720 unused by train/calibration/audit',stride=10,max_steps=1800,
        control='Original fixed public-color schedule and observation/execution hold; semantic predictions do not affect actions',checkpoint_sha256=digest,cloud_calls=0,package=manifest))
    results=[]
    for seed in (106401,106402):
        source=out/'rollouts'/str(seed);dest=out/'evaluation'/str(seed)
        run(str(seed)+'_collect',[sys.executable,str(ROOT/'scripts/astra_semantic_memory_v1/collect.py'),
            '--seed',str(seed),'--split','fresh','--mode','all','--stride','10','--max-steps','1800','--output',str(source),'--headless','--device','cuda:0']+KIT,source/'report.json')
        run(str(seed)+'_evaluate',[sys.executable,str(ROOT/'scripts/astra_semantic_memory_v1/evaluate.py'),
            '--source',str(source),'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--output',str(dest),'--device','cuda:0'],dest/'result.json')
        results.append(json.loads((dest/'result.json').read_text()));save(out/'partial_results.json',results)
    result=dict(status='completed',results=results,cloud_calls=0,checkpoint_sha256=digest)
    save(out/'result.json',result)
    blob=gzip.compress(json.dumps(result).encode(),mtime=0)
    print('MEMORY_RESULT_SHA256 '+hashlib.sha256(blob).hexdigest(),flush=True)
    print('MEMORY_RESULT_BEGIN',flush=True)
    encoded=base64.b64encode(blob).decode()
    for i in range(0,len(encoded),64):print(encoded[i:i+64],flush=True)
    print('MEMORY_RESULT_END',flush=True)
    progress(status='completed',current=None,child_pid=None)
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=traceback.format_exc());raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
