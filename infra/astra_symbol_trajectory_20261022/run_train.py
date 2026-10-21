"""Train one predeclared RGB variant from completed, verified trajectory shards."""
import base64, hashlib, io, json, os, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
out = Path(os.environ['VERIFIED_OUTPUT']); out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((ROOT/'package_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
base = Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_symbol_invariant_20261022/learning_r1/train/head.pt')
assert hashlib.sha256(base.read_bytes()).hexdigest() == 'e7443b94238c178c70f82c8094148c8ee864bbd3a6c2a27c3aa3b3b882380e41'
data_root = Path(os.environ['VERIFIED_DATA_ROOT'])
variant = os.environ['VERIFIED_VARIANT']; assert variant in ('invariant','verified')
expected = {106201,106202,106203,106204,106205,106206,106301,106302}
shards=[]
for rank in range(2):
    result=json.loads((data_root/f'rank{rank}/result.json').read_text())
    assert result['status']=='completed' and result['cloud_calls']==0
    shards+=result['shards']
assert {s['seed'] for s in shards} == expected and len(shards)==8
assert all(s['replay_max_error']<=1e-6 for s in shards)
for split in ('train','calibration','audit'):
    assert sum(s['phase_counts'].get('execute',0) for s in shards if s['split']==split)>0
state=dict(status='starting',variant=variant,cloud_calls=0); child=None

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def progress(**values):
    state.update(values,updated=time.time());save(out/'progress.json',state)
    print('SYMBOL_REGION_PROGRESS '+json.dumps(state),flush=True)

try:
    save(out/'source_manifest.json',manifest)
    save(out/'data_manifest.json',dict(shards=shards,variant=variant,cloud_calls=0))
    command=[sys.executable,str(ROOT/'scripts/astra_symbol_verified_v1/train.py'),'--data',
        str(data_root/'rank0/data'),str(data_root/'rank1/data'),'--base',str(base),
        '--output',str(out/'train'),'--variant',variant,'--epochs','50']
    with (out/'train.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.time()
    while child.poll() is None:
        path=out/'train/progress.json';measured=json.loads(path.read_text()) if path.exists() else None
        progress(status='training',child_pid=child.pid,child_progress=measured,elapsed_s=time.time()-start)
        if (out/'train/failure.json').exists():raise RuntimeError((out/'train/failure.json').read_text())
        if time.time()-start>5400:raise TimeoutError('Bounded trajectory-symbol training')
        time.sleep(15)
    if child.returncode or not (out/'train/result.json').exists():raise RuntimeError((out/'train.log').read_text()[-10000:])
    result=json.loads((out/'train/result.json').read_text())
    print('SYMBOL_REGION_RESULT '+json.dumps(result),flush=True)
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name in ['head.pt','result.json','protocol.json','history.json']:archive.add(out/'train'/name,arcname=name)
    payload=stream.getvalue();assert len(payload)<3000000
    print('SYMBOL_MODEL_SHA256 '+hashlib.sha256(payload).hexdigest(),flush=True);print('SYMBOL_MODEL_BEGIN',flush=True)
    encoded=base64.b64encode(payload).decode()
    for i in range(0,len(encoded),64):print(encoded[i:i+64],flush=True)
    print('SYMBOL_MODEL_END',flush=True);progress(status='completed',child_pid=None)
except BaseException as error:
    save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=str(error));raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
