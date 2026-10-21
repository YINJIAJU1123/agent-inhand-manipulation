"""Bounded native-RGB data smoke. Does not train or call a remote model."""
import base64, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
out=Path(os.environ['SYMBOL_REGION_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
(out/'tmp').mkdir()
manifest=json.loads((ROOT/'package_manifest.json').read_text())
for rel,digest in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
state=dict(status='starting',pid=os.getpid(),cloud_calls=0)
def progress(**values):
    state.update(values,updated=time.time());save(out/'progress.json',state)
    print('SYMBOL_REGION_PROGRESS '+json.dumps(state),flush=True)
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',TMPDIR=str(out/'tmp'),PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),str(ROOT/'outputs/learning_dependencies'),os.environ.get('PYTHONPATH','')]))
child=None
try:
    subprocess.run([sys.executable,'-c','import isaaclab,toml,torch; print("CLOUD_CHILD_IMPORT_OK",flush=True)'],check=True,env=env)
    dest=out/'data'
    command=[sys.executable,str(ROOT/'scripts/astra_symbol_regions_v1/collect.py'),'--source',str(ROOT/'inputs/train_96111'),'--output',str(dest),'--seed','96111','--split','train','--max-steps','60','--views','6','--headless','--device','cuda:0']+KIT
    save(out/'command.json',command)
    with(out/'collect.log').open('w')as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.time()
    while child.poll()is None:
        path=dest/'progress.json';measured=json.loads(path.read_text())if path.exists()else None
        progress(status='collecting',child_pid=child.pid,elapsed_s=time.time()-start,child_progress=measured)
        if(dest/'failure.json').exists():raise RuntimeError((dest/'failure.json').read_text())
        if time.time()-start>1200:raise TimeoutError('collection smoke')
        time.sleep(10)
    if child.returncode or not(dest/'manifest.json').exists():raise RuntimeError((out/'collect.log').read_text()[-12000:])
    result=json.loads((dest/'manifest.json').read_text());assert result['views']>0 and result['readable']>0
    save(out/'result.json',result);print('SYMBOL_REGION_RESULT '+json.dumps(result),flush=True)
    preview=(dest/'label_preview.png').read_bytes();assert len(preview)<2000000
    print('SYMBOL_REGION_PREVIEW_SHA256 '+hashlib.sha256(preview).hexdigest(),flush=True)
    print('SYMBOL_REGION_PREVIEW_BEGIN',flush=True)
    encoded=base64.b64encode(preview).decode()
    for i in range(0,len(encoded),64):print(encoded[i:i+64],flush=True)
    print('SYMBOL_REGION_PREVIEW_END',flush=True)
    progress(status='completed',child_pid=None)
except BaseException as error:
    save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=str(error));raise
finally:
    if child and child.poll()is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
