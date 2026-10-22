"""Finite asynchronous completion-data collection and visual-head adaptation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha,STUDENT,ADAPTER
from prepare import KIT
out=Path(os.environ['TRANSITION_COMPLETION_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
for name in ('data','cache','logs','tmp'): (out/name).mkdir()
manifest=json.loads((ROOT/'package_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    if sha(ROOT/name)!=digest:raise RuntimeError('Package changed: '+name)
schedule=[('train',s) for s in range(121101,121109)]+[('calibration',121201),('audit',121202)]
save(out/'protocol.json',dict(schedule=schedule,cases_per_seed=12,max_steps=1200,epochs=30,cloud_calls=0,
    frozen_package=manifest,goal='Adapt visual pose completion to wait/goal-switch/reply transitions. Keep original student/V3 frozen.',
    split='Eight new training seeds with old fixed-goal training anchors; one new calibration and one new audit seed. Do not tune on audit.',
    gate='At least eight first-stop proposals with >=90% precision and >=30% reached recall; fewer false first stops and no fewer correct stops than original dense verifier on the same new audit.',
    limits='Scripted colors, no semantic task success or automatic deployment. Independent live verification required.'))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',TMPDIR=str(out/'tmp'),
    PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),str(ROOT/'outputs/learning_dependencies'),os.environ.get('PYTHONPATH','')]))
state=dict(status='starting',completed=[],cloud_calls=0);child=None;started=time.monotonic()

def progress(**kw):
    state.update(kw,updated=time.time());save(out/'progress.json',state)
    print('TRANSITION_COMPLETION_PROGRESS '+json.dumps(state),flush=True)

def run(command,name,dest,timeout):
    global child
    save(out/'logs'/(name+'_command.json'),command)
    with (out/'logs'/(name+'.log')).open('w') as f:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    start=time.monotonic()
    while child.poll() is None:
        p=dest/'progress.json'
        progress(status='training' if name=='train_dense' else 'collecting',current=name,child_pid=child.pid,
            child_progress=json.loads(p.read_text()) if p.exists() else None,elapsed_s=time.monotonic()-start)
        if time.monotonic()-start>timeout or time.monotonic()-started>3600:raise TimeoutError('Bounded collection/training deadline')
        time.sleep(15)
    if child.returncode:raise RuntimeError(name+' exited '+str(child.returncode)+'; '+(out/'logs'/(name+'.log')).read_text()[-1200:])
    progress(completed=state['completed']+[name],child_pid=None)

try:
    cache=[]
    for split,seed in schedule:
        name=f'{split}_{seed}';dest=out/'data'/name
        run([sys.executable,str(ROOT/'scripts/astra_transition_completion_v1/collect.py'),'--output',str(dest),
            '--seed',str(seed),'--split',split,'--cases','12','--max-steps','1200','--headless','--device','cuda:0']+KIT,name,dest,1500)
        metadata=json.loads((dest/'manifest.json').read_text());assert metadata['status']=='completed' and metadata['student_sha256']==sha(STUDENT)
        assert metadata['data_sha256']==sha(dest/'data.pt')
        target=out/'cache'/(name+'.pt');target.symlink_to(dest/'data.pt')
        cache.append(dict(shard=name,frames=metadata['frames'],data_sha256=metadata['data_sha256'],ready_counts=metadata['ready_counts'],near_counts=metadata['near_counts']))
    anchor=ROOT/'anchor.pt';am=json.loads((ROOT/'anchor_manifest.json').read_text());assert sha(anchor)==am['sha256']
    shutil.copy2(anchor,out/'cache/anchor.pt')
    save(out/'cache/manifest.json',dict(status='completed',shards=cache,anchor=am,student_sha256=sha(STUDENT),adapter_sha256=sha(ADAPTER),
        sensor_fields=['x','roi_rgb','roi_depth','roi_box','roi_found'],supervision_only_fields=['error','rotation_label'],cloud_calls=0))
    run([sys.executable,str(ROOT/'scripts/astra_transition_completion_v1/train_dense.py'),'--cache',str(out/'cache'),'--output',str(out/'training'),
        '--initial',str(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'),'--epochs','30','--device','cuda:0'], 'train_dense',out/'training',1500)
    result=json.loads((out/'training/result.json').read_text())
    for name in ('audit','baseline_audit'):
        result[name]={k:v for k,v in result[name].items() if k!='details'}
    # Small summary per line. Detailed data stay on the owned NAS.
    summary=dict(status='completed',result={k:result[k] for k in ('status','best_epoch','threshold','audit','baseline_audit','offline_gate_passed','checkpoint_sha256')},
        cloud_calls=0,deployed=False,cache=cache)
    save(out/'result.json',summary)
    print('TRANSITION_COMPLETION_RESULT '+json.dumps({k:v for k,v in summary.items() if k!='cache'}),flush=True)
    progress(status='completed',current=None,child_pid=None,offline_gate_passed=result['offline_gate_passed'])
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=traceback.format_exc()[-1400:]);raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
