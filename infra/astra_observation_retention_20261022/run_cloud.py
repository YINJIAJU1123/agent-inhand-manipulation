"""One bounded GPU job to compare observation retention on recorded physical states."""
import hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
out = Path(os.environ['OBSERVATION_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
for part in ('logs', 'tmp', 'data'):
    (out / part).mkdir()
manifest = json.loads((ROOT / 'package_manifest.json').read_text())
for rel, digest in manifest['sha256'].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
plan = json.loads((ROOT / 'inputs/plan.json').read_text())
for row in plan:
    row['source'] = str(ROOT / 'inputs' / Path(row['source']).name)
save(out / 'plan.json', plan)
save(out / 'protocol.json', json.loads((ROOT / 'inputs/protocol.json').read_text()))
state = dict(status='starting', pid=os.getpid(), completed=[], expected_jobs=len(plan)*3, cloud_calls=0)
child = None
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0',
           TMPDIR=str(out / 'tmp'), PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'), str(ROOT/'outputs/learning_dependencies'), os.environ.get('PYTHONPATH', '')]))
def progress(**changes):
    state.update(changes, updated=time.time())
    save(out/'progress.json', state)
    print('OBSERVATION_PROGRESS '+json.dumps(state), flush=True)
try:
    subprocess.run([sys.executable,'-c','import isaaclab,toml,torch; print("CLOUD_CHILD_IMPORT_OK",flush=True)'],env=env,check=True)
    for item in plan:
        for rate in (1., .25, 0.):
            name = Path(item['source']).name + f"_case{item['case']}_rate{rate}"
            dest = out/'data'/name
            command = [sys.executable,str(ROOT/'scripts/astra_retention_v1/branch.py'),'--source',item['source'],
                       '--plan',str(out/'plan.json'),'--case',str(item['case']),'--output',str(dest),
                       '--rate',str(rate),'--headless','--device','cuda:0'] + KIT
            with (out/'logs'/(name+'.log')).open('w') as log:
                child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start = time.time()
            while child.poll() is None:
                path = dest/'progress.json'
                measured = json.loads(path.read_text()) if path.exists() else None
                progress(status='branching',current=name,child_pid=child.pid,elapsed_s=time.time()-start,child_progress=measured)
                if (dest/'failure.json').exists():
                    raise RuntimeError((dest/'failure.json').read_text())
                if time.time()-start>1200 or path.exists() and time.time()-path.stat().st_mtime>300:
                    raise TimeoutError(name)
                time.sleep(10)
            if child.returncode or not (dest/'report.json').exists():
                raise RuntimeError((out/'logs'/(name+'.log')).read_text()[-10000:])
            report = json.loads((dest/'report.json').read_text())
            print('OBSERVATION_BRANCH '+json.dumps(dict(job=name,results=report['results'])),flush=True)
            progress(completed=state['completed']+[name],child_pid=None)
    subprocess.run([sys.executable,str(ROOT/'scripts/astra_retention_v1/analyze_branches.py'),'--output',str(out)],check=True,env=env)
    raw = json.loads((out/'summary.json').read_text())
    rates = {rate:{k:v for k,v in row.items() if k not in ('correct_binding_branches','semantically_correct_final')} for rate,row in raw['rates'].items()}
    result = dict(rates=rates,paired_state_audits=raw['paired_state_audits'],cloud_calls=0,
                  selected_cases=len(plan),source_sha256=manifest['sha256']['scripts/astra_retention_v1/branch.py'],
                  note='Six-second retention of the commanded observation color. Does not establish task grounding, glyph readability, or online success. Historical replies after intervention are not applied.')
    save(out/'observation_result.json',result)
    print('OBSERVATION_RESULT '+json.dumps(result),flush=True)
    progress(status='completed',current=None,child_pid=None)
except BaseException as error:
    save(out/'failure.json',dict(error=traceback.format_exc()))
    progress(status='failed',error=str(error))
    raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
