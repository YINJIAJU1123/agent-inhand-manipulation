"""CPU-only finite post-training audit on the existing NAS artifacts."""
import base64,gzip,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=Path(os.environ['AUDIT_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
training=Path(os.environ['RETENTION_TRAINING']);parent=training.parent
start=time.time()
def progress(**state):
    state.update(updated=time.time(),cloud_calls=0);(out/'progress.json').write_text(json.dumps(state,indent=2)+'\n');print('RETENTION_AUDIT_PROGRESS '+json.dumps(state),flush=True)
try:
    while not(training/'result.json').exists():
        failures=[p for p in parent.glob('worker*/failure.json')]
        if failures:raise RuntimeError('Training worker failed: '+str(failures))
        if time.time()-start>3600:raise TimeoutError('Finite wait for existing training result')
        progress(status='waiting_for_training',dependency=str(training));time.sleep(30)
    config=json.loads((ROOT/'audit_config.json').read_text())
    subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--training',str(training),'--output',str(out/'runtime_cohort.json'),'--threshold',str(config['threshold']),'--base-sha256',config['base_sha256']],check=True)
    payload=gzip.compress((training/'features.pt').read_bytes());assert len(payload)<1000000
    print('RETENTION_FEATURES_SHA256 '+hashlib.sha256(payload).hexdigest(),flush=True)
    print('RETENTION_FEATURES_BEGIN',flush=True);encoded=base64.b64encode(payload).decode()
    for i in range(0,len(encoded),64):print(encoded[i:i+64],flush=True)
    print('RETENTION_FEATURES_END',flush=True);progress(status='completed')
except BaseException:
    (out/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc())));progress(status='failed');raise
