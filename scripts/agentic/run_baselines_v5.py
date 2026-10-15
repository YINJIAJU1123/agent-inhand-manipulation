"""Frozen, resumable local evaluation with an owned VLM service and watchdog.

This process runs a finite experiment. It creates no scheduled automation.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CRITICAL = ['run_baselines_v5.py', 'analyze_baselines_v5.py', 'evaluate_baselines_v5.py', 'rule_client_v5.py',
            'vlm_server.py', 'semantic_agent_v3.py', 'semantic_protocol_v3.py', 'visual_agent.py',
            'student_features.py', 'symbol_scene.py', 'rollout_video.py', 'agent_observation_v4.py',
            'latent_goal_v4.py', 'latent_supervisor_v4.py', 'latent_data_v4.py', 'astra_agent_v4.py']
STUDENT = ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'
ADAPTER = ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--protocol', default=str(ROOT/'configs/agentic_baselines_v5_large.json'))
    p.add_argument('--rule-model', default='/home/yin/Grasping/Grasppose/run/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--latent-checkpoint', default=str(ROOT/'outputs/agentic_v4/checkpoints/latent_v2'))
    p.add_argument('--port', type=int, default=18751)
    a = p.parse_args()
    dest = Path(a.output).resolve(); dest.mkdir(parents=True, exist_ok=True)
    lock = (dest/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(Path(a.protocol).read_text())
    model = Path(a.rule_model).resolve(); latent = Path(a.latent_checkpoint).resolve()
    sources = lambda: {name: sha(ROOT/'scripts/agentic'/name) for name in CRITICAL}
    weights = [STUDENT, ADAPTER, latent/'goal_head.pt', latent/'lora/adapter_model.safetensors']
    frozen_weights = {str(path): sha(path) for path in weights}
    freeze = {'protocol': protocol, 'source_sha256': sources(), 'weights': frozen_weights,
              'rule_model': str(model), 'rule_model_config_sha256': sha(model/'config.json'),
              'rule_model_weights': {x.name:sha(x) for x in sorted(model.glob('*.safetensors'))},
              'latent_checkpoint': str(latent), 'symbol_assets': {x.name:sha(x) for x in sorted((ROOT/'assets/usd/semantic_symbols').glob('*.usda'))}}
    freeze_path = dest/'freeze.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != freeze: raise RuntimeError('Resume rejected: frozen code/protocol/weights changed')
    else:
        write(freeze_path, freeze)
        (dest/'sources').mkdir(exist_ok=True)
        for name in CRITICAL: (dest/'sources'/name).write_bytes((ROOT/'scripts/agentic'/name).read_bytes())
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = ':'.join(str(ROOT/'outputs'/name) for name in ['vlm_dependencies','learning_dependencies','video_dependencies'])
    address = f'http://127.0.0.1:{a.port}'
    def health():
        try:
            with urllib.request.urlopen(address, timeout=2) as f: return json.load(f)
        except Exception: return None
    state = {'pid':os.getpid(), 'status':'starting', 'completed_jobs':0,
             'total_jobs':len(protocol['scenarios'])*len(protocol['modes']), 'planned_episodes':protocol['total_episodes']}
    def progress(**fields):
        state.update(fields); state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(dest/'progress.json', state)
    def validate(report, scenario, mode):
        m = report['metadata']; episodes = report['episodes']
        assert len(episodes)==6 and {e['case'] for e in episodes}==set(range(6))
        for key, expected in [('seeds',[scenario['seed']]),('modes',[mode]),('template_set',scenario['template_set']),
                              ('instruction_offset',scenario['instruction_offset']),('call_budget',protocol['call_budget']),('max_steps',protocol['max_steps'])]:
            assert m[key]==expected, (key,m[key],expected)
        assert all(freeze['source_sha256'][key]==value for key,value in m['source_sha256'].items())
        assert m['checkpoint_sha256']==frozen_weights[str(STUDENT)] and m['adapter_sha256']==frozen_weights[str(ADAPTER)]
        if mode=='latent':
            assert m['latent_checkpoint_sha256']==frozen_weights[str(latent/'goal_head.pt')]
            assert m['latent_adapter_sha256']==frozen_weights[str(latent/'lora/adapter_model.safetensors')]
        assert all(e['calls']<=protocol['call_budget'] for e in episodes)
    server = child = server_log = None
    def stop(proc):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        progress(status='loading_rule_model')
        if health() is not None: raise RuntimeError('Port occupied; refusing an unverified service')
        server_log=(dest/'vlm_server.log').open('a')
        server=subprocess.Popen([sys.executable,str(ROOT/'scripts/agentic/vlm_server.py'),'--model',str(model),'--quantize','--port',str(a.port)],cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+180
        while health() is None:
            if server.poll() is not None: raise RuntimeError('VLM service exited during startup')
            if time.monotonic()>deadline: raise TimeoutError('VLM startup timeout')
            time.sleep(1)
        assert Path(health()['model']).resolve()==model and health()['quantize'] is True
        completed=0
        for scenario in protocol['scenarios']:
            for mode in protocol['modes']:
                if sources()!=freeze['source_sha256']: raise RuntimeError('Source changed during frozen evaluation')
                if {str(x):sha(x) for x in weights}!=frozen_weights: raise RuntimeError('Checkpoint changed during evaluation')
                seed=scenario['seed']; template=scenario['template_set']; offset=scenario['instruction_offset']
                name=f'seed{seed}_offset{offset}_{template}_{mode}'; out=dest/name; report_path=out/'report.json'
                if report_path.exists():
                    validate(json.loads(report_path.read_text()),scenario,mode); completed+=1; continue
                progress(status='running', current_job=name, completed_jobs=completed)
                print(json.dumps({'starting':name,'completed_jobs':completed}),flush=True)
                command=[sys.executable,str(ROOT/'scripts/agentic/evaluate_baselines_v5.py'),'--headless','--output',str(out),
                         '--seeds',str(seed),'--modes',mode,'--template-set',template,'--instruction-offset',str(offset),
                         '--max-steps',str(protocol['max_steps']),'--budget',str(protocol['call_budget']),
                         '--latent-checkpoint',str(latent),'--vlm-url',address]
                if seed in protocol['record_video_seeds']: command.append('--record-video')
                start=time.monotonic()
                with (dest/(name+'.log')).open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        if time.monotonic()-start>protocol['job_timeout_seconds']: raise TimeoutError(name+' watchdog expired')
                        if server.poll() is not None: raise RuntimeError('Owned VLM service exited')
                        progress(current_job_elapsed_s=round(time.monotonic()-start,1));time.sleep(5)
                    if child.returncode: raise RuntimeError(f'{name} exited {child.returncode}; see job log')
                validate(json.loads(report_path.read_text()),scenario,mode);completed+=1
                subprocess.run([sys.executable,str(ROOT/'scripts/agentic/analyze_baselines_v5.py'),str(dest)],cwd=ROOT,env=env,check=True,stdout=subprocess.DEVNULL)
                progress(completed_jobs=completed,current_job=None)
                print(json.dumps({'completed':name,'completed_jobs':completed}),flush=True)
        progress(status='completed',completed_jobs=completed,current_job=None)
    except BaseException as error:
        progress(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        stop(child);stop(server)
        if server_log:server_log.close()
        lock.close()


if __name__=='__main__': main()
