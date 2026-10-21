"""Finite local training + two-seed diagnostic, with no recurring automation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/latency_counterfactual'))
from run_original import gpu_available

MODES=['continue','slow','hold','slow_smooth','hold_smooth','learned','wait_only']
SEEDS=[82001,82002]
N=12

def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def counts(rows):
    valid=[r for r in rows if r['eligible']]
    return dict(nominal=len(rows),eligible=len(valid),prefix_failures=len(rows)-len(valid),
        wait_drops=sum(r['wait_drop'] for r in valid),resume_drops=sum(r['resume_drop'] for r in valid),
        successes=sum(r['success'] for r in valid),success_rate=sum(r['success'] for r in valid)/len(valid) if valid else None)

def summarize(out):
    completed=[]
    for complete in (out/'eval').glob('*/complete.json'):
        folder=complete.parent
        rows=json.loads((folder/'eval_trials.json').read_text())
        audit=json.loads((folder/'prefix_replay_audit.json').read_text())
        if len(rows)!=N or audit['max_abs_error']!=0 or not audit['eligible_equal']:
            raise RuntimeError(f'Invalid completed arm {folder}')
        completed.append((folder.name,rows))
    positive=[r for _,rows in completed for r in rows if r['delay_s']>0]
    groups={(r['seed'],r['delay_s']) for r in positive}
    balanced=[(seed,delay) for seed,delay in groups if {r['mode'] for r in positive if (r['seed'],r['delay_s'])==(seed,delay)}==set(MODES)]
    rows=[r for r in positive if (r['seed'],r['delay_s']) in balanced]
    zero=[dict(arm=name,seed=batch[0]['seed'],**counts(batch)) for name,batch in completed if batch[0]['delay_s']==0]
    variation=[]
    for seed in SEEDS:
        vals=[v['success_rate'] for v in zero if v['seed']==seed and v['success_rate'] is not None]
        variation.append(dict(seed=seed,completed_repeats=len(vals),success_rate_spread=max(vals)-min(vals) if vals else None))
    predictions=[]
    for name,batch in completed:
        if batch[0]['mode']!='learned' or batch[0]['delay_s']==0:continue
        for snap in json.loads((out/'eval'/name/'outcome_predictions.json').read_text()):
            for i,r in enumerate(batch):
                if snap['active'][i]:predictions.append((snap['phase'],snap['probability'][i],float(r['success'])))
    outcome={phase:dict(samples=len(items),brier=sum((p-y)**2 for _,p,y in items)/len(items))
             for phase in ('wait','arrival') if (items:=[v for v in predictions if v[0]==phase])}
    result=dict(status='exploratory_pilot',completed_eval_arms=len(completed),completed_eval_trials=sum(len(b) for _,b in completed),
        balanced_positive_groups=[list(g) for g in sorted(balanced)],
        positive_delay={m:counts([r for r in rows if r['mode']==m]) for m in MODES},
        per_seed_delay={f'{seed}_{delay:g}':{m:counts([r for r in rows if r['seed']==seed and r['delay_s']==delay and r['mode']==m]) for m in MODES} for seed,delay in balanced},
        zero_identical_control_repeats=zero,zero_variation=variation,outcome_head_diagnostic=outcome,
        caveats=['Only two evaluation seeds / at most 24 independent request states; repeated delays are not independent states.',
                 'Original renderer variation retained; zero-control spread must accompany all method comparisons.',
                 'Known delivered goals and simulated delays; no live Astra/VLM inference.',
                 'Outcome head has on-policy Monte Carlo labels; Brier score on two seeds is not calibration validation.',
                 'Final training checkpoint selected by a fixed budget, not these evaluation outcomes.'])
    save(out/'summary.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--iterations',type=int,default=32);p.add_argument('--python',required=True);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'tmp').mkdir();(out/'eval').mkdir();(out/'logs').mkdir()
    lock=Path('/tmp',f'agent_inhand_paired_gpu_{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=gpu_available(a.gpu)
    if not snapshot['available']:raise RuntimeError(f'GPU is not available: {snapshot}')
    paths=[f for base in ('source','scripts/agentic','scripts/recovery_controller_v2','scripts/latency_counterfactual','configs','assets') for f in (ROOT/base).rglob('*')
           if f.is_file() and f.suffix in ('.py','.usd','.usda','.usdc','.json','.yaml','.yml')]
    paths += [ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt',
              ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt',ROOT/'outputs/latency_controller_v1/train_v2_20261015/controller_v1_final.pt']
    frozen={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(paths))}
    rng=random.Random(181026);groups=[(seed,delay) for seed in SEEDS for delay in (6.,20.)];rng.shuffle(groups)
    schedule=[]
    for seed,delay in groups:
        modes=MODES.copy();rng.shuffle(modes);schedule.append(dict(seed=seed,delay=delay,modes=modes))
    protocol=dict(version='joint_wait_recovery_pilot_v2',seeds=SEEDS,num_envs=N,training_iterations_per_objective=a.iterations,
        training_seeds=list(range(920000,920000+a.iterations)),training_delays=[2.,6.,12.],evaluation_delays=[0.,6.,20.],
        modes=MODES,planned_training_rollouts=2*a.iterations*N,planned_evaluation_trials=432,
        zero_repeats='two continue repeats before training, plus learned and wait_only bypass repeats after training, per seed',
        schedule=schedule,source_asset_weight_sha256=frozen,initial_gpu=snapshot,
        success='six consecutive post-response steps within .16 rad; no prior drop; 300 recovery steps',
        learned_control='15Hz continuous bounded residual during wait and first 60 post-response steps, then frozen student',
        training_comparison='same model initialization, reset seeds, budget, delays and action parametrization; completion objective versus wait-only reward with outcome loss disabled',
        stop_conditions=['runtime errors or nonfinite updates','nonzero mechanical prefix audit','source/weight drift','all zero preflight success for either seed','incomplete trial set'],
        evidence='two-seed diagnostic, no significance claim; retain variability and all failures',created=time.time())
    save(out/'protocol.json',protocol)
    with tarfile.open(out/'frozen_sources_and_weights.tar.gz','w:gz') as tar:
        for name in frozen:tar.add(ROOT/name,arcname=name,recursive=False)
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu),
        PAIRED_RENDER_GPU=str(a.gpu),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
    env['PYTHONPATH']=':'.join(str(ROOT/q) for q in ('scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'))
    state=dict(pid=os.getpid(),status='starting',gpu=a.gpu,completed_stages=0,planned_evaluation_trials=432)
    child=None
    def progress(**extra):state.update(extra,updated=time.time());save(out/'progress.json',state)
    def interrupt(*unused):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def run(name,folder,args,timeout=1200):
        nonlocal child
        if folder.exists():raise RuntimeError(f'Existing result requires review: {folder}')
        if any(sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('Frozen files changed')
        cmd=[a.python,str(ROOT/'scripts/agentic/train_recovery_controller_v2.py'),'--headless','--device','cuda:0','--num-envs',str(N),
             '--output',str(folder),'--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']+args
        save(out/'logs'/f'{name}_command.json',cmd)
        with (out/'logs'/f'{name}.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            begin=time.monotonic();progress(status=name,child_pid=child.pid,current_output=str(folder))
            while child.poll() is None:
                progress(stage_wall_seconds=round(time.monotonic()-begin,1))
                if time.monotonic()-begin>timeout:raise TimeoutError(f'Watchdog expired: {name}')
                time.sleep(5)
            if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}; log retained')
        if not (folder/'complete.json').exists():raise RuntimeError(f'Missing completion: {name}')
        progress(completed_stages=state['completed_stages']+1,child_pid=None)
    def evaluation(seed,delay,mode,name,checkpoint=None):
        cache=out/f'cache_{seed}/request_state.pt'
        expected=json.loads((out/f'cache_{seed}/cache_hash.json').read_text())['sha256']
        if sha(cache)!=expected:raise RuntimeError('Frozen request cache changed')
        if checkpoint and sha(checkpoint)!=json.loads((out/'checkpoint_hashes.json').read_text())[mode]:raise RuntimeError('Evaluation checkpoint changed')
        args=['--mode',mode,'--seed',str(seed),'--delay',str(delay),'--request-cache',str(cache)]
        if checkpoint:args+=['--checkpoint',str(checkpoint),'--objective','completion' if mode=='learned' else 'wait_survival']
        if delay and mode in ('learned','slow_smooth','hold_smooth'):args+=['--record-slots','0','5']
        run(name,out/'eval'/name,args)
        summary=summarize(out);progress(completed_evaluation_trials=summary['completed_eval_trials'])
    try:
        for seed in SEEDS:
            run(f'cache_{seed}',out/f'cache_{seed}',['--mode','cache','--seed',str(seed)])
            cache_sha=sha(out/f'cache_{seed}/request_state.pt')
            save(out/f'cache_{seed}/cache_hash.json',dict(sha256=cache_sha))
            for rep in (1,2):evaluation(seed,0.,'continue',f'zero_{seed}_continue_r{rep}')
            zero=[v for v in summarize(out)['zero_identical_control_repeats'] if v['seed']==seed]
            if sum(v['successes'] for v in zero)==0:raise RuntimeError(f'All repeated zero-delay preflight tasks fail for seed {seed}; no seed substitution')
        for objective in ('completion','wait_survival'):
            run(f'train_{objective}',out/f'train_{objective}',['--mode','train','--seed','920000','--iterations',str(a.iterations),'--objective',objective],timeout=7200)
        checkpoints={m:out/f'train_{obj}/controller_final.pt' for m,obj in [('learned','completion'),('wait_only','wait_survival')]}
        save(out/'checkpoint_hashes.json',{m:sha(path) for m,path in checkpoints.items()})
        for seed in SEEDS:
            for mode in checkpoints:evaluation(seed,0.,mode,f'zero_{seed}_{mode}',checkpoints[mode])
        for group in schedule:
            for mode in group['modes']:
                seed,delay=group['seed'],group['delay']
                evaluation(seed,delay,mode,f'delay_{delay:g}_seed_{seed}_{mode}',checkpoints.get(mode))
        summary=summarize(out)
        if summary['completed_eval_trials']!=432:raise RuntimeError('Incomplete final evaluation')
        progress(status='completed_diagnostic',child_pid=None,current_output=None)
    except BaseException as exc:
        progress(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        lock.close()

if __name__=='__main__':main()
