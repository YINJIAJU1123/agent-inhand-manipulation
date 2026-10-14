"""Bounded fresh-process launcher; simulator reset state cannot leak across modes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[811])
p.add_argument('--modes',nargs='+',default=['oracle','once','periodic','event_memory'])
p.add_argument('--max-steps',type=int,default=600)
p.add_argument('--grounding',choices=['inventory','direct'],default='inventory')
p.add_argument('--disable-recovery',action='store_true'); a=p.parse_args()
root=Path(__file__).resolve().parents[2]; dest=Path(a.output).resolve(); dest.mkdir(parents=True,exist_ok=True)
for seed in a.seeds:
  for mode in a.modes:
    name=f'seed{seed}_{mode}'; out=dest/name
    command=[sys.executable,str(root/'scripts/agentic/evaluate_visual_agent.py'),'--headless','--output',str(out),'--seeds',str(seed),'--modes',mode,'--max-steps',str(a.max_steps),'--grounding',a.grounding]
    if a.disable_recovery: command.append('--disable-recovery')
    print(json.dumps({'starting':name}),flush=True)
    with (dest/(name+'.log')).open('w') as log:
      result=subprocess.run(command,cwd=root,env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,timeout=900)
    if result.returncode: raise RuntimeError(f'{name} failed: see {dest/(name+".log")}')
    report=json.loads((out/'report.json').read_text()); rows=report['episodes']
    print(json.dumps({'finished':name,'episodes':len(rows),'success':sum(r['success'] for r in rows),'grounding_correct':sum(r['grounding_correct'] for r in rows),'calls':sum(r['calls'] for r in rows)}),flush=True)
