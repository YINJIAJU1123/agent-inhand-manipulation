"""Print a portable development command; execute only with --execute."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def relocate(bundle,obj,cfg,cache):
    """Relocate only a derived asset copy; original weights/contracts stay intact."""
    original=json.loads((bundle/'objects'/obj/'OBJECT.json').read_text())
    runtime=bundle/cfg['runtime']
    if cfg['family']!='inhand4':return {}
    target=cache/obj/'marked_assets'
    shutil.copytree(bundle/'objects'/obj/'marked_assets',target,dirs_exist_ok=True)
    old=original['original_q1']
    mapping={old+'/repository':str(runtime),old+'/'+obj+'/marked_assets_v2':str(target)}
    for p in target.rglob('*'):
        if p.suffix not in ('.json','.usda') or not p.is_file():continue
        text=p.read_text()
        for a,b in mapping.items():text=text.replace(a,b)
        p.write_text(text)
    manifest=target/'manifest.json';m=json.loads(manifest.read_text())
    for entry in m['conditions'].values():entry['sha256']=sha(Path(entry['asset']))
    manifest.write_text(json.dumps(m,indent=2)+'\n')
    declaration=json.loads((bundle/'objects'/obj/'relocation.original.json').read_text())
    declaration['local_prefix']=str(runtime)
    declaration['source_inventory']=str(bundle/'runtime/inhand4_source_inventory.json')
    assert sha(Path(declaration['source_inventory']))==declaration['source_inventory_sha256']
    path=cache/obj/'relocation.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    return {'relocation':str(path),'assets_manifest':str(manifest)}

def command(a):
    bundle=a.bundle.resolve();cfg=json.loads((ROOT/'configs/objects.json').read_text())[a.object]
    if not (bundle/'BUNDLE.json').exists():raise ValueError('Missing separately delivered artifact bundle')
    python=str(Path(os.environ.get('VISERDEX_PYTHON',sys.executable)).resolve())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYGLET_HEADLESS_DEVICE=str(a.gpu),
        ISAACLAB_PATH=str(bundle/'runtime/IsaacLab'),VISERDEX_PYTHON=python,UV_BIN=os.environ.get('UV_BIN','uv'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1',
        XDG_CACHE_HOME=str(ROOT/'outputs/handoff_cache'),VISERDEX_OBJECT_NAME=a.object)
    if cfg.get('object_key'):env['VISERDEX_OBJECT']=cfg['object_key']
    if a.mode=='service':
        if a.method=='astra':raise ValueError('Use api_bridge.py for Astra')
        if not a.model or not a.model.is_dir():raise ValueError('--model must point to a separately obtained Qwen3-VL-4B-Instruct directory')
        model=str(a.model.resolve());grounding=str(bundle/cfg['grounding'])
        if a.object=='cube':
            if a.method=='vlm':argv=[python,str(ROOT/'scripts/agentic/latent_service_v7.py'),'--backend','rslcube','--checkpoint',grounding,'--model_path',model,'--port',str(a.port)]
            else:argv=[python,str(ROOT/'scripts/agentic/vlm_server.py'),'--model',model,'--port',str(a.port),'--quantize']
        else:
            folder='dice_latent_v1' if a.object=='die' else 'inhand4_q1'
            argv=[python,str(ROOT/'scripts'/folder/'vlm_service.py'),'--kind',a.method,'--model',model,'--port',str(a.port)]
            if a.method=='vlm':argv+=['--checkpoint',grounding]
        return argv,env
    if a.output is None:raise ValueError('Evaluation requires a fresh --output directory')
    if a.output.exists():raise FileExistsError('Refusing to replace existing evaluation output')
    if not 1<=a.episodes<=40:raise ValueError('Development runs use 1..40 episodes; formal protocols require a separate release')
    if a.seed in (183101,183102,186301,186302,187501,187502):raise ValueError('Use a new development seed, not a reserved historical seed')
    files={k:str(bundle/v) for k,v in cfg['files'].items()}
    for k,v in files.items():
        if not Path(v).exists():raise FileNotFoundError(k+': '+v)
    # Dry runs use a separate derived cache and perform no GPU/API work.
    files.update(relocate(bundle,a.object,cfg,ROOT/'outputs/handoff_cache'))
    evaluator=str(ROOT/cfg['evaluator']);args=[]
    for flag,value in files.items():args+=['--'+flag,value]
    args+=['--provider',a.method,'--seed',str(a.seed),'--episodes',str(a.episodes),'--num_envs','1','--output',str(a.output.resolve()),'--viz','none' if a.object=='cube' else 'newton_gl']
    if a.object=='cube':args+=['--project_root',str(ROOT),'--stop_mode','measured','--agent_protocol','feedback','--device','cuda:0']
    else:args+=['--development']
    if a.method=='astra':
        args+=['--maintenance','learned','--actor_checkpoint',str(bundle/cfg['actor_checkpoint'])]
        for k in ('execution_actor_checkpoint','stop_checkpoint'):
            if k in cfg:args+=['--'+k,str(bundle/cfg[k])]
        if a.execute and not all(env.get(k) for k in ('ASTRA_BRIDGE_URL','ASTRA_BRIDGE_KEY_FILE')):raise ValueError('Start your own API bridge and export ASTRA_BRIDGE_URL / ASTRA_BRIDGE_KEY_FILE')
    elif a.object=='cube':args+=['--maintenance','continue','--vlm_url' if a.method=='vlm' else '--rule_url',f'http://127.0.0.1:{a.port}']
    else:args+=['--service_url',f'http://127.0.0.1:{a.port}']
    if a.object=='cube':args+=['env.sim.physics.solver_cfg.nconmax=4096','env.sim.physics.solver_cfg.njmax=1024']
    entry=[evaluator,*args]
    if a.object=='die':entry=[str(ROOT/'scripts/handoff/bootstrap.py'),'--bundle',str(bundle),'--',*entry]
    return ['bash',str(bundle/cfg['runtime']/'scripts/run_newton.sh'),*entry],env

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--object',choices=['cube','die','duck','hedgehog','pig','milk_carton'],required=True)
    p.add_argument('--mode',choices=['eval','service'],default='eval');p.add_argument('--method',choices=['astra','vlm','rule'],required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--port',type=int,default=19081);p.add_argument('--model',type=Path)
    p.add_argument('--seed',type=int,default=990601);p.add_argument('--episodes',type=int,default=1);p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args();argv,env=command(a)
    print(shlex.join(argv),flush=True)
    if a.execute:subprocess.run(argv,cwd=ROOT,env=env,check=True)
    else:print('DRY RUN: command only; no simulator or model call was started.')
if __name__=='__main__':main()
