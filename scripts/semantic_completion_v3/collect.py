"""Dense sensor evidence for completion learning. Pose labels are supervision only."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic'),str(ROOT/'scripts/semantic_completion_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--max-steps',type=int,default=900);p.add_argument('--cases',type=int,default=24)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from assets import *
from sensors import object_regions
from latent_data_v4 import split_layouts
from latency_controller_v1 import normalized_target
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch

def errors(q):
    s=2**-.5;targets=torch.tensor([[s,0,-s,0],[s,0,s,0],[s,s,0,0],[s,-s,0,0],[1,0,0,0],[0,1,0,0]],device=q.device)
    return 2*torch.acos((q[:,None]*targets[None]).sum(-1).abs().clamp(0,1))

out=Path(a.output);out.mkdir(parents=True,exist_ok=False);env=None
try:
    assert a.cases in [6,12,24]
    n=a.cases;student,text,pack=v1.load_student(a.device);env,raw,symbols=v1.environment(a.seed,a.max_steps,a.device,num_envs=n)
    layout=split_layouts(a.seed,a.split,count=n);symbols.set_layout(layout)
    goals=np.random.default_rng(a.seed+763).permutation(np.tile(np.arange(6),n//6)).tolist()
    for i,c in enumerate(goals):raw.cfg.fixed_target_face=c;raw._reset_target_pose(torch.tensor([i],device=a.device))
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
    ids=torch.tensor(goals,device=a.device);active=torch.ones(n,dtype=torch.bool,device=a.device);ih=ph=None
    arrays=[];rows=[];actions_list=[];started=time.time();endings={}
    with torch.inference_mode():
        for step in range(a.max_steps):
            if step%2==0:
                camera,rgb,f,_,prop=v1.sensor_frame(raw,symbols);ih=v1.history_append(ih,f,pack['history']);ph=v1.history_append(ph,prop,pack['history'])
            if step%10==0:
                e=errors(raw.object_rot);regions=object_regions(rgb,camera['depth'])
                if step==0:
                    from PIL import Image
                    Image.fromarray(rgb[0].cpu().numpy()).save(out/'sensor_case0_step0.png')
                    Image.fromarray(regions['roi_rgb'][0].permute(1,2,0).cpu().numpy()).save(out/'roi_case0_step0.png')
                for i in range(n):
                    if active[i]:
                        arrays.append(dict(features=ih[i].cpu(),proprio=ph[i].cpu(),error=e[i].cpu(),rotation_label=raw.object_rot[i].cpu(),**{k:v[i].cpu() for k,v in regions.items()}))
                        rows.append(dict(seed=a.seed,case=i,step=step,skill=goals[i],segment=0,split=a.split))
            action=student(VisualStudentBatch(ih,text[ids],ph))['action'];action[~active]=normalized_target(raw)[~active]
            actions_list.append(action.cpu());_,_,term,trunc,info=env.step(action)
            done=term|trunc|info['semantic_metrics']['dropped'].bool()
            for i in range(n):
                if active[i] and done[i]:endings[i]=dict(step=step+1,drop=bool(info['semantic_metrics']['dropped'][i]))
            active &= ~done
            if step%100==0:
                save(out/'progress.json',dict(step=step,frames=len(rows),active=int(active.sum()),wall_s=time.time()-started));print(f'step={step} frames={len(rows)} active={int(active.sum())}',flush=True)
            if not active.any():break
    d={k:torch.stack([r[k] for r in arrays]) for k in arrays[0]};d['rows']=rows;torch.save(d,out/'data.pt');torch.save(torch.stack(actions_list),out/'actions.pt')
    save(out/'manifest.json',dict(status='completed',seed=a.seed,split=a.split,frames=len(rows),fixed_goals=goals,endings=endings,
        ready_counts=(d['error']<=.16).sum(0).tolist(),near_counts=((d['error']>.16)&(d['error']<=.4)).sum(0).tolist(),
        cases=n,student_sha256=sha(STUDENT),source_sha256=sha(__file__),sensor_source_sha256=sha(Path(__file__).with_name('sensors.py')),data_sha256=sha(out/'data.pt'),
        roi_found=int(d['roi_found'].sum()),sensor_fields=['features','proprio','roi_rgb','roi_depth','roi_box','roi_found'],supervision_only_fields=['error','rotation_label'],
        protocol='Frozen student, balanced fixed goals, natural approach/reach/drift/failure, no success termination. Added 96x96 RGB-D crops use RGB color evidence only, not GT boxes. No pixel edits or pose-based crop selection. All six errors and object quaternion are supervision only. No Astra calls.'))
except BaseException:
    import traceback
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
