"""Local development diagnostic with given goals; no Astra, no language success claim."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--max-steps',type=int,default=900)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from assets import *
from runtime import CompletionRuntime
from features import module
from latent_data_v4 import split_layouts
from latency_controller_v1 import normalized_target
from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);env=None
try:
    gate=CompletionRuntime(a.checkpoint,a.device);env,raw,symbols=v1.environment(a.seed,a.max_steps,a.device)
    symbols.set_layout(split_layouts(a.seed,'val'))
    colors=np.random.default_rng(a.seed+763).permutation(6).tolist();ids=torch.tensor(colors,device=a.device)
    for i,c in enumerate(colors):raw.cfg.fixed_target_face=c;raw._reset_target_pose(torch.tensor([i],device=a.device))
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
    active=torch.ones(6,dtype=torch.bool,device=a.device);ih=ph=None;trace=[]
    first=[None]*6;first_reach=[None]*6;drops=[False]*6;start=time.time()
    with torch.inference_mode():
        for step in range(a.max_steps):
            if step%2==0:
                _,_,f,_,prop=v1.sensor_frame(raw,symbols);ih=v1.history_append(ih,f,gate.encoder.history);ph=v1.history_append(ph,prop,gate.encoder.history)
            e=rotation_distance(raw.object_rot,raw.goal_rot)
            if step%10==0:
                pred=gate.update(ih,ph)
                for i,c in enumerate(colors):
                    if not active[i]:continue
                    if first_reach[i] is None and float(e[i])<=.16:first_reach[i]=step
                    proposed=gate.can_stop(i,c)
                    row=dict(case=i,step=step,goal=c,probability=float(pred['ready'][i,c]),progress=float(pred['progress'][i,c]),error=float(e[i]),proposed=proposed)
                    trace.append(row)
                    if first[i] is None and proposed:first[i]=row
            action=gate.encoder.policy(module.VisualStudentBatch(ih,gate.encoder.text[ids],ph))['action']
            action[~active]=normalized_target(raw)[~active]
            _,_,term,trunc,info=env.step(action)
            for i in range(6):
                if active[i] and bool(info['semantic_metrics']['dropped'][i]):drops[i]=True
                if active[i] and first_reach[i] is None and float(info['semantic_metrics']['orientation_error'][i])<=.16:first_reach[i]=step+1
            active &= ~(term|trunc|info['semantic_metrics']['dropped'].bool())
            if step%100==0:
                save(out/'progress.json',dict(step=step,active=int(active.sum()),proposed=sum(x is not None for x in first),wall_s=time.time()-start))
                print(f'step={step} active={int(active.sum())}',flush=True)
            if not active.any():break
    episodes=[dict(case=i,goal=c,first_true_reach=first_reach[i],first_stop_proposal=first[i],valid_first_proposal=first[i] is not None and first[i]['error']<=.16,drop_during_full_rollout=drops[i]) for i,c in enumerate(colors)]
    save(out/'report.json',dict(protocol='Given-goal shadow completion diagnostic; policy keeps running after proposals. Not autonomous semantic task success and not an Astra evaluation.',
         seed=a.seed,checkpoint_sha256=sha(a.checkpoint),calibrated=bool(gate.pack.get('calibration')),cloud_eligible=gate.pack['cloud_eligible'],
         threshold=gate.threshold,episodes=episodes,trace=trace,wall_s=time.time()-start))
except BaseException:
    import traceback
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
