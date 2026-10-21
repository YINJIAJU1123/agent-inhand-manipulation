"""Paired prospective shadow comparison: all heads see identical live sensors.

The frozen policy runs independently of all proposals. This evaluates stop
advice with known externally assigned goals, not language or Astra success.
"""
import argparse,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--models',required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--max-steps',type=int,default=900);p.add_argument('--cases',type=int,default=6)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np
from assets import v1,save,sha,STUDENT,ADAPTER
from features import FrozenFeatures,module
from completion_model import load_model
from latent_data_v4 import split_layouts
from latency_controller_v1 import normalized_target
from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);env=None
try:
    assert a.cases in [6,12,24], 'Use balanced six-goal case blocks'
    n=a.cases;encoder=FrozenFeatures(a.device);models={};specs=json.loads(Path(a.models).read_text())
    for name,spec in specs.items():
        model,pack=load_model(spec['checkpoint'],a.device)
        for key,path in [('student_sha256',STUDENT),('adapter_sha256',ADAPTER)]:
            if pack['feature_manifest'][key]!=sha(path):raise RuntimeError('Frozen feature asset mismatch')
        models[name]=(model,pack,float(spec.get('threshold',pack['threshold'])))
    env,raw,symbols=v1.environment(a.seed,a.max_steps,a.device,num_envs=n)
    symbols.set_layout(split_layouts(a.seed,'val',count=n))
    goals=np.random.default_rng(a.seed+763).permutation(np.tile(np.arange(6),n//6)).tolist();ids=torch.tensor(goals,device=a.device)
    for i,c in enumerate(goals):
        raw.cfg.fixed_target_face=c;raw._reset_target_pose(torch.tensor([i],device=a.device))
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
    torch.save(dict(object_state=raw.object.data.root_state_w.cpu().clone(),joint_pos=raw.hand.data.joint_pos.cpu().clone(),
        joint_vel=raw.hand.data.joint_vel.cpu().clone(),goal_rot=raw.goal_rot.cpu().clone()),out/'initial_state.pt')
    protocol=dict(seed=a.seed,cases=n,goals=goals,models={name:dict(checkpoint_sha256=sha(spec['checkpoint']),threshold=models[name][2]) for name,spec in specs.items()},
        source_sha256=sha(__file__),initial_state_sha256=sha(out/'initial_state.pt'),
        scope='Same live trajectory and sensors for all heads. Known-goal shadow diagnostic, no stop interventions and no Astra calls. Development seeds reused.',cloud_calls=0)
    save(out/'protocol.json',protocol)
    active=torch.ones(n,dtype=torch.bool,device=a.device);ih=ph=None;trace=[]
    first={name:[None]*n for name in models};reach=[None]*n;drops=[False]*n;start=time.time()
    with torch.inference_mode():
        for step in range(a.max_steps):
            if step%2==0:
                _,_,f,_,prop=v1.sensor_frame(raw,symbols)
                ih=v1.history_append(ih,f,encoder.history);ph=v1.history_append(ph,prop,encoder.history)
            if step%10==0:
                measured=encoder.candidates(ih,ph)
                predictions={name:model(measured)['ordinal'][...,2].sigmoid().cpu() for name,(model,_,_) in models.items()}
                error=rotation_distance(raw.object_rot,raw.goal_rot)
                for i,c in enumerate(goals):
                    if not active[i]:continue
                    if reach[i] is None and float(error[i])<=.16:reach[i]=step
                    row=dict(case=i,step=step,goal=c,error=float(error[i]),probabilities={name:float(pred[i,c]) for name,pred in predictions.items()})
                    trace.append(row)
                    for name,(_,pack,threshold) in models.items():
                        if first[name][i] is None and pack.get('calibration') and predictions[name][i,c]>=threshold:
                            first[name][i]=dict(step=step,error=float(error[i]),probability=float(predictions[name][i,c]),correct=float(error[i])<=.16)
            action=encoder.policy(module.VisualStudentBatch(ih,encoder.text[ids],ph))['action']
            action[~active]=normalized_target(raw)[~active]
            _,_,term,trunc,info=env.step(action)
            for i in range(n):
                if not active[i]:continue
                if bool(info['semantic_metrics']['dropped'][i]):drops[i]=True
                if reach[i] is None and float(info['semantic_metrics']['orientation_error'][i])<=.16:reach[i]=step+1
            active &= ~(term|trunc|info['semantic_metrics']['dropped'].bool())
            if step%100==0:
                save(out/'progress.json',dict(step=step,active=int(active.sum()),wall_s=time.time()-start,
                    proposals={name:sum(r is not None for r in rows) for name,rows in first.items()}))
                print('step',step,'active',int(active.sum()),flush=True)
            if not active.any():break
    report={}
    for name,rows in first.items():
        report[name]=dict(proposals=sum(r is not None for r in rows),correct=sum(r is not None and r['correct'] for r in rows),
            false_first_stops=sum(r is not None and not r['correct'] for r in rows),
            episodes=[dict(case=i,goal=goals[i],first_stop=rows[i],first_true_reach=reach[i],drop_during_full_rollout=drops[i]) for i in range(n)])
    save(out/'report.json',dict(protocol=protocol,heads=report,trace=trace,wall_s=time.time()-start))
except BaseException:
    import traceback
    save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if env is not None:env.close()
    app.close()
