"""Live asynchronous semantic evaluation. No inference wait on the physics thread.

Response delivery is aligned to measured wall latency in simulation seconds;
late responses after terminal states never affect the scored episode.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--mode',choices=['astra','latent'],required=True)
p.add_argument('--waiting',choices=['continue','slow','v3','learned'],required=True)
p.add_argument('--template-set',choices=['canonical','paraphrase'],default='canonical')
p.add_argument('--max-steps',type=int,default=3600)
p.add_argument('--budget',type=int,default=6)
p.add_argument('--latent-url',default='http://127.0.0.1:18772')
p.add_argument('--controller',default=str(ROOT/'outputs/latency_controller_v1/checkpoints/controller_v1_final.pt'))
p.add_argument('--record-video',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.enable_cameras=True
if os.environ.get('ASTRA_RENDER_GPU'):
    a.kit_args=(a.kit_args+' --/renderer/activeGpu='+str(int(os.environ['ASTRA_RENDER_GPU']))).strip()
if a.mode=='astra':
    from astra_bridge_v6 import bridge_health
    bridge_health()
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch,numpy as np,gymnasium as gym
import BrainCo_DexHand
from PIL import Image,ImageDraw
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_mul,quat_conjugate
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent,VisualStudentBatch
from symbol_scene import SymbolScene
from student_features import _semantic_rgb_features
from semantic_agent_v3 import SYMBOLS
from semantic_protocol_v3 import make_cases
from latent_data_v4 import split_layouts
from async_agent_v7 import AsyncAgent
from latency_controller_v1 import Supervisor,normalized_target,command,sensor_observation

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
if (dest/'manifest.json').exists():raise RuntimeError('Fresh output required')
pool=ThreadPoolExecutor(max_workers=6);recorders=[];wrappers=[]
try:
    device=a.device;n=6
    student_path=ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'
    adapter_path=ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'
    ck=torch.load(student_path,map_location=device,weights_only=True)
    freeze=json.loads((ROOT/'configs/visual_student_freeze.json').read_text())
    assert ck['freeze_id']==freeze['freeze_id'] and float(ck['action_scale'])==1.
    policy=VisualLanguageStudent(ck['rgb_dim'],ck['language_dim'],ck['proprio_dim'],ck['action_dim'],memory_mode=ck['memory_mode']).to(device).eval()
    policy.load_state_dict(ck['model']);policy.requires_grad_(False)
    text=torch.load(adapter_path,map_location=device,weights_only=True)['canonical_embeddings'].float()
    controller=None;runtime=None
    if a.waiting in ('v3','learned'):
        if a.waiting=='learned':
            from model import load_controller
            controller,pack=load_controller(a.controller,device)
            assert pack['protocol']['adapter_sha256']==sha(adapter_path)
        else:
            from recovery_controller_v3 import RecoveryController
            pack=torch.load(a.controller,map_location=device,weights_only=True)
            controller=RecoveryController(616).to(device).eval()
            controller.load_state_dict(pack['model']);controller.requires_grad_(False)
        from runtime import ControllerRuntime
    cfg=parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',device=device,num_envs=n)
    cfg.tiled_camera.width=512;cfg.tiled_camera.height=512
    cfg.seed=a.seed;cfg.max_consecutive_success=0;cfg.freeze_goal_for_episode=True;cfg.record_eval_metrics=True;cfg.goal_yaw=0.
    cfg.episode_length_s=(a.max_steps+30)/30.
    env=gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',cfg=cfg);raw=env.unwrapped
    assert abs(raw.step_dt-1/30)<1e-7
    raw.cfg.sim.render_interval=1000000
    symbols=SymbolScene(raw)
    torch.manual_seed(a.seed);np.random.seed(a.seed);env.reset(seed=a.seed)
    cases=make_cases(a.seed,0,a.template_set);layout=split_layouts(a.seed,'test')
    targets=[]
    for i,case in enumerate(cases):
        case['layout']=layout[i];target=layout[i].index(SYMBOLS.index(case['target_symbol']));targets.append(target)
        raw.cfg.fixed_target_face=target;raw._reset_target_pose(torch.tensor([i],device=device))
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False);raw._compute_intermediate_values();raw._update_face_markers();symbols.set_layout(layout)
    save(dest/'initial.json',dict(joint_pos=raw.hand.data.joint_pos.cpu().tolist(),joint_vel=raw.hand.data.joint_vel.cpu().tolist(),object_root=raw.object.data.root_state_w.cpu().tolist(),prev_targets=raw.prev_targets.cpu().tolist(),layout=layout))
    predictor=None
    if a.mode=='astra':
        from astra_agent_v4 import AstraAgent
        from astra_bridge_v6 import BridgeClient
        agents=[AstraAgent(BridgeClient(),c['instruction'],budget=a.budget) for c in cases]
    else:
        from latent_service_v7 import RemotePredictor
        from latent_supervisor_v4 import LatentAgent
        predictor=RemotePredictor(a.latent_url)
        agents=[LatentAgent(predictor,c['instruction'],budget=a.budget) for c in cases]
    wrappers=[AsyncAgent(agent,pool,a.mode) for agent in agents]
    metadata=dict(benchmark='live_astra_learning_v1',seed=a.seed,mode=a.mode,waiting=a.waiting,template_set=a.template_set,max_steps=a.max_steps,call_budget=a.budget,
        student_sha256=sha(student_path),adapter_sha256=sha(adapter_path),controller_sha256=sha(a.controller) if controller else None,
        latent_checkpoint_sha256=predictor.checkpoint_sha256 if predictor else None,latent_adapter_sha256=predictor.adapter_sha256 if predictor else None,
        timing='actual inference runs concurrently; deliver after both Future completion and ceil(measured wall latency*30) physics steps; no realtime throughput claim',
        cold_start='latch normalized commanded joint targets until first delivered skill; no privileged goal',
        success='correct delivered semantic binding AND execute/verify AND error<=0.16 rad without drop; evaluator termination, autonomous stop separate',
        handoff=('controller during pending known skill and first two seconds after each delivered response' if controller else 'immediate frozen student at response arrival'),
        outcome_prediction='diagnostic only; not sent to Astra or used to stop',
        learning_source_sha256={x.name:sha(x) for x in Path(__file__).parent.glob('*.py')},
        source_sha256={x.name:sha(x) for x in (ROOT/'scripts/agentic').glob('*.py')},camera_resolution=512,student_resolution=256,control_hz=30,vision_hz=15,gate_hz=2)
    save(dest/'manifest.json',metadata)
    if a.record_video:
        from rollout_video import RolloutVideo
        recorders=[RolloutVideo(dest/f'case{i}.mp4') for i in range(n)]
    ih=torch.zeros(n,ck['history'],ck['rgb_dim'],device=device);ph=torch.zeros(n,ck['history'],ck['proprio_dim'],device=device)
    anchor=normalized_target(raw).clone();cold=anchor.clone();lastmode=torch.full((n,),2,device=device,dtype=torch.long);chosen=lastmode.clone()
    if controller is not None:runtime=ControllerRuntime(controller,anchor,a.waiting=='learned')
    active=[True]*n;success=[False]*n;dropped=[False]*n;false_stop=[False]*n;stop_claim=[False]*n;wait_drop=[False]*n
    minimum=[float('inf')]*n;steps=[0]*n;counts=[dict(continue_steps=0,slow_steps=0,hold_steps=0,cold_steps=0,pending_steps=0) for _ in range(n)]
    visual=oldvisual=None;start=time.monotonic();gate_trace=[]
    for step in range(a.max_steps):
        delivered=[False]*n
        with torch.inference_mode():
            if step%2==0:
                symbols.update();camera=raw.capture_camera(refresh=True);rgb=camera['rgb'][...,:3].to(torch.uint8)
                student_rgb=torch.nn.functional.interpolate(rgb.permute(0,3,1,2).float(),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
                depth=camera['depth'];depth=depth.unsqueeze(-1) if depth.ndim==3 else depth
                depth=torch.nn.functional.interpolate(depth.permute(0,3,1,2),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
                feat=_semantic_rgb_features(student_rgb,16,depth,16)
                compact=_semantic_rgb_features(student_rgb,4,depth,4)
                oldvisual=compact.clone() if visual is None else visual;visual=compact
                if runtime is not None:runtime.update_visual(compact)
                prop=raw.compute_student_proprio().float()
                ih=torch.cat([ih[:,1:],feat[:,None]],1);ph=torch.cat([ph[:,1:],prop[:,None]],1)
                if step==0:ih[:]=feat[:,None];ph[:]=prop[:,None]
            for i,w in enumerate(wrappers):
                if not active[i]:continue
                delivered[i]=w.poll(step)
                if w.closed:active[i]=False;continue
                decision=w.snapshot['decision']
                if decision['state']=='stop':
                    error=2*torch.asin(torch.linalg.vector_norm(quat_mul(raw.object_rot[i],quat_conjugate(raw.goal_rot[i]))[1:]).clamp(max=1.))
                    stop_claim[i]=True
                    ok=w.snapshot['goal']==targets[i] and w.snapshot['goal_symbol']==cases[i]['target_symbol'] and float(error)<=.16
                    success[i]=ok;false_stop[i]=not ok;active[i]=False;w.close(step);continue
                if decision['state']=='abstain':active[i]=False;w.close(step);continue
                if step%10==0:
                    w.submit(rgb[i].cpu().numpy().copy(),raw.compute_student_proprio()[i].cpu().numpy().copy(),step)
            ids=torch.tensor([w.snapshot['decision']['color'] or 0 for w in wrappers],device=device)
            goals=text[ids].clone()
            for i,w in enumerate(wrappers):
                if w.snapshot['goal_embedding'] is not None and w.snapshot['decision']['state']=='execute':goals[i]=torch.as_tensor(w.snapshot['goal_embedding'],device=device)
            proposal=policy(VisualStudentBatch(ih,goals,ph))['action']
            current=normalized_target(raw)
            pending=torch.tensor([active[i] and w.future is not None for i,w in enumerate(wrappers)],device=device)
            known=torch.tensor([w.snapshot['decision']['color'] is not None for w in wrappers],device=device)
            chosen=torch.zeros_like(lastmode)
            if a.waiting=='slow':chosen[pending & known]=1
            chosen[~known]=2
            actions,anchor=command(proposal,current,anchor,chosen,lastmode)
            if runtime is not None:
                age=torch.tensor([(step-w.request_step)/30 if w.future is not None or delivered[i] else 0 for i,w in enumerate(wrappers)],device=device)
                actions,diag=runtime.action(step,goals,raw.compute_student_proprio().float(),proposal,current,pending,known,delivered,age)
                if step%2==0 or any(delivered):gate_trace.append(diag)
            actions[~known]=cold[~known]
            for i in range(n):
                if not active[i]:actions[i]=current[i]
                else:
                    counts[i][('continue_steps','slow_steps','hold_steps')[int(chosen[i])]]+=1
                    if runtime is not None:counts[i]['controller_steps']=counts[i].get('controller_steps',0)+int(diag['controller_active'][i])
                    counts[i]['pending_steps']+=int(pending[i]);counts[i]['cold_steps']+=int(not known[i])
            if runtime is not None:runtime.commit(actions)
            assert torch.isfinite(actions).all() and float(actions.abs().max())<=1.00001
            if recorders and step%2==0:
                frames=rgb.cpu().numpy()
                for i,rec in enumerate(recorders):
                    if active[i]:
                        im=Image.fromarray(frames[i]);draw=ImageDraw.Draw(im);draw.rectangle((0,0,512,54),fill=(0,0,0))
                        draw.text((5,5),f'{a.mode}/{a.waiting} sim={step/30:.2f}s pending={bool(pending[i])}',fill='white')
                        draw.text((5,22),f"state={wrappers[i].snapshot['decision']['state']} gate={int(chosen[i])} calls={wrappers[i].submitted}",fill='white')
                        draw.text((5,38),'LIVE inference; physics continues',fill='white');rec.append(np.asarray(im))
            _,_,terminated,truncated,info=env.step(actions);lastmode=chosen.clone()
            metrics=info['semantic_metrics']
            for i,w in enumerate(wrappers):
                if not active[i]:continue
                w.advanced();steps[i]+=1;minimum[i]=min(minimum[i],float(metrics['orientation_error'][i]))
                dropped[i]=bool(metrics['dropped'][i]);wait_drop[i]=dropped[i] and bool(pending[i])
                grounded=w.snapshot['goal']==targets[i] and (a.mode=='latent' or w.snapshot['goal_symbol']==cases[i]['target_symbol']) and w.snapshot['decision']['state'] in ('execute','verify')
                success[i]=bool(metrics['goal_reached'][i]) and grounded and not dropped[i]
                if success[i] or bool(terminated[i] or truncated[i]):active[i]=False;w.close(step+1)
        if step%100==0:
            progress=dict(step=step,sim_time_s=(step+1)/30,wall_s=time.monotonic()-start,active=sum(active),success=sum(success),drop=sum(dropped),pending=sum(w.future is not None for w in wrappers),calls=sum(w.submitted for w in wrappers))
            save(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
        if not any(active):break
    for w in wrappers:w.close(step+1)
    for rec in recorders:rec.close()
    # Scored physics is finished. Drain outstanding requests solely for cost/error audit.
    pool.shutdown(wait=True,cancel_futures=True)
    for w in wrappers:w.poll(step+1)
    records=[]
    for i,w in enumerate(wrappers):
        records.append(dict(case=i,seed=a.seed,mode=a.mode,waiting=a.waiting,template_set=a.template_set,instruction=cases[i]['instruction'],layout=layout[i],target_symbol=cases[i]['target_symbol'],target_color=targets[i],success=success[i],drop=dropped[i],wait_drop=wait_drop[i],stop_claimed=stop_claim[i],false_stop=false_stop[i],steps=steps[i],min_orientation_error=minimum[i] if np.isfinite(minimum[i]) else None,final_delivered=w.snapshot,calls=w.submitted,counts=counts[i],async_events=w.events,trace=agents[i].trace,feedback_trace=agents[i].feedback_trace,video=f'case{i}.mp4' if recorders else None))
    # Numpy latent is converted only at persistence; model consumes the continuous tensor.
    for row in records:
        if row['final_delivered']['goal_embedding'] is not None:row['final_delivered']['goal_embedding']=np.asarray(row['final_delivered']['goal_embedding']).tolist()
    save(dest/'gate_trace.json',gate_trace)
    save(dest/'report.json',dict(metadata=metadata,wall_s=time.monotonic()-start,episodes=records))
    print(json.dumps(dict(completed=True,success=sum(success),drop=sum(dropped),episodes=6)),flush=True)
    env.close()
finally:
    pool.shutdown(wait=True,cancel_futures=True)
    for r in recorders:r.close()
    app.close()
