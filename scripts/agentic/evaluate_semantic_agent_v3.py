"""Paired simulation pilot; privileged values stay in renderer/scorer/oracle only.

This is a NEW symbol benchmark, not a rerun of the frozen language protocol.
The high-level client receives RGB, actual instruction, proprioception and time.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/BrainCo_DexHand'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[811])
p.add_argument('--modes',nargs='+',default=['oracle'])
p.add_argument('--instruction-offset',type=int,default=0)
p.add_argument('--template-set',choices=['canonical','paraphrase'],default='canonical')
p.add_argument('--vlm-url',default='http://127.0.0.1:18741')
p.add_argument('--max-steps',type=int,default=600)
p.add_argument('--budget',type=int,default=6)
p.add_argument('--record-video',action='store_true',help='Save all six pre-action RGB streams at 15 fps of simulation time')
p.add_argument('--no-symbols',action='store_true',help='Oracle appearance control only')
p.add_argument('--checkpoint',default=str(ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'))
p.add_argument('--adapter',default=str(ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'))
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(); a.enable_cameras=True
if len(a.seeds)!=1 or len(a.modes)!=1: raise ValueError('Each seed/mode/instruction offset needs a fresh simulator process')
if a.no_symbols and a.modes!=['oracle']: raise ValueError('no-symbols is an oracle-only control')
app=AppLauncher(a).app
import torch, numpy as np, gymnasium as gym
from PIL import Image
import BrainCo_DexHand
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch
from symbol_scene import SymbolScene
from student_features import _semantic_rgb_features
from visual_agent import Decision
from semantic_agent_v3 import SemanticAgent,SemanticVLMClient,SYMBOLS
from semantic_protocol_v3 import make_cases

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data): path.write_text(json.dumps(data,indent=2)+'\n')

recorders=[]
try:
    dest=Path(a.output); dest.mkdir(parents=True,exist_ok=True)
    device='cuda:0'; n=6
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    frozen=json.loads((ROOT/'configs/visual_student_freeze.json').read_text())
    assert ckpt.get('freeze_id')==frozen['freeze_id']
    assert float(ckpt['action_scale'])==float(frozen['action_contract']['action_scale'])==1.0
    policy=VisualLanguageStudent(ckpt['rgb_dim'],ckpt['language_dim'],ckpt['proprio_dim'],ckpt['action_dim'],memory_mode=ckpt['memory_mode']).to(device).eval()
    policy.load_state_dict(ckpt['model'])
    text=torch.load(a.adapter,map_location=device,weights_only=False)['canonical_embeddings'].float()
    cfg=parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',device=device,num_envs=n)
    cfg.tiled_camera.width=512; cfg.tiled_camera.height=512
    cfg.seed=a.seeds[0]; cfg.max_consecutive_success=0; cfg.freeze_goal_for_episode=True; cfg.record_eval_metrics=True; cfg.goal_yaw=0.0
    cfg.episode_length_s=a.max_steps/30.0
    env=gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',cfg=cfg); raw=env.unwrapped
    symbols=SymbolScene(raw)
    if a.no_symbols: symbols.markers.set_visibility(False)
    client=SemanticVLMClient(url=a.vlm_url)
    records=[]; start=time.monotonic()
    metadata={'benchmark':'language_symbol_agent_v3','checkpoint_sha256':sha(a.checkpoint),'adapter_sha256':sha(a.adapter),'seeds':a.seeds,'modes':a.modes,'grounding':'instruction_conditioned_inventory','instruction_offset':a.instruction_offset,'template_set':a.template_set,'no_symbols':a.no_symbols,'feedback_reobservation_enabled':a.modes==['feedback'],'camera_resolution':512,'student_resolution':256,'vision_stride':2,'agent_stride':10,'call_budget':a.budget,'max_steps':a.max_steps,'env_max_episode_length':int(raw.max_episode_length),'success':'accepted correct visual goal AND executing it AND orientation error <= 0.16 rad without drop; evaluator stops scored slot, not VLM-certified stopping','video_capture':'existing pre-action RGB every 2 control steps; 15 fps simulation time' if a.record_video else None,'agent_inputs':['RGB','actual full instruction','measured proprioception','elapsed control steps'],'oracle_has_target_label':True,'source_sha256':{name:sha(Path(__file__).parent/name) for name in ['evaluate_semantic_agent_v3.py','semantic_agent_v3.py','semantic_protocol_v3.py','visual_agent.py','student_features.py','symbol_scene.py','vlm_server.py']}}
    write(dest/'manifest.json',metadata)
    for seed in a.seeds:
      for mode in a.modes:
        if mode not in ['oracle','once','periodic','feedback']: raise ValueError(mode)
        torch.manual_seed(seed); np.random.seed(seed)
        env.reset(seed=seed)
        cases=make_cases(seed,a.instruction_offset,a.template_set)
        layout=[x['layout'] for x in cases]
        targets=[x['target_color_index'] for x in cases]
        instructions=[x['instruction'] for x in cases]
        for i in range(n):
            raw.cfg.fixed_target_face=targets[i]
            raw._reset_target_pose(torch.tensor([i],device=device))
        raw.cfg.fixed_target_face=None
        raw.goal_markers.set_visibility(False)
        raw._compute_intermediate_values()
        symbols.set_layout(layout)
        initial={'object_rotation':raw.object_rot.cpu().tolist(),'object_position':raw.object_pos.cpu().tolist(),'proprio':raw.compute_student_proprio().cpu().tolist(),'layout':layout}
        write(dest/f'initial_{seed}_{mode}.json',initial)
        if a.record_video:
            from rollout_video import RolloutVideo
            recorders=[RolloutVideo(dest/f'{seed}_{mode}_case{i}_continuous.mp4') for i in range(n)]
        agents=[SemanticAgent(mode,client,instructions[i],budget=a.budget) for i in range(n)] if mode!='oracle' else [None]*n
        decisions=[Decision(targets[i],'oracle','privileged_control_reference') if mode=='oracle' else Decision(None,'initial','') for i in range(n)]
        active=[True]*n; success=[False]*n; accidental_reach=[False]*n; dropped=[False]*n; minimum=[float('inf')]*n; steps=[0]*n
        action_states=[{} for _ in range(n)]; snapshots=[[] for _ in range(n)]
        ih=torch.zeros(n,ckpt['history'],ckpt['rgb_dim'],device=device)
        ph=torch.zeros(n,ckpt['history'],ckpt['proprio_dim'],device=device)
        for step in range(a.max_steps):
          with torch.inference_mode():
            if step%2==0:
              symbols.update(); camera=raw.capture_camera(); rgb=camera['rgb'][...,:3].to(torch.uint8)
              if recorders:
                video_rgb=rgb.cpu().numpy()
                for i,recorder in enumerate(recorders):
                    if active[i]:recorder.append(video_rgb[i])
              student_rgb=torch.nn.functional.interpolate(rgb.permute(0,3,1,2).float(),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
              depth=camera['depth']
              if depth.ndim==3: depth=depth.unsqueeze(-1)
              depth=torch.nn.functional.interpolate(depth.permute(0,3,1,2),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
              feat=_semantic_rgb_features(student_rgb,16,depth,16)
              prop=raw.compute_student_proprio().float()
              ih=torch.cat([ih[:,1:],feat[:,None]],1); ph=torch.cat([ph[:,1:],prop[:,None]],1)
              if step==0: ih[:]=feat[:,None]; ph[:]=prop[:,None]
            if step%10==0:
              frames=rgb.cpu().numpy()
              for i in range(n):
                if not active[i]: continue
                old_calls=agents[i].calls if agents[i] else 0
                if agents[i] is not None: decisions[i]=agents[i].tick(frames[i],prop[i].cpu().numpy(),step)
                new_call=agents[i] is not None and agents[i].calls>old_calls
                if step%30==0 or new_call:
                  path=f'{seed}_{mode}_case{i}_step{step:04d}.png'; Image.fromarray(frames[i]).save(dest/path)
                  # Privileged annotation is stored AFTER the agent decision, never in its input.
                  normal_z=quat_apply(raw.object_rot[i:i+1].expand(6,-1),raw._face_normals)[:,2].cpu().tolist()
                  snapshots[i].append({'path':path,'step':step,'decision':vars(decisions[i]).copy(),'scorer_face_normal_z':normal_z})
                if decisions[i].color is None: active[i]=False
            ids=torch.tensor([d.color if d.color is not None else 0 for d in decisions],device=device)
            actions=policy(VisualStudentBatch(ih,text[ids],ph))['action']
            assert torch.isfinite(actions).all() and actions.abs().max()<=1.00001
            for i in range(n):
              if active[i]:
                key=decisions[i].state; action_states[i][key]=action_states[i].get(key,0)+1
              else: actions[i]=0
            _,_,terminated,truncated,info=env.step(actions)
            metrics=info['semantic_metrics']
            for i in range(n):
              if not active[i]: continue
              steps[i]+=1
              minimum[i]=min(minimum[i],float(metrics['orientation_error'][i]))
              dropped[i]|=bool(metrics['dropped'][i])
              reached=bool(metrics['goal_reached'][i]) and not bool(metrics['dropped'][i])
              grounded=(mode=='oracle' or (agents[i].goal==targets[i] and agents[i].goal_symbol==cases[i]['target_symbol'] and decisions[i].state=='execute'))
              accidental_reach[i]|=reached and not grounded
              success[i]|=reached and grounded
              if success[i] or bool(terminated[i] or truncated[i]): active[i]=False
          if step%100==0: print(json.dumps({'seed':seed,'mode':mode,'step':step,'active':sum(active),'success':sum(success),'drop':sum(dropped)}),flush=True)
          if not any(active): break
        for recorder in recorders:recorder.close()
        for i in range(n):
            agent=agents[i]
            row={'seed':seed,'mode':mode,'case':i,'target_color_index':targets[i],'target_symbol':cases[i]['target_symbol'],'instruction':instructions[i],'layout':layout[i],'success':success[i] and not dropped[i],'incidental_reach_without_grounding':accidental_reach[i],'drop':dropped[i],'steps':steps[i],'min_orientation_error_rad':minimum[i] if np.isfinite(minimum[i]) else None,'final_goal':agent.goal if agent else targets[i],'parsed_goal_symbol':agent.goal_symbol if agent else cases[i]['target_symbol'],'language_correct':(agent.goal_symbol==cases[i]['target_symbol'] and agent.goal_direction=='up') if agent else True,'grounding_correct':(agent.goal==targets[i] and agent.goal_symbol==cases[i]['target_symbol']) if agent else True,'calls':agent.calls if agent else 0,'recovery_used':agent.recovery_used if agent else False,'action_states':action_states[i],'trace':agent.trace if agent else [],'feedback_trace':agent.feedback_trace if agent else [],'transitions':agent.transitions if agent else [],'snapshots':snapshots[i]}
            if recorders:
                row['video']={'path':recorders[i].path.name,'frames':recorders[i].frames,'fps':recorders[i].fps,'time_base':'simulation time, excludes synchronous VLM waiting','terminal_frame':'last available pre-action frame; simulator may reset at terminal step'}
            records.append(row)
        write(dest/'report.json',{'metadata':metadata,'elapsed_s':time.monotonic()-start,'episodes':records})
        print(json.dumps({'completed_seed':seed,'completed_mode':mode,'success':sum(success),'drop':sum(dropped),'elapsed_s':time.monotonic()-start}),flush=True)
    env.close()
finally:
    for recorder in recorders:recorder.close()
    app.close()
