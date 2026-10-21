"""Baseline validity diagnostic: one naturally continuous rollout, no state restore.

Records single-step reach alongside the six-step post-response criterion.
No inference service is called and no policy weights are changed.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/BrainCo_DexHand'), str(ROOT/'scripts/agentic')]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--num-envs', type=int, default=12)
p.add_argument('--delays', type=float, nargs='+', default=[0., .5, 2., 6., 12., 20.])
p.add_argument('--modes', nargs='+', choices=['continue','slow','hold','learned'], default=['continue','slow','hold','learned'])
p.add_argument('--resume-steps', type=int, default=300)
p.add_argument('--controller', default=str(ROOT/'outputs/latency_controller_v1/train_v2_20261015/controller_v1_final.pt'))
p.add_argument('--request-cache')
p.add_argument('--replay-prefix',action='store_true')
p.add_argument('--antialiasing', choices=['Off','FXAA','DLSS','TAA','DLAA'])
p.add_argument('--deterministic-render', action='store_true')
p.add_argument('--analytic-render', action='store_true')
p.add_argument('--no-dither', action='store_true')
p.add_argument('--fixed-render-clock', action='store_true')
p.add_argument('--ambient-only', action='store_true')
p.add_argument('--cache-only', action='store_true')
p.add_argument('--audit-steps', type=int, default=0)
p.add_argument('--record-slots', type=int, nargs='*', default=[])
p.add_argument('--record-delays', type=float, nargs='*', default=[6.,20.])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True
assert a.num_envs % 6 == 0 and a.resume_steps >= 6
assert a.delays == [0.] and a.modes == ["continue"]
assert not a.request_cache and not a.replay_prefix and not a.cache_only
if a.replay_prefix and not a.cache_only:
    assert len(a.modes)==len(a.delays)==1, 'Prefix replay requires one branch per fresh process'
assert all(d >= 0 and abs(d*30-round(d*30)) < 1e-6 for d in a.delays)
if os.environ.get('PAIRED_RENDER_GPU'):
    a.kit_args = (a.kit_args+' --/renderer/activeGpu='+str(int(os.environ['PAIRED_RENDER_GPU']))).strip()
app = AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch, numpy as np, gymnasium as gym
import BrainCo_DexHand
from isaaclab_tasks.utils import parse_env_cfg
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch
from student_features import _semantic_rgb_features
from symbol_scene import SymbolScene
from latency_controller_v1 import Supervisor, normalized_target, command, sensor_observation
from latency_request_cache_v1 import capture, restore
from rollout_video import RolloutVideo
from PIL import Image, ImageDraw

def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def compact_stats(rows):
    ok = [r for r in rows if r['eligible']]; survivors = [r for r in ok if not r['wait_drop']]
    return dict(nominal=len(rows),eligible=len(ok),prefix_failures=len(rows)-len(ok),
        wait_drops=sum(r['wait_drop'] for r in ok),wait_survivors=len(survivors),
        wait_survival_rate=len(survivors)/len(ok) if ok else None,
        recovery_drops=sum(r['resume_drop'] for r in ok),successes=sum(r['success'] for r in ok),
        success_rate=sum(r['success'] for r in ok)/len(ok) if ok else None,
        recovery_success_given_survival=sum(r['success'] for r in survivors)/len(survivors) if survivors else None)

dest = Path(a.output); dest.mkdir(parents=True,exist_ok=True)
if (dest/'manifest.json').exists(): raise RuntimeError('Fresh output directory required')
env = None; recorders = []; started = time.monotonic(); allrows = []
try:
    device=a.device; n=a.num_envs
    student_path=ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'
    adapter_path=ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'
    ck=torch.load(student_path,map_location=device,weights_only=False)
    freeze=json.loads((ROOT/'configs/visual_student_freeze.json').read_text())
    assert ck['freeze_id']==freeze['freeze_id'] and float(ck['action_scale'])==1.
    student=VisualLanguageStudent(ck['rgb_dim'],ck['language_dim'],ck['proprio_dim'],ck['action_dim'],memory_mode=ck['memory_mode']).to(device).eval()
    student.load_state_dict(ck['model']);student.requires_grad_(False)
    text=torch.load(adapter_path,map_location=device,weights_only=False)['canonical_embeddings'].float()
    pack=torch.load(a.controller,map_location=device,weights_only=False)
    assert pack['manifest']['student_sha256']==sha(student_path) and pack['manifest']['adapter_sha256']==sha(adapter_path)
    controller=Supervisor(pack['obs_dim']).to(device).eval();controller.load_state_dict(pack['model']);controller.requires_grad_(False)
    cfg=parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',device=device,num_envs=n)
    cfg.tiled_camera.width=512;cfg.tiled_camera.height=512
    if a.antialiasing is not None:cfg.sim.render.antialiasing_mode=a.antialiasing
    if a.deterministic_render:
        # NVIDIA Isaac Lab's RTX reproducibility settings. Applied before scene construction.
        cfg.sim.render.carb_settings={
            '/rtx/rendermode':'RealTimePathTracing',
            '/rtx/rtpt/cached/enabled':False,
            '/rtx/rtpt/lightcache/cached/enabled':False,
        }
    if a.analytic_render:
        cfg.sim.render.rendering_mode='performance'
        cfg.sim.render.antialiasing_mode='Off'
        cfg.sim.render.carb_settings={'/rtx/rendermode':'RaytracedLighting'}
    if a.no_dither:
        cfg.sim.render.carb_settings=dict(cfg.sim.render.carb_settings or {},**{'/rtx/post/tonemap/dither':0.0})
    cfg.seed=a.seed;cfg.max_consecutive_success=0;cfg.freeze_goal_for_episode=True
    cfg.record_eval_metrics=True;cfg.goal_yaw=0.;cfg.episode_length_s=120.
    env=gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',cfg=cfg);raw=env.unwrapped
    import carb
    settings=carb.settings.get_settings()
    if a.analytic_render:
        # Replicator AA initialization re-enables sampled lighting; set this last.
        settings.set('/rtx/directLighting/sampledLighting/enabled',False)
        settings.set('/rtx/shadows/enabled',False)
    if a.ambient_only:
        for key in ['/rtx/directLighting/enabled','/rtx/post/histogram/enabled','/rtx/post/colorcorr/enabled','/rtx/post/colorgrad/enabled','/rtx/post/tvNoise/enabled']:
            settings.set_bool(key,False)
    selected_settings=['/rtx/directLighting/enabled','/rtx/post/histogram/enabled','/rtx/post/tvNoise/enabled','/rtx/shadows/enabled','/rtx/post/tonemap/dither','/rtx/rendermode','/rtx/rtpt/cached/enabled','/rtx/rtpt/lightcache/cached/enabled','/rtx/directLighting/sampledLighting/enabled','/rtx/indirectDiffuse/enabled','/rtx/domeLight/upperLowerStrategy','/rtx/raytracing/subpixel/mode','/rtx/raytracing/cached/enabled','/rtx/post/aa/op']
    save(dest/'render_settings.json',{k:settings.get(k) for k in selected_settings})
    assert abs(raw.step_dt-1/30)<1e-7
    raw.cfg.sim.render_interval=1000000;symbols=SymbolScene(raw)
    torch.manual_seed(a.seed);np.random.seed(a.seed);env.reset(seed=a.seed)
    goals=torch.arange(n,device=device)%6
    for face in range(6):
        raw.cfg.fixed_target_face=face;raw._reset_target_pose((goals==face).nonzero().flatten())
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
    raw._compute_intermediate_values();raw._update_face_markers()
    layout=np.stack([np.random.default_rng(a.seed+i).permutation(6) for i in range(n)])
    symbols.set_layout(layout)
    prefix=30*(1+a.seed%3)
    sources=[Path(__file__), ROOT/'scripts/agentic/latency_controller_v1.py',ROOT/'scripts/agentic/latency_request_cache_v1.py',ROOT/'scripts/agentic/student_features.py']
    manifest=dict(experiment='baseline_validity_diagnostic_v1',args=vars(a),seed=a.seed,prefix_steps=prefix,
        student_sha256=sha(student_path),adapter_sha256=sha(adapter_path),controller_sha256=sha(a.controller),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
        goal='already delivered canonical goal, reaffirmed at fixed response time; no live VLM/API',
        physics_hz=120,control_hz=30,vision_hz=15,gate_hz=2,camera_resolution=512,student_resolution=256,render_antialiasing=a.antialiasing,
        matching='fresh simulator; natural prefix followed immediately by zero-delay continuation; no state overwrite',
        render_profile='ambient-only formal profile' if a.ambient_only else 'original default renderer',render_settings=json.loads((dest/'render_settings.json').read_text()),
        contact_solver_limit='no reconstruction or intermediate state overwrite in this diagnostic',
        success='after response: orientation error <=0.16 rad for six consecutive control steps, no prior drop',
        recovery_window_steps=a.resume_steps,layout=layout.tolist(),
        hold='fixed commanded joint target; active low-level position servo, not zero torque',
        slow='15 percent interpolation toward fresh student target',
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    save(dest/'manifest.json',manifest)
    ih=torch.zeros(n,ck['history'],ck['rgb_dim'],device=device);ph=torch.zeros(n,ck['history'],ck['proprio_dim'],device=device)
    visual=previous_visual=None;rgb=None
    def observe(first=False):
        global ih,ph,visual,previous_visual,rgb
        if a.fixed_render_clock:
            settings.set_bool('/rtx/animationTimeUseWallclock',False)
            settings.set_float('/rtx/animationTime',raw.common_step_counter/30)
            settings.set_int('/rtx/externalFrameCounter',int(raw.common_step_counter)//2)
            settings.set_bool('/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges',True)
        symbols.update();cam=raw.capture_camera(refresh=True);rgb=cam['rgb'][...,:3].to(torch.uint8)
        if a.audit_steps and int(raw.common_step_counter)<a.audit_steps:
            torch.save({'rgb':rgb.cpu(),'depth':cam['depth'].cpu()},dest/f'camera_{raw.common_step_counter}.pt')
        srgb=torch.nn.functional.interpolate(rgb.permute(0,3,1,2).float(),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
        depth=cam['depth'];depth=depth.unsqueeze(-1) if depth.ndim==3 else depth
        depth=torch.nn.functional.interpolate(depth.permute(0,3,1,2),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
        feat=_semantic_rgb_features(srgb,16,depth,16);small=_semantic_rgb_features(srgb,4,depth,4)
        previous_visual=small.clone() if visual is None else visual;visual=small
        prop=raw.compute_student_proprio().float()
        ih=torch.cat((ih[:,1:],feat[:,None]),1);ph=torch.cat((ph[:,1:],prop[:,None]),1)
        if first:ih[:]=feat[:,None];ph[:]=prop[:,None]
    with torch.inference_mode():
        replay_error=None
        if a.request_cache:
            state=torch.load(a.request_cache,map_location='cpu',weights_only=False)
            assert state['student_sha256']==manifest['student_sha256'] and state['adapter_sha256']==manifest['adapter_sha256']
            assert state['joint_pos'].shape[0]==n and state['rgb'].shape[1:3]==(512,512)
            assert torch.equal(state['goals'],goals.cpu())
            layout=state['layout'].numpy();prefix=state['prefix_steps'];symbols.set_layout(layout)
            manifest.update(prefix_steps=prefix,layout=layout.tolist(),external_request_cache_sha256=sha(a.request_cache))
            save(dest/'manifest.json',manifest)
            if a.replay_prefix:
                assert state['prefix_actions'].shape[0]==prefix
                for step in range(prefix+1):
                    if step%2==0:observe(first=step==0)
                    if step==prefix:break
                    env.step(state['prefix_actions'][step].to(device))
                actual=capture(raw,ih,ph,visual,previous_visual,{'rgb':rgb},state['eligible'],manifest)
                keys=('joint_pos','joint_vel','object_root','prev_targets','cur_targets','hand_dof_targets','actions','prev_actions')
                replay_errors={key:float((actual[key]-state[key]).abs().max()) for key in keys}
                replay_error=max(replay_errors.values())
                save(dest/'prefix_replay_audit.json',dict(errors=replay_errors,max_abs_error=replay_error,steps=prefix))
                if replay_error!=0.:raise RuntimeError(f'Prefix mechanical replay differs: {replay_errors}')
        else:
            active=torch.ones(n,dtype=torch.bool,device=device);prefix_actions=[]
            prefix_reached=torch.zeros_like(active);prefix_min_error=torch.full((n,),float('inf'),device=device)
            prefix_run=torch.zeros(n,dtype=torch.long,device=device);prefix_max_run=prefix_run.clone()
            for step in range(prefix+1):
                if step%2==0:observe(first=step==0)
                if step==prefix:break
                proposal=student(VisualStudentBatch(ih,text[goals],ph))['action']
                prefix_action=torch.where(active[:,None],proposal,normalized_target(raw))
                prefix_actions.append(prefix_action.cpu().clone())
                _,_,term,trunc,info=env.step(prefix_action)
                met=info['semantic_metrics']; valid=active & ~met['dropped'].bool()
                prefix_min_error=torch.minimum(prefix_min_error,torch.where(valid,met['orientation_error'],float('inf')))
                reached=valid & (met['orientation_error']<=.16);prefix_reached|=reached
                prefix_run=torch.where(reached,prefix_run+1,0);prefix_max_run=torch.maximum(prefix_max_run,prefix_run)
                active &= ~(term|trunc|info['semantic_metrics']['dropped'].bool())
            state=capture(raw,ih,ph,visual,previous_visual,{'rgb':rgb},active,manifest)
            state.update(layout=torch.as_tensor(layout),goals=goals.cpu(),prefix_steps=prefix,prefix_actions=torch.stack(prefix_actions))
        torch.save(state,dest/'request_state.pt');cache_sha=sha(a.request_cache) if a.request_cache else sha(dest/'request_state.pt')
        save(dest/'request.json',dict(seed=a.seed,eligible=state['eligible'].tolist(),prefix_steps=prefix,cache_sha256=cache_sha))
        audits=[]
        # Rotate branch order across seeds and delays, keeping all four methods paired.
        for di,delay in enumerate([] if a.cache_only else a.delays):
            shift=(a.seed+di)%len(a.modes);modes=a.modes[shift:]+a.modes[:shift]
            for mode in modes:
                torch.manual_seed(a.seed);np.random.seed(a.seed)
                error=0.0  # Natural continuation: do not write simulator state here.
                symbols.set_layout(layout);symbols.update()
                ih=state['ih'].to(device).clone();ph=state['ph'].to(device).clone()
                visual=state['visual'].to(device).clone();previous_visual=state['oldvisual'].to(device).clone();rgb=state['rgb'].to(device).clone()
                active=state['eligible'].to(device).clone();eligible=active.clone()
                anchor=normalized_target(raw).clone();lastmode=torch.zeros(n,dtype=torch.long,device=device);chosen=lastmode.clone()
                wait_drop=torch.zeros_like(active);resume_drop=wait_drop.clone();succeeded=wait_drop.clone()
                good_count=torch.zeros(n,dtype=torch.long,device=device);completion=torch.full_like(good_count,-1)
                ever_reached=torch.zeros_like(active);max_good_count=good_count.clone()
                minimum_error=torch.full((n,),float('inf'),device=device);first_reach=torch.full_like(good_count,-1)
                drop_step=torch.full_like(good_count,-1);counts=torch.zeros(n,3,dtype=torch.long,device=device);switches=good_count.clone()
                handoff_error=torch.full((n,),float('nan'),device=device);handoff_speed=handoff_error.clone()
                wait_steps=round(delay*30);decisions=[];recorders=[];diagnostic=[];trajectory=hashlib.sha256()
                videos={i:RolloutVideo(dest/f'{delay:g}_{mode}_case{i}.mp4') for i in a.record_slots if delay in a.record_delays}
                recorders=list(videos.values());counter=int(raw.common_step_counter)
                for step in range(wait_steps+a.resume_steps):
                    if step>0 and step%2==0:observe()
                    proposal=student(VisualStudentBatch(ih,text[goals],ph))['action'];current=normalized_target(raw)
                    pending=step<wait_steps
                    if step==wait_steps:
                        from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance
                        handoff_error=torch.where(active,rotation_distance(raw.object_rot,raw.goal_rot),handoff_error)
                        handoff_speed=torch.where(active,raw.object_linvel.norm(dim=-1),handoff_speed)
                    if pending:
                        if step%15==0:
                            age=torch.full((n,),step/30,device=device)
                            obs=sensor_observation(visual,previous_visual,raw.compute_student_proprio().float(),proposal,current,anchor,lastmode,age)
                            dist=controller.distribution(obs)
                            chosen=dist.logits.argmax(-1) if mode=='learned' else torch.full_like(chosen,{'continue':0,'slow':1,'hold':2}[mode])
                            switches+=((chosen!=lastmode)&active).long()
                            decisions.append(dict(step=step,selected=chosen.cpu().tolist(),active=active.cpu().tolist(),probabilities=dist.probs.cpu().tolist() if mode=='learned' else None))
                        actions,anchor=command(proposal,current,anchor,chosen,lastmode)
                        counts+=torch.nn.functional.one_hot(chosen,3)*active[:,None];lastmode=chosen.clone()
                    else:actions=proposal
                    actions=torch.where(active[:,None],actions,current)
                    assert torch.isfinite(actions).all() and float(actions.abs().max())<=1.00001
                    if step%2==0:
                        for i,writer in videos.items():
                            if bool(active[i]):
                                im=Image.fromarray(rgb[i].cpu().numpy());draw=ImageDraw.Draw(im);draw.rectangle((0,0,512,67),fill='black')
                                draw.text((5,5),f'{mode}  delay={delay:g}s  elapsed={step/30:.2f}s',fill='white')
                                draw.text((5,24),'WAIT: physics continues' if pending else 'RESPONSE: frozen student resumes',fill='white')
                                draw.text((5,43),f'Fixed response / no API; seed={a.seed} slot={i}',fill='white');writer.append(np.asarray(im))
                    if step<a.audit_steps:
                        before_state=torch.cat((raw.hand.data.joint_pos,raw.hand.data.joint_vel,raw.object.data.root_state_w),-1).cpu().tolist()
                    for value in (raw.hand.data.joint_pos,raw.hand.data.joint_vel,raw.object.data.root_state_w,ih,ph,visual,actions,active):
                        trajectory.update(value.contiguous().cpu().numpy().tobytes())
                    before=active.clone();_,_,term,trunc,info=env.step(actions);metrics=info['semantic_metrics']
                    for value in (raw.hand.data.joint_pos,raw.hand.data.joint_vel,raw.object.data.root_state_w):
                        trajectory.update(value.contiguous().cpu().numpy().tobytes())
                    if step<a.audit_steps:
                        diagnostic.append(dict(step=step,before=before_state,action=actions.cpu().tolist(),after=torch.cat((raw.hand.data.joint_pos,raw.hand.data.joint_vel,raw.object.data.root_state_w),-1).cpu().tolist(),visual=visual.cpu().tolist(),proprio=raw.compute_student_proprio().cpu().tolist()))
                        save(dest/f'audit_steps_{delay:g}_{mode}.json',diagnostic)
                    drop=metrics['dropped'].bool()&before;drop_step[drop]=step+1
                    if bool((trunc&before).any()):raise RuntimeError('Unexpected timeout')
                    if bool((term&before&~drop).any()):raise RuntimeError('Unexpected non-drop termination')
                    if pending:wait_drop|=drop
                    else:
                        resume_drop|=drop;good=(metrics['orientation_error']<=.16)&before&~drop
                        minimum_error=torch.minimum(minimum_error,torch.where(before & ~drop,metrics['orientation_error'],float('inf')))
                        first_reach=torch.where(good & ~ever_reached,step+1,first_reach);ever_reached|=good
                        good_count=torch.where(good,good_count+1,torch.zeros_like(good_count))
                        max_good_count=torch.maximum(max_good_count,good_count)
                        new=(good_count>=6)&before;succeeded|=new;completion[new]=step-wait_steps+1;active&=~new
                    active&=~drop
                    if step%100==0:
                        save(dest/'progress.json',dict(status='running',seed=a.seed,mode=mode,delay_s=delay,step=step,total_steps=wait_steps+a.resume_steps,completed_trials=len(allrows),active=int(active.sum()),updated=time.time()))
                    # Keep batch length fixed across branches to avoid changing reset timing.
                assert int(raw.common_step_counter)-counter==wait_steps+a.resume_steps
                for writer in recorders:writer.close()
                recorders=[]
                rows=[]
                for i in range(n):
                    valid=bool(eligible[i])
                    rows.append(dict(seed=a.seed,slot=i,goal=int(goals[i]),prefix_steps=prefix,delay_s=delay,mode=mode,eligible=valid,
                        wait_drop=bool(wait_drop[i]),resume_drop=bool(resume_drop[i]),success=bool(succeeded[i]),
                        prefix_single_step_reach=bool(prefix_reached[i]),prefix_max_consecutive_reach=int(prefix_max_run[i]),
                        prefix_min_error=float(prefix_min_error[i]) if torch.isfinite(prefix_min_error[i]) else None,
                        post_single_step_reach=bool(ever_reached[i]),post_max_consecutive_reach=int(max_good_count[i]),
                        post_min_error=float(minimum_error[i]) if torch.isfinite(minimum_error[i]) else None,
                        post_first_reach_step=int(first_reach[i]),
                        resume_steps_to_success=int(completion[i]),drop_time_from_request_s=int(drop_step[i])/30 if int(drop_step[i])>=0 else None,
                        mode_steps=counts[i].cpu().tolist(),mode_switches=int(switches[i]),
                        handoff_orientation_error=float(handoff_error[i]) if torch.isfinite(handoff_error[i]) else None,
                        handoff_linear_speed=float(handoff_speed[i]) if torch.isfinite(handoff_speed[i]) else None,
                        request_cache_sha256=cache_sha,request_restore_max_abs_error=error))
                allrows+=rows;audits.append(dict(delay_s=delay,mode=mode,max_abs_error=error,cache_sha256=cache_sha,trajectory_sha256=trajectory.hexdigest()))
                save(dest/f'trials_{delay:g}_{mode}.json',rows);save(dest/f'decisions_{delay:g}_{mode}.json',decisions)
                save(dest/'eval_trials.json',allrows);save(dest/'state_audit.json',audits)
                print(json.dumps(dict(completed_branch=mode,delay_s=delay,seed=a.seed,**compact_stats(rows))),flush=True)
        zero=[r for r in allrows if r['delay_s']==0]
        zero_ok=all(len({(r['success'],r['wait_drop'],r['resume_drop'],r['resume_steps_to_success']) for r in zero if r['slot']==i})<=1 for i in range(n)) if zero and len({r['mode'] for r in zero})>1 else None
        if zero_ok is False:raise RuntimeError('Zero-delay identical-control branches disagree; inspect solver reconstruction')
        save(dest/'complete.json',dict(status='completed',trials=len(allrows),eligible_request_states=int(state['eligible'].sum()),zero_delay_consistent=zero_ok,elapsed_s=time.monotonic()-started))
        save(dest/'progress.json',dict(status='completed',completed_trials=len(allrows),updated=time.time()))
except BaseException as error:
    save(dest/'failure.json',dict(error=str(error),type=type(error).__name__,completed_trials=len(allrows),updated=time.time()));raise
finally:
    for writer in recorders:writer.close()
    if env is not None:env.close()
    app.close()
