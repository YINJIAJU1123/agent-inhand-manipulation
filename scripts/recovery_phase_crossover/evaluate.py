"""Phase crossover evaluator: frozen, independently selected wait/recovery actors.

Known delivered goals are reaffirmed by simulated responses. No live VLM calls.
Evaluation replays a cached prefix in a fresh process; it never writes an
intermediate physical state back into the contact solver.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/BrainCo_DexHand'),str(ROOT/'scripts/agentic')]
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--num-envs',type=int,default=12)
p.add_argument('--seed',type=int,default=920000)
p.add_argument('--iterations',type=int,default=32)
p.add_argument('--train-delays',nargs='+',type=float,default=[2.,6.,12.])
p.add_argument('--objective',choices=['completion','wait_survival'],default='completion')
p.add_argument('--mode',choices=['train','collect','cache','continue','slow','hold','slow_smooth','hold_smooth','learned','wait_only'],default='train')
p.add_argument('--delay',type=float,default=6.)
p.add_argument('--checkpoint')
p.add_argument('--recovery-checkpoint',required=True)
p.add_argument('--request-cache')
p.add_argument('--goal-branch',type=int,choices=range(6),default=0)
p.add_argument('--noise-seed',type=int,default=184000)
p.add_argument('--collection')
p.add_argument('--draw-id',type=int,choices=range(6),default=0)
p.add_argument('--training-arm',default='evaluation')
p.add_argument('--resume-steps',type=int,default=300)
p.add_argument('--bridge-steps',type=int,default=60)
p.add_argument('--record-slots',nargs='*',type=int,default=[])
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.enable_cameras=True
if a.mode!='learned':raise ValueError('Phase crossover supports evaluation of learned checkpoints only')
assert a.num_envs%6==0 and a.bridge_steps==60 and a.resume_steps==300
assert a.iterations>0 and all(d>0 and round(d*30)%2==0 for d in a.train_delays)
assert a.delay>=0 and abs(a.delay*30-round(a.delay*30))<1e-6
if a.mode not in ('train','cache') and not a.request_cache:raise ValueError('Evaluation requires a frozen prefix cache')
if a.mode in ('learned','wait_only','collect') and not a.checkpoint:raise ValueError('Learned evaluation requires a checkpoint')
if os.environ.get('PAIRED_RENDER_GPU'):
    a.kit_args=(a.kit_args+' --/renderer/activeGpu='+str(int(os.environ['PAIRED_RENDER_GPU']))).strip()
if a.mode=='train':raise ValueError('Use paired runner for frozen-policy collection and grouped updates')
if a.mode=='collect' and not a.collection:raise ValueError('Collection output required')
dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=True)
if (dest/'manifest.json').exists():raise RuntimeError('Fresh output directory required')

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch
import numpy as np
import gymnasium as gym
import carb
import BrainCo_DexHand
from isaaclab_tasks.utils import parse_env_cfg
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent,VisualStudentBatch
from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance
from student_features import _semantic_rgb_features
from symbol_scene import SymbolScene
from latency_controller_v1 import normalized_target
from latency_request_cache_v1 import capture
from recovery_goal_pairing import paired_goal_ids, switch_goal, tensor_digest
from phase_policy import load_recovery_actor,select_controller
from recovery_controller_v3 import RecoveryController,observation,bridge_fraction,residual_command,fixed_command,update,make_optimizers,OutcomeReplay

RENDER_KEYS=['/rtx/directLighting/enabled','/rtx/post/histogram/enabled','/rtx/post/tvNoise/enabled','/rtx/shadows/enabled','/rtx/post/tonemap/dither','/rtx/rendermode','/rtx/rtpt/cached/enabled','/rtx/rtpt/lightcache/cached/enabled','/rtx/directLighting/sampledLighting/enabled','/rtx/indirectDiffuse/enabled','/rtx/domeLight/upperLowerStrategy','/rtx/raytracing/subpixel/mode','/rtx/raytracing/cached/enabled','/rtx/post/aa/op']
env=None;writers={};started=time.monotonic()

def summarize(rows):
    valid=[r for r in rows if r['eligible']]
    return dict(nominal=len(rows),eligible=len(valid),prefix_failures=len(rows)-len(valid),
        successes=sum(r['success'] for r in valid),success_rate=sum(r['success'] for r in valid)/len(valid) if valid else None,
        wait_drops=sum(r['wait_drop'] for r in valid),resume_drops=sum(r['resume_drop'] for r in valid))

try:
    device=a.device;n=a.num_envs
    student_path=ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'
    adapter_path=ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'
    ck=torch.load(student_path,map_location=device,weights_only=False)
    contract=json.loads((ROOT/'configs/visual_student_freeze.json').read_text())
    assert ck['freeze_id']==contract['freeze_id'] and float(ck['action_scale'])==1.
    student=VisualLanguageStudent(ck['rgb_dim'],ck['language_dim'],ck['proprio_dim'],ck['action_dim'],memory_mode=ck['memory_mode']).to(device).eval()
    student.load_state_dict(ck['model']);student.requires_grad_(False)
    text=torch.load(adapter_path,map_location=device,weights_only=False)['canonical_embeddings'].float()
    cfg=parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',device=device,num_envs=n)
    cfg.tiled_camera.width=512;cfg.tiled_camera.height=512
    cfg.seed=a.seed;cfg.max_consecutive_success=0;cfg.freeze_goal_for_episode=True
    cfg.record_eval_metrics=True;cfg.goal_yaw=0.;cfg.episode_length_s=120.
    env=gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',cfg=cfg);raw=env.unwrapped
    assert abs(raw.step_dt-1/30)<1e-7
    raw.cfg.sim.render_interval=1000000;symbols=SymbolScene(raw)
    settings=carb.settings.get_settings();render={k:settings.get(k) for k in RENDER_KEYS}
    save(dest/'render_settings.json',render)
    manifest=dict(version='recovery_factorial_v1',args=vars(a),student_sha256=sha(student_path),adapter_sha256=sha(adapter_path),
        checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None,render_settings=render,
        render_profile='original default renderer',physics_hz=120,control_hz=30,vision_hz=15,controller_hz=15,
        camera_resolution=512,student_resolution=256,bridge_steps=60,resume_steps=300,
        goal='correct goal changed after common cached prefix, delivered before waiting, reaffirmed at simulated response; no future unknown goal input',
        actor_inputs=['four compact RGB-D frames','proprioception','frozen student proposal','current target','request anchor','previous command','elapsed wait and response clocks','delivered 768D language embedding'],
        forbidden_actor_inputs=['ground-truth object pose or velocity','orientation error','face ID','future response','remaining delay'],
        critic_only=['orientation error','object distance','linear and angular velocity'],
        outcome_label='realized success under the collecting residual policy, then frozen student; uncalibrated on-policy prediction',
        objective=a.objective,full_reward='12 success, -8 drop, -3 horizon failure, .15 orientation progress, -.002 command change',
        survival_ablation='same actor/bridge and action penalty; +2 surviving wait, -8 wait drop; no recovery outcome, progress or outcome-head loss',
        success='orientation error <=.16 rad for six consecutive post-response control steps, no prior drop',
        zero_delay='bypass residual and bridge; every mode executes the identical frozen student',
        interpretation='pilot on known goals; no live agent, unknown goal, new-object or calibrated recoverability claim',
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'scripts/agentic/recovery_controller_v3.py',ROOT/'scripts/agentic/recovery_controller_v2.py',ROOT/'scripts/agentic/student_features.py',ROOT/'scripts/agentic/symbol_scene.py',ROOT/'scripts/agentic/latency_request_cache_v1.py']})
    manifest.update(initial_policy='exact fixed slow during wait, immediate student after response; learned residuals decay over two seconds',
        residual_bound=.02,optimizer='disjoint policy/value/outcome encoders and optimizers; value targets scaled by 12; exact KL <= .02 with rollback',
        outcome_training='episode-uniform replay; whole-seed train/validation split (seed % 5); behavior-policy mixture labels, not counterfactual oracle',
        source_v2_sha256=sha(ROOT/'scripts/agentic/train_recovery_controller_v2.py'))
    manifest.update(version='recovery_phase_crossover_v1',wait_checkpoint_sha256=sha(a.checkpoint),recovery_checkpoint_sha256=sha(a.recovery_checkpoint),phase_switch='At the response, change only frozen actor weights. Same observation, goal, command map and two-second handoff.')
    save(dest/'manifest.json',manifest)
    recovery_model=None;model=None;optimizer=None;outcome_replay=OutcomeReplay();ih=ph=vh=visual=previous_visual=rgb=None

    def observe(first=False):
        global ih,ph,vh,visual,previous_visual,rgb
        symbols.update();cam=raw.capture_camera(refresh=True);rgb=cam['rgb'][...,:3].to(torch.uint8)
        srgb=torch.nn.functional.interpolate(rgb.permute(0,3,1,2).float(),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
        depth=cam['depth'];depth=depth.unsqueeze(-1) if depth.ndim==3 else depth
        depth=torch.nn.functional.interpolate(depth.permute(0,3,1,2),size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
        feat=_semantic_rgb_features(srgb,16,depth,16);small=_semantic_rgb_features(srgb,4,depth,4)
        previous_visual=small.clone() if visual is None else visual;visual=small
        prop=raw.compute_student_proprio().float()
        if first:
            ih=feat[:,None].repeat(1,ck['history'],1);ph=prop[:,None].repeat(1,ck['history'],1);vh=small[:,None].repeat(1,4,1)
        else:
            ih=torch.cat((ih[:,1:],feat[:,None]),1);ph=torch.cat((ph[:,1:],prop[:,None]),1);vh=torch.cat((vh[:,1:],small[:,None]),1)

    def prepare(seed,cache_path=None):
        global ih,ph,vh,visual,previous_visual,rgb
        torch.manual_seed(seed);np.random.seed(seed);env.reset(seed=seed)
        goals=torch.arange(n,device=device)%6
        for face in range(6):
            raw.cfg.fixed_target_face=face;raw._reset_target_pose((goals==face).nonzero().flatten())
        raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
        raw._compute_intermediate_values();raw._update_face_markers()
        layout=np.stack([np.random.default_rng(seed+i).permutation(6) for i in range(n)])
        prefix=30*(1+seed%3);state=None
        if cache_path:
            state=torch.load(cache_path,map_location='cpu',weights_only=False)
            assert state['seed']==seed and state['prefix_steps']==prefix and state['joint_pos'].shape[0]==n
            assert state['student_sha256']==manifest['student_sha256'] and state['adapter_sha256']==manifest['adapter_sha256']
            assert state['render_settings']==render and torch.equal(state['goals'],goals.cpu())
            layout=state['layout'].numpy()
        symbols.set_layout(layout)
        visual=previous_visual=None;active=torch.ones(n,dtype=torch.bool,device=device);commands=[]
        for step in range(prefix+1):
            if step%2==0:observe(first=step==0)
            if step==prefix:break
            proposal=student(VisualStudentBatch(ih,text[goals],ph))['action']
            action=state['prefix_actions'][step].to(device) if state else torch.where(active[:,None],proposal,normalized_target(raw))
            commands.append(action.cpu().clone())
            _,_,term,trunc,info=env.step(action)
            active &= ~(term|trunc|info['semantic_metrics']['dropped'].bool())
        actual=capture(raw,ih,ph,visual,previous_visual,{'rgb':rgb},active,manifest)
        if state:
            keys=('joint_pos','joint_vel','object_root','prev_targets','cur_targets','hand_dof_targets','actions','prev_actions')
            errors={key:float((actual[key]-state[key]).abs().max()) for key in keys}
            audit=dict(errors=errors,max_abs_error=max(errors.values()),eligible_equal=torch.equal(active.cpu(),state['eligible']))
            save(dest/'prefix_replay_audit.json',audit)
            if audit['max_abs_error']!=0. or not audit['eligible_equal']:raise RuntimeError('Physical prefix replay mismatch')
            ih=state['ih'].to(device);ph=state['ph'].to(device);vh=state['visual_history'].to(device)
            visual=state['visual'].to(device);previous_visual=state['oldvisual'].to(device);rgb=state['rgb'].to(device)
        else:
            state=actual
            state.update(seed=seed,layout=torch.as_tensor(layout),goals=goals.cpu(),prefix_steps=prefix,
                         prefix_actions=torch.stack(commands),visual_history=vh.cpu(),render_settings=render)
        return goals,active,state

    def run_batch(seed,delay,mode,training=False):
        global model,recovery_model,optimizer,writers
        with torch.no_grad():goals,active,state=prepare(seed,a.request_cache)
        if mode=='cache':
            torch.save(state,dest/'request_state.pt');return [],[],active
        prefix_goals=goals.clone()
        goals=paired_goal_ids(prefix_goals,a.goal_branch)
        goal_audit=switch_goal(raw,goals,state,ih,ph,vh)
        goal_audit.update(branch=a.goal_branch,prefix_goals=prefix_goals.cpu().tolist(),goals=goals.cpu().tolist(),
                          request_cache_sha256=sha(a.request_cache),eligible=active.cpu().tolist())
        save(dest/'goal_switch_audit.json',goal_audit)
        # Separate action sampling RNG from prefix reconstruction. Match streams
        # across factor cells by draw id, with independent draws within a group.
        torch.manual_seed(a.noise_seed);np.random.seed(a.noise_seed)
        wait_steps=round(delay*30);eligible=active.clone();anchor=normalized_target(raw).clone();previous_action=anchor.clone()
        wait_drop=torch.zeros_like(active);resume_drop=wait_drop.clone();success=wait_drop.clone();reached=wait_drop.clone()
        consecutive=torch.zeros(n,dtype=torch.long,device=device);completion=torch.full_like(consecutive,-1)
        rollout=[];predictions=[];decisions=[];diagnostic_trace=[];request_probe=None;speed_sum=torch.zeros(n,device=device);decision_count=torch.zeros_like(speed_sum)
        error0=rotation_distance(raw.object_rot,raw.goal_rot);held_raw=torch.zeros(n,22,device=device)
        counter=int(raw.common_step_counter)
        error_at_request=error0.clone();error_at_response=error0.clone();error_at_handoff=error0.clone();last_error=error0.clone()
        alive_at_response=torch.zeros_like(active);alive_at_handoff=torch.zeros_like(active)
        best_recovery_error=torch.full_like(error0,float('inf'))
        if a.record_slots and not training:
            from rollout_video import RolloutVideo
            from PIL import Image,ImageDraw
            writers={i:RolloutVideo(dest/f'{mode}_case{i}.mp4') for i in a.record_slots}
        for step in range(wait_steps+a.resume_steps):
            with torch.no_grad():
                if step>0 and step%2==0:observe()
                if step==wait_steps:
                    error_at_response=rotation_distance(raw.object_rot,raw.goal_rot).clone();alive_at_response=active.clone()
                if step==wait_steps+a.bridge_steps:
                    error_at_handoff=last_error.clone();alive_at_handoff=active.clone()
                proposal=student(VisualStudentBatch(ih,text[goals],ph))['action'];current=normalized_target(raw)
                pending=torch.full((n,),step<wait_steps,dtype=torch.bool,device=device)
                age=torch.full((n,),min(step,wait_steps)/30,device=device)
                response_age=torch.full((n,),max(0,step-wait_steps)/30,device=device)
                fraction=bridge_fraction(response_age,pending)
                controlled=wait_steps>0 and step<wait_steps+a.bridge_steps
                if training and a.objective=='wait_survival' and step==wait_steps:
                    rollout[-1]['reward']+=2.*active.float()
                if controlled and step%2==0 and (training or mode in ('learned','wait_only')):
                    obs=observation(vh,raw.compute_student_proprio().float(),proposal,current,anchor,previous_action,age,pending,response_age)
                    goal=text[goals]
                    if model is None:
                        # Initialization independent of training reset seed and objective.
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(181026);model=RecoveryController(obs.shape[-1]).to(device)
                        if a.checkpoint:
                            pack=torch.load(a.checkpoint,map_location=device,weights_only=False)
                            assert pack['obs_dim']==obs.shape[-1]
                            for k in ('student_sha256','adapter_sha256'):
                                assert pack['manifest'][k]==manifest[k]
                            model.load_state_dict(pack['model'])
                        optimizer=make_optimizers(model)
                        recovery_model=model if sha(a.checkpoint)==sha(a.recovery_checkpoint) else load_recovery_actor(a.recovery_checkpoint,device,obs.shape[-1],manifest)
                    active_model=select_controller(model,recovery_model,step,wait_steps)
                    if step==0:
                        alternatives=[]
                        for branch in range(6):
                            probe_goals=paired_goal_ids(prefix_goals,branch)
                            probe_proposal=student(VisualStudentBatch(ih,text[probe_goals],ph))['action']
                            probe_obs=observation(vh,raw.compute_student_proprio().float(),probe_proposal,current,anchor,previous_action,age,pending,response_age)
                            probe_raw=model.distribution(probe_obs,text[probe_goals]).mean
                            probe_action,probe_rate,_=residual_command(probe_proposal,current,probe_raw,fraction,pending)
                            alternatives.append(dict(branch=branch,goals=probe_goals.cpu().tolist(),student_proposal=probe_proposal.cpu().tolist(),
                                command=probe_action.cpu().tolist(),rate=probe_rate.cpu().tolist()))
                        request_probe=dict(eligible=active.cpu().tolist(),alternatives=alternatives,
                            interpretation='Identical sensor state, correct goal changed for student and controller, no simulator advance. Not physical success counterfactuals.')
                        save(dest/'paired_request_probe.json',request_probe)
                    dist=active_model.distribution(obs,goal)
                    held_raw=dist.sample() if training else dist.mean
                    predicted=active_model.outcome_logit(obs,goal).sigmoid()
                    if step in (0,wait_steps):
                        predictions.append(dict(step=step,phase='wait' if step==0 else 'arrival',probability=predicted.cpu().tolist(),active=active.cpu().tolist()))
                    if training:
                        priv=torch.cat((error0[:,None],(raw.object_pos-raw.in_hand_pos).norm(dim=-1)[:,None],raw.object_linvel,raw.object_angvel),-1)
                        rollout.append(dict(obs=obs.clone(),goal=goal.clone(),priv=priv.clone(),action=held_raw.clone(),
                            logprob=dist.log_prob(held_raw),mean=dist.mean.clone(),std=dist.base_dist.scale.clone(),value=model.value(obs,goal,priv),reward=torch.zeros(n,device=device),
                            done=torch.zeros(n,dtype=torch.bool,device=device),valid=active.clone()))
                if wait_steps==0 or not controlled:
                    action=proposal;speed=torch.ones(n,device=device)
                elif training or mode in ('learned','wait_only'):
                    action,speed,residual=residual_command(proposal,current,held_raw,fraction,pending)
                else:
                    action=fixed_command(mode,proposal,current,anchor,pending,fraction)
                    wait_rate=1. if mode=='continue' else .15 if mode.startswith('slow') else 0.
                    resume_rate=wait_rate+(1-wait_rate)*fraction if mode.endswith('_smooth') else torch.ones(n,device=device)
                    speed=torch.where(pending,torch.full((n,),wait_rate,device=device),resume_rate)
                if controlled and step%2==0:
                    speed_sum+=speed*active;decision_count+=active
                    if not training:
                        decisions.append(dict(step=step,policy_phase='wait' if step<wait_steps else 'recovery',active=active.cpu().tolist(),rate=speed.cpu().tolist(),
                            residual_l2=residual.norm(dim=-1).cpu().tolist() if mode in ('learned','wait_only') else None))
                action=torch.where(active[:,None],action,current)
                if not torch.isfinite(action).all() or action.abs().max()>1.00001:raise RuntimeError('Invalid action')
                if writers and step%2==0:
                    for i,writer in writers.items():
                        if bool(active[i]):
                            im=Image.fromarray(rgb[i].cpu().numpy());draw=ImageDraw.Draw(im);draw.rectangle((0,0,512,66),fill='black')
                            draw.text((5,5),f'{mode} delay={delay:g}s elapsed={step/30:.2f}s',fill='white')
                            draw.text((5,24),'WAIT' if step<wait_steps else 'RECOVERY; physics continues',fill='white')
                            draw.text((5,43),f'Simulated response, no API; seed={seed} slot={i}',fill='white');writer.append(np.asarray(im))
                before=active.clone();_,_,term,trunc,info=env.step(action);metrics=info['semantic_metrics']
                if not torch.equal(metrics['target_face'][before].long(),goals[before]):raise RuntimeError('Scorer goal mismatch')
                last_error=torch.where(before,metrics['orientation_error'],last_error)
                if step>=wait_steps:
                    best_recovery_error=torch.where(before,torch.minimum(best_recovery_error,metrics['orientation_error']),best_recovery_error)
                if not training and step%30==0:
                    diagnostic_trace.append(dict(step=step,phase='wait' if step<wait_steps else 'recovery',active=before.cpu().tolist(),
                        orientation_error=metrics['orientation_error'].cpu().tolist(),rate=speed.cpu().tolist(),
                        dropped=metrics['dropped'].bool().cpu().tolist()))
                drop=metrics['dropped'].bool()&before
                if (trunc&before).any() or (term&before&~drop).any():raise RuntimeError('Unexpected termination')
                new=torch.zeros_like(active)
                if step<wait_steps:wait_drop|=drop
                else:
                    resume_drop|=drop;good=(metrics['orientation_error']<=.16)&before&~drop;reached|=good
                    consecutive=torch.where(good,consecutive+1,torch.zeros_like(consecutive))
                    new=(consecutive>=6)&before;success|=new;completion[new]=step-wait_steps+1
                if training:
                    # During frozen continuation the final residual decision spans
                    # the remainder of the episode and receives its terminal outcome.
                    reward=torch.zeros(n,device=device)
                    if a.objective=='completion':
                        reward+=12.*new.float()-8.*drop.float()
                        reward+=.15*(error0-metrics['orientation_error'])*before*~drop
                    elif step<wait_steps:reward-=8.*drop.float()
                    if controlled:reward-=.002*(action-current).square().mean(-1)*before
                    rollout[-1]['reward']+=reward;rollout[-1]['done']|=drop|new
                active &= ~(drop|new);error0=metrics['orientation_error'];previous_action=action
            if step%100==0:
                save(dest/'rollout_progress.json',dict(seed=seed,delay=delay,mode=mode,step=step,total_steps=wait_steps+a.resume_steps,
                    alive=int(active.sum()),updated=time.time()))
        assert int(raw.common_step_counter)-counter==wait_steps+a.resume_steps
        if training:
            if a.objective=='completion':rollout[-1]['reward']-=3.*active.float()
            rollout[-1]['done'][:]=True
        for writer in writers.values():writer.close()
        writers={}
        rows=[dict(seed=seed,slot=i,goal=int(goals[i]),prefix_goal=int(prefix_goals[i]),goal_branch=a.goal_branch,noise_seed=a.noise_seed,delay_s=delay,mode=mode,draw_id=a.draw_id,training_arm=a.training_arm,eligible=bool(eligible[i]),
            orientation_error_at_request=float(error_at_request[i]),orientation_error_at_response=float(error_at_response[i]) if not bool(wait_drop[i]) else None,
            alive_at_response=bool(alive_at_response[i]),alive_at_handoff=bool(alive_at_handoff[i]),orientation_error_at_handoff=float(error_at_handoff[i]),last_orientation_error=float(last_error[i]),
            best_recovery_error=float(best_recovery_error[i]) if torch.isfinite(best_recovery_error[i]) else None,
            wait_drop=bool(wait_drop[i]),resume_drop=bool(resume_drop[i]),success=bool(success[i]),post_single_step_reach=bool(reached[i]),
            resume_steps_to_success=int(completion[i]),mean_controller_rate=float(speed_sum[i]/decision_count[i].clamp_min(1)),
            request_cache_sha256=sha(a.request_cache) if a.request_cache else None) for i in range(n)]
        if not training:
            save(dest/'outcome_predictions.json',predictions);save(dest/'decisions.json',decisions);save(dest/'diagnostic_trace.json',diagnostic_trace)
        return rows,rollout,success

    batch,rollout,success=run_batch(a.seed,a.delay,a.mode,training=a.mode=='collect')
    if a.mode!='cache':save(dest/'eval_trials.json',batch)
    if a.mode=='collect':
        export=Path(a.collection).resolve()
        torch.save(dict(rollout=[{k:v.detach().cpu() for k,v in row.items()} for row in rollout],success=success.cpu(),
            seed=a.seed,branch=a.goal_branch,draw_id=a.draw_id,training_arm=a.training_arm,delay=a.delay,noise_seed=a.noise_seed,checkpoint_sha256=manifest['checkpoint_sha256'],
            request_cache_sha256=sha(a.request_cache),manifest=manifest),export)
    save(dest/'complete.json',dict(status='completed',trials=len(batch),elapsed_s=time.monotonic()-started,**summarize(batch)))
except BaseException as exc:
    save(dest/'failure.json',dict(error=str(exc),type=type(exc).__name__,updated=time.time()));raise
finally:
    for writer in writers.values():writer.close()
    if env is not None:env.close()
    app.close()
