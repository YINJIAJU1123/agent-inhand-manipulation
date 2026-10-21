"""Train/evaluate a sensor-only supervisor with physics running throughout delay.

V1 scope: a verification request about the CURRENT delivered goal. Responses
reaffirm that goal. No live API, changed/unknown goal or response validity model.
"""
from pathlib import Path
import argparse, sys, json, time, hashlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/BrainCo_DexHand'))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--num-envs', type=int, default=24)
p.add_argument('--iterations', type=int, default=12)
p.add_argument('--eval-seeds', type=int, nargs='+', default=[62001,62002,62003])
p.add_argument('--eval-delays', type=float, nargs='+', default=[2.,8.,16.])
p.add_argument('--train-delays', type=float, nargs='+', default=[2.,4.,8.,12.,16.])
p.add_argument('--resume-steps', type=int, default=300)
p.add_argument('--checkpoint')
p.add_argument('--eval-only', action='store_true')
p.add_argument('--skip-eval', action='store_true')
p.add_argument('--eval-modes', nargs='+', default=['continue','slow','hold','learned'])
p.add_argument('--seed', type=int, default=51000)
p.add_argument('--record-slots', type=int, nargs='*', default=[])
p.add_argument('--request-cache')
p.add_argument('--cache-only',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(); a.enable_cameras=True
if a.request_cache and not a.eval_only: raise ValueError('Request caching is evaluation-only')
if a.cache_only and not a.request_cache: raise ValueError('cache-only requires request-cache')
if a.eval_only and 'learned' in a.eval_modes and not a.checkpoint:
    raise ValueError('Learned evaluation requires a trained checkpoint')
app=AppLauncher(dict(vars(a),multi_gpu=False)).app
import torch, numpy as np, gymnasium as gym
import BrainCo_DexHand
from isaaclab_tasks.utils import parse_env_cfg
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch
from student_features import _semantic_rgb_features
from symbol_scene import SymbolScene
from latency_controller_v1 import Supervisor, normalized_target, command, sensor_observation, ppo_update

def save_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

dest=Path(a.output); dest.mkdir(parents=True, exist_ok=True)
if (dest/'manifest.json').exists(): raise RuntimeError('Use a fresh output directory')
student_path=ROOT/'outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt'
adapter_path=ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'
device=a.device; n=a.num_envs
ck=torch.load(student_path,map_location=device,weights_only=False)
assert ck['action_scale']==1.
student=VisualLanguageStudent(ck['rgb_dim'],ck['language_dim'],ck['proprio_dim'],ck['action_dim'],memory_mode=ck['memory_mode']).to(device).eval()
student.load_state_dict(ck['model']); student.requires_grad_(False)
text=torch.load(adapter_path,map_location=device,weights_only=False)['canonical_embeddings'].float()
cfg=parse_env_cfg('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',device=device,num_envs=n)
cfg.tiled_camera.width=256; cfg.tiled_camera.height=256
cfg.seed=a.seed; cfg.max_consecutive_success=0; cfg.freeze_goal_for_episode=True
cfg.record_eval_metrics=True; cfg.goal_yaw=0.; cfg.episode_length_s=120.
env=gym.make('BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0',cfg=cfg); raw=env.unwrapped
symbols=SymbolScene(raw)
# Initialize Kit with its normal rendering dt. During rollouts camera capture
# explicitly renders every two control steps; suppress duplicate env.step renders.
raw.cfg.sim.render_interval=1000000
model=None; optimizer=None; history=[]; records=[]; starts={}; started=time.monotonic()
manifest={'experiment':'latency_supervisor_v1', 'args':vars(a), 'student_sha256':sha(student_path),
 'controller_sha256':sha(a.checkpoint) if a.checkpoint else None,
 'adapter_sha256':sha(adapter_path), 'step_dt':raw.step_dt, 'physics_continues_during_wait':True,
 'request':'simulated verification of last delivered goal; no live Astra call',
 'actor_inputs':['RGB-D compact features + temporal difference','robot proprioception','frozen student proposal',
 'current position target','latched hold target','previous mode','elapsed request age','pending flag'],
 'critic_only':['orientation error','object distance','object linear/angular velocity'],
 'request_start':'shared reconstructed physical state and identical student history' if a.request_cache else 'seeded reset followed by policy prefix; trajectories may diverge',
 'remaining_delay_in_actor':False, 'gate_stride':15, 'camera_stride':2, 'slow_target_interpolation':.15,
 'render_schedule':'explicit fresh camera render every two control steps',
 'hold':'latch commanded joint position target on hold entry; prescribed primitive',
 'success':'after response, orientation error <= .16 rad for six consecutive steps; no prior drop',
 'scoring':'prefix failures reported separately; success conditional on reaching request alive; all trials retained',
 'training':'PPO; wait survival + progress + drop penalty + terminal frozen-student recovery outcome',
 'source_sha256':{f:sha(Path(__file__).parent/f) for f in ['latency_controller_v1.py','train_latency_controller_v1.py','latency_request_cache_v1.py']}}
save_json(dest/'manifest.json',manifest)

def run_batch(seed, delay, mode, training=False):
    global model, optimizer
    torch.manual_seed(seed); np.random.seed(seed)
    env.reset(seed=seed)
    # All faces are balanced; delivered goal remains valid during verification.
    goals=torch.arange(n,device=device)%6
    for face in range(6):
        raw.cfg.fixed_target_face=face; raw._reset_target_pose((goals==face).nonzero().flatten())
    raw.cfg.fixed_target_face=None; raw.goal_markers.set_visibility(False)
    raw._compute_intermediate_values(); raw._update_face_markers()
    rng=np.random.default_rng(seed)
    symbols.set_layout(np.stack([rng.permutation(6) for _ in range(n)]))
    initial=torch.cat((raw.object_pos,raw.object_rot,raw.compute_student_proprio()),-1).detach().cpu()
    key=(seed,delay)
    if key in starts:
        difference=float((initial-starts[key]).abs().max())
        if difference>1e-5: raise RuntimeError(f'Unpaired reset: {difference}')
    else: starts[key]=initial.clone()
    prefix=30+(seed%3)*30
    waiting=round(delay/raw.step_dt)
    assert waiting%15==0
    active=torch.ones(n,dtype=torch.bool,device=device)
    eligible=active.clone(); wait_dropped=~active; resume_dropped=~active; succeeded=~active
    counts=torch.zeros(n,3,device=device,dtype=torch.long); holdcount=torch.zeros(n,device=device,dtype=torch.long)
    completion=torch.full((n,),-1,device=device,dtype=torch.long)
    switches=torch.zeros(n,device=device,dtype=torch.long); trace=[]
    writers={}; last_frames={}
    if a.record_slots and not training:
        from rollout_video import RolloutVideo
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        for slot in a.record_slots:
            if not 0<=slot<n: raise ValueError('Invalid recording slot')
            writers[slot]=RolloutVideo(dest/f'{seed}_{delay:g}_{mode}_case{slot}.mp4',width=576,height=320)
    start_counter=int(raw.common_step_counter)
    ih=torch.zeros(n,ck['history'],ck['rgb_dim'],device=device)
    ph=torch.zeros(n,ck['history'],ck['proprio_dim'],device=device)
    anchor=normalized_target(raw).clone(); previous_mode=torch.zeros(n,device=device,dtype=torch.long)
    chosen=previous_mode.clone(); rollout=[]; visual=None; oldvisual=None; last_error=None
    request_initial=None
    for step in range(prefix+waiting+a.resume_steps):
        pending=prefix<=step<prefix+waiting
        with torch.no_grad():
            prop=raw.compute_student_proprio().float()
            if step%2==0:
                symbols.update(); camera=raw.capture_camera()
                feat=_semantic_rgb_features(camera['rgb'][...,:3],16,camera['depth'],16)
                compact=_semantic_rgb_features(camera['rgb'][...,:3],4,camera['depth'],4)
                oldvisual=compact.clone() if visual is None else visual
                visual=compact
                ih=torch.cat((ih[:,1:],feat[:,None]),1); ph=torch.cat((ph[:,1:],prop[:,None]),1)
                if step==0: ih[:]=feat[:,None]; ph[:]=prop[:,None]
            proposal=student(VisualStudentBatch(ih,text[goals],ph))['action']
            current=normalized_target(raw)
            if step==prefix:
                if a.request_cache:
                    from latency_request_cache_v1 import capture,restore
                    cache_root=Path(a.request_cache); cache_root.mkdir(parents=True,exist_ok=True)
                    cache_path=cache_root/f'request_{seed}.pt'
                    if a.cache_only:
                        if cache_path.exists():raise RuntimeError('Refusing to overwrite frozen request cache')
                        state=capture(raw,ih,ph,visual,oldvisual,camera,active,manifest)
                        torch.save(state,cache_path)
                    if not cache_path.exists():raise RuntimeError('Generate all shared request caches before evaluation')
                    state=torch.load(cache_path,map_location=device,weights_only=False)
                    assert state['student_sha256']==manifest['student_sha256'] and state['adapter_sha256']==manifest['adapter_sha256']
                    reconstruction_error=restore(env,raw,state,seed,goals)
                    ih=state['ih'].clone(); ph=state['ph'].clone()
                    visual=state['visual'].clone(); oldvisual=state['oldvisual'].clone();camera={'rgb':state['rgb']}
                    prop=raw.compute_student_proprio().float()
                    proposal=student(VisualStudentBatch(ih,text[goals],ph))['action']
                    current=normalized_target(raw); active=state['eligible'].clone()
                    print(json.dumps({'request_restore':seed,'mode':mode,'max_state_error':reconstruction_error}),flush=True)
                eligible=active.clone(); anchor=current.clone()
                request_initial=torch.cat((raw.object_pos,raw.object_rot,prop),-1).detach().cpu()
                if a.cache_only:break
            if pending and (step-prefix)%15==0:
                age=torch.full((n,), (step-prefix)*raw.step_dt,device=device)
                obs=sensor_observation(visual,oldvisual,prop,proposal,current,anchor,previous_mode,age)
                # Critic/reward data is never passed to model.distribution().
                from BrainCo_DexHand.tasks.direct.inhand_manipulation_env import rotation_distance
                error=rotation_distance(raw.object_rot,raw.goal_rot)
                distance=(raw.object_pos-raw.in_hand_pos).norm(dim=-1)
                priv=torch.cat((error[:,None],distance[:,None],raw.object_linvel,raw.object_angvel),-1)
                if model is None:
                    model=Supervisor(obs.shape[-1]).to(device)
                    if a.checkpoint:
                        loaded=torch.load(a.checkpoint,map_location=device,weights_only=False)
                        for key in ('student_sha256','adapter_sha256'):
                            if loaded['manifest'][key]!=manifest[key]:raise ValueError('Controller/frozen student contract mismatch: '+key)
                        model.load_state_dict(loaded['model'])
                    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
                dist=model.distribution(obs)
                chosen=dist.sample() if training else dist.logits.argmax(-1) if mode=='learned' else torch.full_like(chosen,{'continue':0,'slow':1,'hold':2}[mode])
                switches+=((chosen!=previous_mode)&active).long()
                if mode=='learned':
                    trace.append({'step':step,'age_s':float(age[0]),'mode':chosen.cpu().tolist(),
                                  'probabilities':dist.probs.cpu().tolist(),'active':active.cpu().tolist()})
                if training:
                    rollout.append(dict(obs=obs.detach().clone(),priv=priv.detach().clone(),action=chosen.clone(),
                        logprob=dist.log_prob(chosen).detach(),value=model.value(obs,priv).detach(),
                        reward=torch.zeros(n,device=device),done=torch.zeros(n,dtype=torch.bool,device=device),valid=active.clone()))
                last_error=error
            if pending:
                action,anchor=command(proposal,current,anchor,chosen,previous_mode)
                previous_mode=chosen.clone()
                counts+=torch.nn.functional.one_hot(chosen,3)*active[:,None]
            else: action=proposal
            # Dead slots are never scored or reintroduced into training after auto-reset.
            action=torch.where(active[:,None],action,current)
            if writers and step%2==0 and (not a.request_cache or step>=prefix):
                phase='WAITING FOR VERIFICATION' if pending else 'PREFIX MANIPULATION' if step<prefix else 'RESPONSE RECEIVED: RESUME'
                for slot,writer in writers.items():
                    if not bool(active[slot]): continue
                    frame=Image.new('RGB',(576,320),(17,23,32));draw=ImageDraw.Draw(frame)
                    frame.paste(Image.fromarray(camera['rgb'][slot,...,:3].cpu().numpy()),(8,48))
                    lines=[phase,f'simulation: {step*raw.step_dt:.2f} s',f'wait duration: {delay:g} s',
                           f'wait age: {max(0,step-prefix)*raw.step_dt:.2f} s' if pending else '',
                           'action: '+(['continue','slow','hold'][int(chosen[slot])] if pending else 'frozen student'),
                           f'controller: {mode}',f'seed {seed} / slot {slot}', 'SIMULATED response; no API']
                    draw.text((8,14),'Continuous physics during reasoning delay',font=font,fill='white')
                    for j,line in enumerate(lines):draw.text((276,50+29*j),line,font=font,fill=(220,230,240))
                    array=np.asarray(frame);writer.append(array);last_frames[slot]=frame
            before=active.clone()
            _,_,terminated,truncated,info=env.step(action)
            metrics=info['semantic_metrics']; drop=metrics['dropped'].bool()&before
            if bool((truncated&before).any()): raise RuntimeError('Unexpected environment timeout')
            if pending:
                wait_dropped|=drop
                if training:
                    reward=.005*before.float()-8.*drop.float()+.2*(last_error-metrics['orientation_error'])*before
                    reward-=.002*(action-current).square().mean(-1)*before
                    rollout[-1]['reward']+=reward
                    rollout[-1]['done']|=drop
                last_error=metrics['orientation_error']
            elif step>=prefix+waiting:
                resume_dropped|=drop
                good=(metrics['orientation_error']<=.16)&before&~drop
                holdcount=torch.where(good,holdcount+1,torch.zeros_like(holdcount))
                new=(holdcount>=6)&before
                succeeded|=new; completion[new]=step-prefix-waiting+1
                active&=~new
            active&=~drop
        if step and step%300==0:
            print(json.dumps({'progress':mode,'seed':seed,'step':step,'total_steps':prefix+waiting+a.resume_steps,'alive':int(active.sum())}),flush=True)
    if training:
        # Attach post-response outcome to the final pending transition of survivors.
        survivors=eligible&~wait_dropped
        rollout[-1]['reward']+=survivors.float()*(2.+6.*succeeded.float()-4.*resume_dropped.float())
        rollout[-1]['done'][:]=True
    if not a.cache_only:assert int(raw.common_step_counter)-start_counter==prefix+waiting+a.resume_steps
    for slot,writer in writers.items():
        if slot in last_frames:
            frame=last_frames[slot].copy();draw=ImageDraw.Draw(frame)
            status='SUCCESS after response' if bool(succeeded[slot]) else 'DROP' if bool(wait_dropped[slot]|resume_dropped[slot]) else 'TIMEOUT / PREFIX FAILURE'
            draw.rectangle((0,0,576,40),fill=(20,50,30) if bool(succeeded[slot]) else (80,20,20));draw.text((8,12),status,font=font,fill='white')
            for _ in range(30):writer.append(np.asarray(frame))
        writer.close()
    if trace:save_json(dest/f'decisions_{seed}_{delay:g}_{mode}.json',trace)
    batch=[]
    for i in range(n):
        batch.append({'seed':seed,'slot':i,'delay_s':delay,'prefix_steps':prefix,'mode':mode,
          'eligible':bool(eligible[i]),'wait_drop':bool(wait_dropped[i]),'resume_drop':bool(resume_dropped[i]),
          'success':bool(succeeded[i]),'resume_steps_to_success':int(completion[i]),'mode_steps':counts[i].cpu().tolist(),'mode_switches':int(switches[i])})
    torch.save({'reset':initial,'request':request_initial,'eligible':eligible.cpu()},dest/f'initial_{seed}_{delay:g}_{mode}.pt')
    return batch,rollout

def summary(rows):
    eligible=[x for x in rows if x['eligible']]
    return {'trials':len(rows),'eligible':len(eligible),'prefix_failures':len(rows)-len(eligible),
       'successes':sum(x['success'] for x in eligible),'success_rate':sum(x['success'] for x in eligible)/max(1,len(eligible)),
       'wait_drops':sum(x['wait_drop'] for x in eligible),'resume_drops':sum(x['resume_drop'] for x in eligible),
       'mode_steps':np.sum([x['mode_steps'] for x in rows],axis=0).tolist()}

try:
    for iteration in range(0 if a.eval_only else a.iterations):
        delay=a.train_delays[iteration%len(a.train_delays)]
        batch,rollout=run_batch(a.seed+iteration,delay,'train',True)
        stats=ppo_update(model,optimizer,rollout)
        row={'iteration':iteration+1,'delay_s':delay,'elapsed_s':time.monotonic()-started,**summary(batch),**stats}
        history.append(row); print('TRAIN '+json.dumps(row),flush=True)
        save_json(dest/'training.json',history)
        save_json(dest/f'train_trials_{iteration+1:03d}.json',batch)
        torch.save({'model':model.state_dict(),'obs_dim':model.obs_dim,'iteration':iteration+1,
                    'actor_parameters':sum(x.numel() for x in model.actor.parameters()),'manifest':manifest},dest/'controller_latest.pt')
    if not a.skip_eval:
        for seed in a.eval_seeds:
            for delay in a.eval_delays:
                for mode in a.eval_modes:
                    batch,_=run_batch(seed,delay,mode)
                    records.extend(batch); save_json(dest/'eval_trials.json',records)
                    result={m:summary([r for r in records if r['mode']==m]) for m in a.eval_modes if any(r['mode']==m for r in records)}
                    save_json(dest/'eval_summary.json',result)
                    print('EVAL '+json.dumps({'seed':seed,'delay':delay,'mode':mode,**summary(batch)}),flush=True)
    save_json(dest/'complete.json',{'elapsed_s':time.monotonic()-started,'training_iterations':len(history),'eval_trials':len(records)})
finally:
    env.close(); app.close()
