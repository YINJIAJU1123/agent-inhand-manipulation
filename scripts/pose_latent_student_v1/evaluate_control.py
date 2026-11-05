"""Closed-loop wait/recovery evaluation with scripted, correct semantic delivery.

Runs the real Newton dynamics and RGB pose estimator. Semantic bindings are
oracle labels delivered after injected simulation-time delays. This isolates
control effects and is explicitly not an Astra/VLM main-comparison result.
"""
import argparse
import json
from pathlib import Path
import signal
import time

from viserdex.utils.runtime import add_runtime_args, parse_runtime_args

parser=argparse.ArgumentParser(description=__doc__)
from newton_runtime import NewtonRuntime,add_arguments,sha
add_arguments(parser)
parser.add_argument('--episodes',type=int,default=20)
parser.add_argument('--maintenance',choices=('continue','fixed_slow','learned','learned_wait_only'),required=True)
parser.add_argument('--actor_checkpoint',type=Path)
parser.add_argument('--execution_actor_checkpoint',type=Path)
parser.add_argument('--goal_vectors',type=Path)
parser.add_argument('--stop_checkpoint',type=Path)
parser.add_argument('--stop_calibration',type=Path)
parser.add_argument('--stop_mode',choices=('measured','corrected','learned','learned_and_measured'),default='corrected')
parser.add_argument('--completion_during_wait',action='store_true',
                    help='Check the existing local stop gate during pending requests after the first goal delivery.')
add_runtime_args(parser)
args=parse_runtime_args(parser)
if args.output.exists():raise FileExistsError(args.output)
if args.stop_calibration and not args.stop_checkpoint:raise ValueError('Calibration requires a completion checkpoint')
if args.stop_mode=='learned_and_measured' and not args.stop_checkpoint:raise ValueError('Combined gate requires an evidence checkpoint')
if not 0<args.episodes<=40:raise ValueError('Expected 1 to 40 episodes')
args.output.mkdir(parents=True)

import torch
from control_modules import features,CompletionGate,load_maintenance_actor
from goals import COLORS,SYMBOLS

def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def angle(q,target):
    q=torch.nn.functional.normalize(q.double(),dim=-1)
    target=torch.nn.functional.normalize(target.double(),dim=-1)
    return float(2*torch.acos((q*target).sum(-1).abs().clamp(0,1))[0])

def main():
    runtime=None
    rows=[]
    status='failed'
    stopped={'value':False}
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.update(value=True))
    start=time.monotonic()
    try:
        runtime=NewtonRuntime(args)
        gate=CompletionGate(args.stop_checkpoint,sha(args.checkpoint),runtime.vectors,runtime.device,args.stop_calibration) if args.stop_checkpoint else None
        actor=None
        execution_actor=None
        if args.maintenance in ('learned','learned_wait_only'):
            actor=load_maintenance_actor(args.actor_checkpoint,sha(args.checkpoint),runtime.vectors,runtime.device)
        if args.execution_actor_checkpoint:
            if args.maintenance!='learned':
                raise ValueError('An execution actor requires learned maintenance')
            execution_actor=load_maintenance_actor(args.execution_actor_checkpoint,sha(args.checkpoint),runtime.vectors,runtime.device)
        custom_vectors=None
        if args.goal_vectors:
            pack=torch.load(args.goal_vectors,map_location='cpu',weights_only=True)
            custom_vectors=pack['vectors'].to(runtime.device)
            if custom_vectors.shape!=(6,768) or not torch.isfinite(custom_vectors).all():
                raise ValueError('Expected six finite continuous VLM vectors')
        manifest=dict(runtime.metadata,maintenance=args.maintenance,episodes=args.episodes,seed=args.seed,
            completion_checkpoint_sha256=sha(args.stop_checkpoint) if gate else None,stop_mode=args.stop_mode if gate else 'measured',
            completion_during_wait=args.completion_during_wait,
            completion_kind=gate.kind if gate else None,
            stop_calibration_sha256=sha(args.stop_calibration) if args.stop_calibration else None,
            source_sha256_evaluator=sha(__file__),actor_sha256=sha(args.actor_checkpoint) if actor else None,
            execution_actor_sha256=sha(args.execution_actor_checkpoint) if execution_actor else None,
            goal_vectors_sha256=sha(args.goal_vectors) if args.goal_vectors else None,
            semantic_provider='scripted correct bindings; no actual model calls',
            delays='1..6 seconds of continuing simulation, deterministic per case/request',
            measured_stop_threshold_rad=gate.measured_threshold if gate else .16,
            stop_policy=('correct binding + learned evidence AND calibrated measured pose at 3Hz' if args.stop_mode=='learned_and_measured' else 'correct binding + calibrated '+gate.kind+' at 3Hz' if gate and args.stop_mode!='measured' else 'correct binding + calibrated measured full quaternion at 3Hz; no hold requirement'),
            scoring='independent true full quaternion <=0.16rad, autonomous local stop, no preceding drop',
            evaluation_scope='development control isolation, not formal autonomous semantic success',
            inference_pacing='injected delay measured in simulation steps; no real wall-time latency claim',
            formal_trials_used=False)
        save(args.output/'manifest.json',manifest)
        with torch.inference_mode(),(args.output/'steps.jsonl').open('w',buffering=1) as step_log:
            for ep in range(args.episodes):
                if stopped['value']:break
                trial_seed=args.seed*1000+ep
                goal=(ep+(0 if args.seed%2 else 2))%6
                initial=runtime.reset(trial_seed)
                runtime.start_video(args.output/f'episode_{ep:03d}',
                    f'Scripted goal delivery | Revo3 latent student | target letter {SYMBOLS[goal]} upward')
                target=runtime.rotations[goal:goal+1]
                known=False
                pending=True
                requests=1
                request_step=0
                due=30*(1+ep%6)
                next_request=None
                runtime.begin_request()
                error=angle(runtime.raw.scene['object'].data.root_quat_w.torch,target)
                minimum=error
                reached=error<=.16
                reason=None
                drop_phase=None
                stop_measurement=None
                deliveries=[]
                while reason is None:
                    if stopped['value'] or not runtime.app.is_running():
                        reason='interrupted'
                        break
                    if pending and runtime.steps>=due:
                        runtime.set_skill(goal,None if custom_vectors is None else custom_vectors[goal])
                        known=True
                        pending=False
                        deliveries.append(dict(step=runtime.steps,request=requests,goal=goal))
                        next_request=runtime.steps+90 if requests<6 else None
                    if not pending and next_request is not None and runtime.steps>=next_request:
                        pending=True
                        requests+=1
                        request_step=runtime.steps
                        due=runtime.steps+30*(1+(ep+requests-1)%6)
                        runtime.begin_request()
                    sensor=runtime.observe()
                    values,hidden=runtime.propose(pending,known,(runtime.steps-request_step)/30 if pending else 0.)
                    gate_result=None
                    stop_eligible=known and (args.completion_during_wait or not pending) and runtime.steps%10==0
                    runtime.record_video(sensor,mode='execution' if known else 'startup',pending=pending,known=known,
                        active_color=COLORS[runtime.active_goal],bound_color=COLORS[goal] if known else None,
                        calls=requests,measured_error_rad=float(sensor['estimated_errors'][runtime.active_goal]))
                    if stop_eligible:
                        measured=float(sensor['estimated_errors'][goal])
                        if gate is not None:
                            gate_result=gate.estimate(values,runtime.latent,sensor['estimated_errors'][goal:goal+1].float())
                        accept=bool(gate_result['accept'][0]) if gate_result and args.stop_mode!='measured' else measured<=(gate.measured_threshold if gate else .16)
                        if args.stop_mode=='learned_and_measured':
                            accept=accept and measured<=gate.measured_threshold
                        if accept:
                            stop_measurement=dict(raw_error_rad=measured,
                                pending=pending,decision_step=runtime.steps,
                                corrected_error_rad=float(gate_result['estimated_error'][0]) if gate_result and gate.kind=='pose_correction' else None,
                                stop_bound_rad=float(gate_result['stop_bound'][0]) if gate_result and gate_result['stop_bound'] is not None else None,
                                evidence_probability=float(gate_result['probability'][0]) if gate_result and gate_result['probability'] is not None else None)
                            error=angle(runtime.raw.scene['object'].data.root_quat_w.torch,target)
                            reason='success' if error<=.16 else 'false_stop'
                            break
                    current=values['proprio'][:,:21].clamp(-1,1)
                    proposal=values['proposal'].clamp(-1,1)
                    action=proposal
                    speed=1.
                    if args.maintenance=='fixed_slow' and pending:
                        action=current+.15*(proposal-current)
                        speed=.15
                    elif actor is not None and (pending or args.maintenance=='learned'):
                        controller=actor if pending or execution_actor is None else execution_actor
                        result=controller(features(values),runtime.latent,proposal,current,values['pending'])
                        action=result['action']
                        speed=float(result['speed'][0])
                    if runtime.steps<5:
                        trace=dict(episode=ep,step=runtime.steps,proprio=values['proprio'][0].cpu().tolist(),
                            pose=values['measured_pose'][0].cpu().tolist(),proposal=proposal[0].cpu().tolist(),
                            action=action[0].cpu().tolist(),latent=runtime.latent[0].cpu().tolist(),
                            object_before=runtime.raw.scene['object'].data.root_state_w.torch[0].cpu().tolist(),
                            joints_before=runtime.raw.scene['robot'].data.joint_pos.torch[0].cpu().tolist())
                    terminal=runtime.step(action,hidden)
                    if runtime.steps<=5:
                        trace['quaternion_after']=terminal['quaternion'][0].cpu().tolist()
                        trace['position_after']=terminal['position'][0].cpu().tolist()
                        with (args.output/'initial_steps.jsonl').open('a') as trace_file:
                            trace_file.write(json.dumps(trace)+'\n')
                    error=angle(terminal['quaternion'],target)
                    minimum=min(minimum,error)
                    reached|=error<=.16
                    dropped=bool(terminal['drop'][0])
                    if dropped:
                        reason='drop'
                        drop_phase='startup_wait' if not known else 'wait' if pending else 'execution'
                    elif bool(terminal['done'][0]):
                        reason='horizon' if bool(terminal['timeout'][0]) else 'unexpected_termination'
                    elif runtime.steps>=3600:
                        reason='horizon'
                    step_log.write(json.dumps(dict(episode=ep,step=runtime.steps,pending=pending,known=known,
                        observation_step=runtime.steps-1,stop_gate_eligible=stop_eligible,
                        active_goal=runtime.active_goal,true_error_rad=error,
                        measured_error_rad=float(sensor['estimated_errors'][goal]),
                        corrected_error_rad=float(gate_result['estimated_error'][0]) if gate_result and gate.kind=='pose_correction' else None,
                        stop_bound_rad=float(gate_result['stop_bound'][0]) if gate_result and gate_result['stop_bound'] is not None else None,
                        evidence_probability=float(gate_result['probability'][0]) if gate_result and gate_result['probability'] is not None else None,
                        fit_error_m=float(sensor['fit_error_m'][0]),speed=speed,drop=dropped,
                        object_quaternion=terminal['quaternion'][0].cpu().tolist(),reason=reason))+'\n')
                    if runtime.steps%300==0:
                        save(args.output/'progress.json',dict(status='running',completed_episodes=len(rows),
                            current_episode=ep,step=runtime.steps,elapsed_seconds=time.monotonic()-start))
                runtime.finish_video(reason)
                row=dict(episode=ep,trial_seed=trial_seed,goal_index=goal,color=COLORS[goal],symbol=SYMBOLS[goal],
                    initial_state=initial,steps=runtime.steps,seconds=runtime.steps/30,reason=reason,
                    success=reason=='success',drop_phase=drop_phase,stop_measurement=stop_measurement,minimum_error_rad=minimum,
                    reached=bool(reached),final_error_rad=error,deliveries=deliveries,
                    simulated_requests=requests,actual_model_calls=0)
                rows.append(row)
                with (args.output/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                progress=dict(status='running',completed_episodes=len(rows),requested_episodes=args.episodes,
                    success=sum(r['success'] for r in rows),drops=sum(r['reason']=='drop' for r in rows),
                    elapsed_seconds=time.monotonic()-start)
                save(args.output/'progress.json',progress)
                print('[WAIT_CONTROL] '+json.dumps(progress),flush=True)
                if reason=='interrupted':break
        status='completed' if len(rows)==args.episodes and all(r['reason']!='interrupted' for r in rows) else 'interrupted'
    except Exception as exc:
        save(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        result=dict(status=status,maintenance=args.maintenance,requested_episodes=args.episodes,
            recorded_episodes=len(rows),success=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/len(rows) if rows else None,
            reasons={key:sum(r['reason']==key for r in rows) for key in sorted({r['reason'] for r in rows})},
            drop_phases={key:sum(r['drop_phase']==key for r in rows) for key in ('startup_wait','wait','execution')},
            real_model_calls=0,formal_agent_result=False,elapsed_seconds=time.monotonic()-start)
        save(args.output/'result.json',result)
        save(args.output/'progress.json',result)
        if runtime is not None:runtime.close()

if __name__=='__main__':main()
