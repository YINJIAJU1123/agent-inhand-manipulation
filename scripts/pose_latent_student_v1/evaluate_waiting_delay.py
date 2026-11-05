"""Live semantic agents and fixed-plan control replay on the Newton backend.

Live requests and recorded waits run while physics and local control continue.
Replay performs no model inference and is reported as a mechanism diagnostic.
Stopping is independently scored from simulator truth. Reserved formal seeds
require a frozen release; formal outcomes are never used for checkpoint selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

from viserdex.utils.runtime import add_runtime_args,parse_runtime_args
from newton_runtime import NewtonRuntime,add_arguments,sha

parser=argparse.ArgumentParser(description=__doc__)
add_arguments(parser)
parser.add_argument('--project_root',type=Path,required=True)
parser.add_argument('--episodes',type=int,default=20)
parser.add_argument('--start_episode',type=int,default=0,help='First scheduled index when continuing unvisited trials in a new output directory')
parser.add_argument('--provider',choices=('astra','vlm','rule','replay'),required=True)
parser.add_argument('--agent_protocol',choices=('feedback','initial_plan'),default='feedback')
parser.add_argument('--visual_intervention',choices=('native','identity','permuted','letter_masked'),default='native')
parser.add_argument('--intervention_geometry',type=Path)
parser.add_argument('--intervention_textures',type=Path)
parser.add_argument('--replay_plan',type=Path,help='Frozen public tool calls and latencies for a model-free controller diagnostic.')
parser.add_argument('--maintenance',choices=('continue','fixed_slow','learned','learned_wait_only','learned_execution_only','continue_wait_learned_execution'),default='fixed_slow')
parser.add_argument('--extra-delay-s',type=float,default=0.,help='One-time added simulated delay on the first request made after a skill is bound.')
parser.add_argument('--actor_checkpoint',type=Path)
parser.add_argument('--execution_actor_checkpoint',type=Path)
parser.add_argument('--controller_goal_intervention',choices=('none','cyclic_shift'),default='none',
    help='Shift only actor goal by one skill, preserving student/task goal. Diagnostic release required.')
parser.add_argument('--formal_release',type=Path,help='Frozen release required before using reserved evaluation seeds.')
parser.add_argument('--formal_condition',help='Condition name in the frozen release.')
parser.add_argument('--prepare_only',action='store_true',help='Initialize the runtime, export its contract and exit before scheduled trials or model inference.')
parser.add_argument('--stop_checkpoint',type=Path)
parser.add_argument('--stop_calibration',type=Path)
parser.add_argument('--stop_mode',choices=('measured','learned'),default='learned')
parser.add_argument('--completion_during_wait',action='store_true',
                    help='Check the existing local stop gate while a fresh request is pending, after binding the goal.')
parser.add_argument('--require-injected-response-before-stop',action='store_true',
                    help='For fixed-plan waiting studies, defer autonomous stop until the registered post-binding reply is delivered.')
parser.add_argument('--vlm_url',default='http://127.0.0.1:18782')
parser.add_argument('--rule_url',default='http://127.0.0.1:18783')
add_runtime_args(parser)
args=parse_runtime_args(parser)
if args.extra_delay_s not in (0.,6.,18.):
    raise ValueError('Registered added delays are 0, 6, or 18 seconds')
from formal_release import load_release,verify_runtime,save_preparation
formal_release=load_release(args,Path(__file__).parent)
if (formal_release is not None and args.controller_goal_intervention!='none'
        and formal_release.get('format')!='cube_controller_goal_ablation_v1'):
    raise ValueError('Goal intervention requires a separate registered diagnostic release')
if args.controller_goal_intervention!='none' and args.maintenance not in ('learned','learned_wait_only','learned_execution_only'):
    raise ValueError('Goal intervention requires a learned controller')
if args.agent_protocol=='initial_plan':
    if args.provider not in ('astra','replay'):raise ValueError('Initial-plan protocol requires Astra or explicit model-free replay')
    if args.provider=='replay' and formal_release is not None:raise ValueError('Initial-plan replay cannot produce formal Astra outcomes')
    if formal_release is not None and formal_release.get('format') not in ('cube_agent_feedback_ablation_v1','cube_waiting_feedback_ablation_v3'):
        raise ValueError('Initial plan requires a separately registered feedback study')
if args.visual_intervention!='native':
    if args.intervention_geometry is None or args.intervention_textures is None:raise ValueError('Visual intervention requires frozen geometry and textures')
    if formal_release is not None and formal_release.get('format')!='cube_visual_semantic_intervention_v1':raise ValueError('Visual intervention requires its own formal release')
elif args.intervention_geometry is not None or args.intervention_textures is not None:raise ValueError('Native images must not receive intervention resources')
if args.output.exists():raise FileExistsError(args.output)
if args.maintenance in ('learned_execution_only','continue_wait_learned_execution') and not args.execution_actor_checkpoint:
    raise ValueError('Learned execution with fixed-slow waiting requires an execution checkpoint')
if args.stop_calibration and not args.stop_checkpoint:raise ValueError('Calibration requires a completion checkpoint')
if (args.provider=='replay') != (args.replay_plan is not None):
    raise ValueError('Replay provider requires its frozen plan, and live providers cannot use a replay plan')
if args.require_injected_response_before_stop and args.provider!='replay':
    raise ValueError('The injected-response stop guard is only valid for fixed-plan replay')
if not 0<args.episodes<=40 or not 0<=args.start_episode<40 or args.start_episode+args.episodes>40:
    raise ValueError('Expected a contiguous subset of at most 40 scheduled episode indices')
args.output.mkdir(parents=True)

import torch
from PIL import Image
from control_modules import features,CompletionGate,load_maintenance_actor,controller_goal_input
from goals import COLORS,SYMBOLS
from semantic import validate_action,SYSTEM
from initial_plan import InitialPlanExecutor,validate_initial_plan,INITIAL_PLAN_SYSTEM
ROOT=args.project_root.resolve()
sys.path[:0]=[str(ROOT/'scripts/astra_transport_v2'),str(ROOT/'scripts/agentic')]

def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def angle(q,target):
    q=torch.nn.functional.normalize(q.double(),dim=-1)
    target=torch.nn.functional.normalize(target.double(),dim=-1)
    return float(2*torch.acos((q*target).sum(-1).abs().clamp(0,1))[0])

def request(client,provider,instruction,rgb,feedback,history,replayed_call=None):
    if provider=='replay':
        return dict(result=dict(tool=replayed_call['tool'],arguments=replayed_call['arguments'],
                                response_id='recorded-plan',call_id='recorded-call'),
                    error=None,elapsed_s=replayed_call['elapsed_s'])
    started=time.monotonic()
    try:
        if provider=='astra':
            result=client.ask(instruction,rgb,feedback,*history)
        else:
            result=client.predict(rgb,instruction)
        return dict(result=result,error=None,elapsed_s=time.monotonic()-started)
    except Exception as exc:
        return dict(result=None,error=type(exc).__name__+': '+str(exc),elapsed_s=time.monotonic()-started)

def main():
    runtime=None
    rows=[]
    status='failed'
    stopped={'value':False}
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.update(value=True))
    start=time.monotonic()
    upstream={}
    replay=None
    predictor=None
    if args.provider=='replay':
        replay=json.loads(args.replay_plan.read_text())
        expected_format='pose_latent_initial_plan_replay_v1' if args.agent_protocol=='initial_plan' else 'pose_latent_public_plan_replay_v1'
        if replay['format']!=expected_format:
            raise ValueError('Unknown replay plan format')
        for trial in replay['trials'].values():
            if not 1<=len(trial['calls'])<=6:raise ValueError('Invalid replay call count')
            if args.agent_protocol=='initial_plan' and len(trial['calls'])!=1:raise ValueError('Initial plan replay requires exactly one delivery')
            for call in trial['calls']:
                (validate_initial_plan if args.agent_protocol=='initial_plan' else validate_action)(call['tool'],call['arguments'])
                if not 0<call['elapsed_s']<=120:raise ValueError('Invalid replay latency')
        upstream=dict(backend='fixed_public_plan_replay',plan_sha256=sha(args.replay_plan),
            actual_model_calls=0,closed_loop_high_level_feedback=False,
            replay_origin=replay.get('origin','recorded_public_calls'))
    elif args.provider=='astra':
        from bridge import BridgeClient,bridge_health
        bridge_health()
        upstream=dict(model='gpt-6-astra',transport='identity-pinned local OAuth bridge',retries=0)
    elif args.provider=='rule':
        from semantic import RSLCubeRulePredictor
        predictor=RSLCubeRulePredictor(args.rule_url)
        upstream=dict(predictor.metadata,backend='frozen_base_vlm_text_rules')
    else:
        from latent_service_v7 import RemotePredictor
        predictor=RemotePredictor(args.vlm_url)
        upstream=predictor.metadata
        if upstream.get('backend')!='rslcube' or upstream.get('frame_size')!=120:
            raise ValueError('Unexpected VLM service contract')
    try:
        runtime=NewtonRuntime(args)
        upper_intervention=None
        if args.visual_intervention!='native':
            from visual_intervention import UpperImageIntervention
            upper_intervention=UpperImageIntervention(args.visual_intervention,args.intervention_geometry,args.intervention_textures,ROOT,runtime)
        gate=CompletionGate(args.stop_checkpoint,sha(args.checkpoint),runtime.vectors,runtime.device,args.stop_calibration) if args.stop_checkpoint else None
        actor=None
        execution_actor=None
        # Load supplied checkpoints in the same order for every factorial cell.
        # Action use is decided by maintenance mode, not by whether a model is loaded.
        if args.actor_checkpoint:
            actor=load_maintenance_actor(args.actor_checkpoint,sha(args.checkpoint),runtime.vectors,runtime.device)
        elif args.maintenance in ('learned','learned_wait_only'):
            raise ValueError('Learned waiting requires an actor checkpoint')
        if args.execution_actor_checkpoint:
            execution_actor=load_maintenance_actor(args.execution_actor_checkpoint,sha(args.checkpoint),runtime.vectors,runtime.device)
        if args.controller_goal_intervention!='none' and any(
                item is not None and item.goal_input!='active' for item in (actor,execution_actor)):
            raise ValueError('Shuffle requires active-goal actors')
        manifest=dict(runtime.metadata,provider=args.provider,upstream=upstream,maintenance=args.maintenance,
            controller_modes=dict(
                waiting=('learned' if args.maintenance in ('learned','learned_wait_only')
                         else 'fixed_slow' if args.maintenance in ('fixed_slow','learned_execution_only') else 'student'),
                execution='learned' if args.maintenance in ('learned','learned_wait_only','learned_execution_only','continue_wait_learned_execution') else 'student'),
            seed=args.seed,episodes=args.episodes,start_episode=args.start_episode,evaluator_sha256=sha(__file__),
            actor_sha256=sha(args.actor_checkpoint) if actor else None,
            execution_actor_sha256=sha(args.execution_actor_checkpoint) if execution_actor else None,
            completion_sha256=sha(args.stop_checkpoint) if gate else None,
            completion_during_wait=args.completion_during_wait,
            completion_kind=gate.kind if gate else None,
            stop_mode=args.stop_mode if gate else 'measured',
            measured_stop_threshold_rad=gate.measured_threshold if gate else .16,
            stop_calibration_sha256=sha(args.stop_calibration) if args.stop_calibration else None,
            stopping=('calibrated '+gate.kind+' with requested binding at 3Hz' if gate and args.stop_mode=='learned' else 'measured-pose gate with model-declared requested binding at 3Hz, no hold'),
            physics='continuous during every request; 30Hz wall pacing while pending; registered extra delay added to the first post-binding request; no real-time throughput claim',
            request_budget=1 if args.agent_protocol=='initial_plan' else 6,service_exclusion='service error or measured request latency >120s; report raw and eligible',
            inspection='Astra model-selected; fine-tuned VLM uses shared deterministic unseen-view schedule when unreadable',
            fine_tuned_goal='continuous returned vector enters student unchanged; nearest skill only for public verification and logs',
            scope=('frozen formal semantic-agent comparison' if formal_release is not None
                else 'real model development rollout, not frozen formal comparison'),
            formal_trials_used=formal_release is not None,
            formal_condition=args.formal_condition,
            formal_release_sha256=sha(args.formal_release) if formal_release is not None else None)
        manifest['waiting_delay_intervention']=dict(extra_delay_s=args.extra_delay_s,
            policy_intervention='Activated only after the first post-binding request is submitted; all arms share fixed 0.15 startup waiting and the same frozen execution actor before that point.',
            frozen_execution_actor_sha256=sha(args.execution_actor_checkpoint) if execution_actor else None,
            trigger='first request submitted after a semantic skill is already bound',
            applied_to='response delivery after the live Astra request returns',
            limitation=('Fixed recorded Astra replies are replayed identically across matched arms.' if replay is not None
                else 'Each policy receives a live request from its own resulting RGB state; generated replies may differ across policy arms.'))
        manifest['extra_delay_s']=args.extra_delay_s
        if replay is not None:
            manifest.update(scope='Model-free fixed-plan controller mechanism diagnostic; not a live Astra result',
                physics='continuous during recorded waits; delivery after ceil(recorded latency*30) steps',
                inspection='fixed recorded tool calls; no adaptation to replay images',
                service_exclusion='none; source schedule contains no service errors or over-120s latencies')
            if replay.get('origin')=='synthetic_runtime_fixture':
                manifest.update(scope='Synthetic model-free executor runtime check, not a model performance result',
                    physics='continuous during specified synthetic waits',actual_model_calls=0)
        if args.agent_protocol=='initial_plan':
            manifest['inspection']='One initial RGB observation and bounded complete plan; later inspect actions provide no new high-level observation'
        manifest['agent_protocol']=args.agent_protocol
        manifest['waiting_residual_mode']=actor.waiting_residual_mode if actor is not None and args.maintenance in ('learned','learned_wait_only') else 'none'
        manifest['visual_intervention']=upper_intervention.metadata if upper_intervention is not None else {'mode':'native'}
        manifest['controller_goal_intervention']=dict(mode=args.controller_goal_intervention,
            mapping=[1,2,3,4,5,0] if args.controller_goal_intervention=='cyclic_shift' else None,
            waiting_goal_input=actor.goal_input if actor is not None else None,
            execution_goal_input=execution_actor.goal_input if execution_actor is not None else None,
            scope='actor explicit goal only; student and independent score unchanged')
        if formal_release is not None:
            verify_runtime(formal_release,args.formal_condition,manifest)
            save(args.output/'formal_release.json',formal_release)
        if args.prepare_only:
            manifest.update(scope='Runtime preparation only; no scheduled trials or model inference',preparation_only=True)
        save(args.output/'manifest.json',manifest)
        if args.prepare_only:
            save_preparation(args.output,Path(__file__).parent,'agent',ROOT)
            status='prepared'
            return
        with ThreadPoolExecutor(max_workers=1) as executor,torch.inference_mode(),(args.output/'steps.jsonl').open('w',buffering=1) as step_log:
            for ep in range(args.start_episode,args.start_episode+args.episodes):
                if stopped['value']:break
                trial_seed=args.seed*1000+ep
                goal=(ep+(0 if args.seed%2 else 2))%6
                symbol=upper_intervention.symbol(goal) if upper_intervention is not None else SYMBOLS[goal]
                replayed_trial=replay['trials'][str(trial_seed)] if replay is not None else None
                if replayed_trial is not None and replayed_trial['requested_symbol']!=symbol:
                    raise ValueError('Replay instruction does not match scheduled target')
                call_limit=len(replayed_trial['calls']) if replayed_trial is not None else (1 if args.agent_protocol=='initial_plan' else 6)
                initial_plan=InitialPlanExecutor() if args.agent_protocol=='initial_plan' else None
                instruction=f'Reorient the cube so that the face printed with the capital letter {symbol} points upward.'
                initial=runtime.reset(trial_seed)
                target=runtime.rotations[goal:goal+1]
                client=BridgeClient() if args.provider=='astra' else predictor
                history=(None,None)
                known=False
                binding_symbol=None
                binding_goal=None
                mode='startup'
                due=0
                calls=[]
                submitted=0
                future=None
                ready=None
                request_step=0
                request_wall=0.
                injected_call=None
                intervention_started=False
                intervention_anchor=None
                intervention_position=None
                intervention_quaternion=None
                intervention_max_rotation_drift=0.
                intervention_max_translation_drift=0.
                reason=None
                drop_phase=None
                injected_wait_drop=False
                first_skill_delivery_step=None
                stop_measurement=None
                excluded=None
                error=angle(runtime.raw.scene['object'].data.root_quat_w.torch,target)
                minimum=error
                reached=error<=.16
                epdir=args.output/f'episode_{ep:03d}'
                epdir.mkdir()
                runtime.start_video(epdir,f'{args.provider} | Revo3 latent student | target letter {symbol} upward')
                while reason is None:
                    if stopped['value'] or not runtime.app.is_running():
                        reason='interrupted';break
                    sensor=runtime.observe()
                    if future is None and ready is None and runtime.steps>=due and submitted<call_limit:
                        runtime.begin_request()
                        submitted+=1
                        request_step=runtime.steps
                        request_wall=time.monotonic()
                        rgb=sensor['rgb'].copy()
                        if upper_intervention is not None:
                            constructed_at=time.monotonic()
                            rgb,image_audit=upper_intervention.image(runtime,rgb)
                            image_audit.update(call=submitted,control_step=runtime.steps,construction_wall_s=time.monotonic()-constructed_at)
                            with (epdir/'visual_intervention.jsonl').open('a') as stream:stream.write(json.dumps(image_audit)+'\n')
                        Image.fromarray(rgb).save(epdir/f'request_{submitted:02d}.png')
                        feedback=dict(control_step=runtime.steps,elapsed_simulation_s=runtime.steps/30,
                            previous_state=mode,current_color_skill=COLORS[runtime.active_goal],
                            bound_symbol=binding_symbol,bound_color=COLORS[binding_goal] if binding_goal is not None else None,
                            calls_remaining=call_limit-submitted,
                            sensor_estimated_error_rad_by_skill={c:float(sensor['estimated_errors'][i]) for i,c in enumerate(COLORS)},
                            pose_fit_error_m=float(sensor['fit_error_m'][0]),
                            caution='No ground truth or success sensor. Errors are estimates from current RGB pose.')
                        replayed_call=replayed_trial['calls'][submitted-1] if replayed_trial is not None else None
                        future=executor.submit(request,client,args.provider,instruction,rgb,feedback,history,replayed_call)
                        if binding_goal is not None and injected_call is None:
                            injected_call=submitted
                            intervention_started=True
                            anchor_position=runtime.raw.scene['object'].data.root_pos_w.torch[0].detach().double().cpu()
                            anchor_quaternion=runtime.raw.scene['object'].data.root_quat_w.torch[0].detach().double().cpu()
                            robot=runtime.raw.scene['robot'].data
                            state_snapshot=dict(
                                object_position=anchor_position.tolist(),
                                object_quaternion=anchor_quaternion.tolist(),
                                object_linear_velocity=runtime.raw.scene['object'].data.root_lin_vel_w.torch[0].detach().double().cpu().tolist(),
                                object_angular_velocity=runtime.raw.scene['object'].data.root_ang_vel_w.torch[0].detach().double().cpu().tolist(),
                                robot_joint_position=robot.joint_pos[0].detach().double().cpu().tolist(),
                                robot_joint_velocity=robot.joint_vel[0].detach().double().cpu().tolist())
                            intervention_position=anchor_position.numpy().tolist()
                            intervention_quaternion=anchor_quaternion.numpy().tolist()
                            intervention_anchor=dict(step=runtime.steps,
                                state_sha256=hashlib.sha256(json.dumps(state_snapshot,sort_keys=True).encode()).hexdigest(),
                                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                feedback_sha256=hashlib.sha256(json.dumps(feedback,sort_keys=True).encode()).hexdigest(),
                                state_snapshot=state_snapshot,
                                position_w=intervention_position,
                                quaternion_w=intervention_quaternion,
                                bound_goal=binding_goal,bound_symbol=binding_symbol)
                    if future is not None and future.done():
                        ready=future.result();future=None
                        ready.update(call=submitted,request_step=request_step)
                        is_injected=submitted==injected_call
                        ready['injected_extra_delay_s']=args.extra_delay_s if is_injected else 0.
                        ready['delivery_step']=request_step+math.ceil(
                            (ready['elapsed_s']+ready['injected_extra_delay_s'])*30)
                        if ready['error'] is not None:excluded='service_error'
                        elif ready['elapsed_s']>120:excluded='request_latency_over_120s'
                        if excluded:
                            calls.append(ready);ready=None;reason='service_excluded';break
                    if ready is not None and runtime.steps>=ready['delivery_step']:
                        if submitted==injected_call and intervention_position is not None:
                            delivered_position=runtime.raw.scene['object'].data.root_pos_w.torch[0].detach().double().cpu()
                            delivered_quaternion=torch.nn.functional.normalize(
                                runtime.raw.scene['object'].data.root_quat_w.torch[0].detach().double().cpu(),dim=-1)
                            anchor_q=torch.nn.functional.normalize(torch.tensor(intervention_quaternion,dtype=torch.float64),dim=-1)
                            rotation_delta=2*torch.acos(torch.dot(delivered_quaternion,anchor_q).abs().clamp(0,1))
                            translation_delta=torch.linalg.vector_norm(delivered_position-
                                torch.tensor(intervention_position,dtype=torch.float64))
                            intervention_max_rotation_drift=max(intervention_max_rotation_drift,float(rotation_delta))
                            intervention_max_translation_drift=max(intervention_max_translation_drift,float(translation_delta))
                        result=ready['result']
                        if args.provider=='astra':
                            expected_prompt=INITIAL_PLAN_SYSTEM if args.agent_protocol=='initial_plan' else SYSTEM
                            if result.get('system_prompt')!=expected_prompt:raise ValueError('Astra bridge protocol differs from evaluator')
                        ready['applied_step']=runtime.steps
                        calls.append(ready)
                        ready=None
                        if initial_plan is not None:
                            initial_plan.accept(result['tool'],result['arguments'],runtime.steps)
                        elif args.provider in ('astra','replay'):
                            tool=result['tool'];arguments=validate_action(tool,result['arguments'])
                            history=(result['response_id'],result['call_id'])
                            if tool=='abstain':
                                reason='abstain';break
                            selected=COLORS.index(arguments['color'])
                            runtime.set_skill(selected)
                            mode=tool
                            if tool!='inspect_face':
                                binding_symbol=arguments['goal_symbol'];binding_goal=selected;known=True
                            else:
                                binding_symbol=None;binding_goal=None;known=False
                            due=runtime.steps+arguments.get('steps',30)
                            # A premature model stop is rejected by the common
                            # measured-sensor gate; its target remains active.
                        else:
                            if bool(result['accepted']):
                                selected=int(result['skill'])
                                if args.provider=='vlm':
                                    vector=torch.tensor(result['goal_embedding'],device=runtime.device)
                                    if vector.shape!=(768,) or not torch.isfinite(vector).all():
                                        raise ValueError('Invalid VLM goal vector')
                                    runtime.set_skill(selected,vector)
                                else:
                                    runtime.set_skill(selected)
                                binding_symbol=symbol;binding_goal=selected;known=True;mode='reorient_target'
                            elif known:
                                mode='verify_target'
                            else:
                                selected=(1,3,0,2,4,5)[(submitted-1)%6]
                                runtime.set_skill(selected)
                                mode='inspect_face'
                            due=runtime.steps+90
                    if initial_plan is not None:
                        planned_action=initial_plan.pop_due(runtime.steps)
                        if planned_action is not None:
                            tool=planned_action['tool'];arguments=planned_action['arguments']
                            with (epdir/'plan_execution.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(step=runtime.steps,index=initial_plan.dispatched,
                                    tool=tool,arguments=arguments))+'\n')
                            if tool=='abstain':reason='abstain';break
                            selected=COLORS.index(arguments['color'])
                            runtime.set_skill(selected);mode=tool
                            if tool=='inspect_face':
                                binding_symbol=None;binding_goal=None;known=False
                            else:
                                binding_symbol=arguments['goal_symbol'];binding_goal=selected;known=True
                            due=initial_plan.next_step
                    if first_skill_delivery_step is None and mode!='startup':
                        first_skill_delivery_step=runtime.steps
                    pending=future is not None or ready is not None
                    injected_pending=(pending and injected_call is not None and submitted==injected_call)
                    values,hidden=runtime.propose(pending,known,(runtime.steps-request_step)/30 if pending else 0.)
                    stop_eligible=(known and mode!='inspect_face' and binding_symbol==symbol
                        and (args.completion_during_wait or not pending)
                        and (not args.require_injected_response_before_stop or injected_call is not None)
                        and runtime.steps%10==0)
                    runtime.record_video(sensor,mode=mode,pending=pending,known=known,
                        active_color=COLORS[runtime.active_goal],bound_color=COLORS[binding_goal] if known else None,
                        calls=submitted,measured_error_rad=float(sensor['estimated_errors'][runtime.active_goal]))
                    if stop_eligible:
                        measured_angle=sensor['estimated_errors'][binding_goal:binding_goal+1].float()
                        gate_result=gate.estimate(values,runtime.latent,measured_angle) if gate else None
                        accept=(bool(gate_result['accept'][0]) if gate_result and args.stop_mode=='learned'
                            else float(measured_angle[0])<=(gate.measured_threshold if gate else .16))
                        if accept:
                            stop_measurement=dict(raw_error_rad=float(measured_angle[0]),
                                pending=pending,decision_step=runtime.steps,
                                corrected_error_rad=float(gate_result['estimated_error'][0]) if gate_result and gate.kind=='pose_correction' else None,
                                stop_bound_rad=float(gate_result['stop_bound'][0]) if gate_result and gate_result['stop_bound'] is not None else None,
                                evidence_probability=float(gate_result['probability'][0]) if gate_result and gate_result['probability'] is not None else None)
                            error=angle(runtime.raw.scene['object'].data.root_quat_w.torch,target)
                            correct_binding=binding_goal==goal
                            reason='success' if correct_binding and error<=.16 else 'false_stop'
                            break
                    current=values['proprio'][:,:21].clamp(-1,1)
                    proposal=values['proposal'].clamp(-1,1)
                    action=proposal
                    speed=1.
                    controller_goal=controller_goal_input(runtime.latent,runtime.vectors,
                        runtime.active_goal,args.controller_goal_intervention)
                    controller=None
                    if pending and not intervention_started:
                        action=current+.15*(proposal-current);speed=.15
                    elif args.maintenance in ('fixed_slow','learned_execution_only') and pending:
                        action=current+.15*(proposal-current);speed=.15
                    elif actor is not None and args.maintenance in ('learned','learned_wait_only') and (pending or args.maintenance=='learned'):
                        controller=actor if pending or execution_actor is None else execution_actor
                        result=controller(features(values),controller_goal,proposal,current,values['pending'])
                        action=result['action'];speed=float(result['speed'][0])
                    elif execution_actor is not None and (args.maintenance=='learned_execution_only'
                            or args.maintenance=='learned_wait_only'
                            or (args.maintenance=='continue_wait_learned_execution' and not pending)):
                        controller=execution_actor
                        result=controller(features(values),controller_goal,proposal,current,values['pending'])
                        action=result['action'];speed=float(result['speed'][0])
                    terminal=runtime.step(action,hidden)
                    error=angle(terminal['quaternion'],target)
                    if injected_pending and intervention_position is not None:
                        current_position=terminal['position'][0].detach().double().cpu()
                        current_quaternion=torch.nn.functional.normalize(
                            terminal['quaternion'][0].detach().double().cpu(),dim=-1)
                        anchor_q=torch.tensor(intervention_quaternion,dtype=torch.float64)
                        rotation_delta=2*torch.acos(torch.dot(current_quaternion,
                            torch.nn.functional.normalize(anchor_q,dim=-1)).abs().clamp(0,1))
                        translation_delta=torch.linalg.vector_norm(current_position-
                            torch.tensor(intervention_position,dtype=torch.float64))
                        intervention_max_rotation_drift=max(intervention_max_rotation_drift,float(rotation_delta))
                        intervention_max_translation_drift=max(intervention_max_translation_drift,float(translation_delta))
                    minimum=min(minimum,error);reached|=error<=.16
                    dropped=bool(terminal['drop'][0])
                    if dropped:
                        reason='drop'
                        drop_phase='startup_wait' if first_skill_delivery_step is None else 'wait' if pending else 'execution'
                        injected_wait_drop=bool(drop_phase=='wait' and injected_pending)
                    elif bool(terminal['done'][0]) or runtime.steps>=3600:
                        reason='horizon'
                    step_log.write(json.dumps(dict(episode=ep,step=runtime.steps,pending=pending,known=known,
                        observation_step=runtime.steps-1,stop_gate_eligible=stop_eligible,
                        mode=mode,active_goal=runtime.active_goal,submitted=submitted,
                        controller_goal_input=(controller.goal_input if controller is not None else None),
                        controller_goal_index=(None if controller is None or controller.goal_input=='center'
                            else (runtime.active_goal+1)%6 if args.controller_goal_intervention=='cyclic_shift'
                            else runtime.active_goal),
                        measured_active_error_rad=float(sensor['estimated_errors'][runtime.active_goal]),
                        measured_bound_error_rad=float(sensor['estimated_errors'][binding_goal]) if known else None,
                        completion_probability=(float(gate_result['probability'][0])
                            if stop_eligible and gate_result is not None and gate_result['probability'] is not None else None),
                        completion_estimated_error_rad=(float(gate_result['estimated_error'][0])
                            if stop_eligible and gate_result is not None else None),
                        completion_gate_accept=(bool(accept) if stop_eligible else None),
                        true_error_rad=error,fit_error_m=float(sensor['fit_error_m'][0]),
                        controller_phase=('waiting' if pending else 'execution') if controller is not None else 'none',
                        residual_l2=float(result['residual'].norm()) if controller is not None else 0.,
                        speed=speed,drop=dropped,reason=reason,
                        injected_pending=injected_pending,injected_call=injected_call,
                        extra_delay_s=args.extra_delay_s))+'\n')
                    if pending and reason is None:
                        delay=(runtime.steps-request_step)/30-(time.monotonic()-request_wall)
                        if delay>0:time.sleep(delay)
                    if runtime.steps%150==0:
                        save(args.output/'progress.json',dict(status='running',completed_episodes=len(rows),
                            current_episode=ep,step=runtime.steps,submitted_calls=submitted,
                            elapsed_seconds=time.monotonic()-start))
                # Finish accounting even when the physical episode has ended.
                # This does not resume or pause any still-valid physical trial.
                if future is not None:
                    ready=future.result();future=None
                    ready.update(call=submitted,request_step=request_step,delivery_step=None)
                if ready is not None:
                    if ready['error'] is not None:excluded='service_error'
                    elif ready['elapsed_s']>120:excluded='request_latency_over_120s'
                    calls.append(ready)
                runtime.finish_video(reason)
                save(epdir/'calls.json',calls)
                row=dict(episode=ep,trial_seed=trial_seed,goal_index=goal,symbol=symbol,instruction=instruction,
                    initial_state=initial,steps=runtime.steps,reason=reason,success=reason=='success',
                    exclusion_reason=excluded,drop_phase=drop_phase,first_skill_delivery_step=first_skill_delivery_step,
                    stop_measurement=stop_measurement,minimum_error_rad=minimum,reached=bool(reached),
                    final_error_rad=error,declared_symbol=binding_symbol,declared_goal=binding_goal,
                    actual_model_calls=0 if replay is not None else submitted,
                    replayed_calls=submitted if replay is not None else 0,
                    latencies_s=[c['elapsed_s'] for c in calls],
                    injected_call=injected_call,extra_delay_s=args.extra_delay_s,
                    injected_wait_drop=injected_wait_drop,
                    injected_wait_max_rotation_drift_rad=intervention_max_rotation_drift,
                    injected_wait_max_translation_drift_m=intervention_max_translation_drift,
                    intervention_anchor=intervention_anchor)
                rows.append(row)
                with (args.output/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                progress=dict(status='running',completed_episodes=len(rows),requested_episodes=args.episodes,
                    success=sum(r['success'] for r in rows),excluded=sum(r['exclusion_reason'] is not None for r in rows),
                    drops=sum(r['reason']=='drop' for r in rows),actual_calls=sum(r['actual_model_calls'] for r in rows),
                    elapsed_seconds=time.monotonic()-start)
                save(args.output/'progress.json',progress)
                print('[LIVE_AGENT] '+json.dumps(progress),flush=True)
                if excluded is not None:
                    # Systemic API/auth failures are actionable; do not burn
                    # forty raw slots against an unavailable service.
                    status='service_blocked'
                    break
                if reason=='interrupted':break
        if status!='service_blocked':
            status='completed' if len(rows)==args.episodes and all(r['reason']!='interrupted' for r in rows) else 'interrupted'
    except Exception as exc:
        save(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        eligible=[r for r in rows if r['exclusion_reason'] is None]
        result=dict(status=status,provider=args.provider,maintenance=args.maintenance,requested_episodes=args.episodes,
            recorded_episodes=len(rows),success=sum(r['success'] for r in rows),
            raw_success_rate=sum(r['success'] for r in rows)/len(rows) if rows else None,
            eligible_episodes=len(eligible),eligible_success_rate=sum(r['success'] for r in eligible)/len(eligible) if eligible else None,
            matched_arm_exclusions_pending=True,
            reasons={key:sum(r['reason']==key for r in rows) for key in sorted({r['reason'] for r in rows})},
            drop_phases={key:sum(r['drop_phase']==key for r in rows) for key in ('startup_wait','wait','execution')},
            actual_model_calls=sum(r['actual_model_calls'] for r in rows),
            formal_agent_result=formal_release is not None,
            formal_condition=args.formal_condition,
            formal_release_sha256=sha(args.formal_release) if formal_release is not None else None,
            evaluation_scope=('runtime_preparation' if args.prepare_only else 'fixed_public_plan_control_diagnostic' if replay is not None else 'live_semantic_agent'),
            elapsed_seconds=time.monotonic()-start)
        save(args.output/'result.json',result)
        save(args.output/'progress.json',result)
        if runtime is not None:runtime.close()

if __name__=='__main__':main()
