"""Fresh-process prefix replay and continuation probe, no high-level model calls."""
import argparse,json,sys
from pathlib import Path
from viserdex.utils.runtime import add_runtime_args,parse_runtime_args

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime_sources',type=Path,required=True)
known,_=parser.parse_known_args();sys.path.insert(0,str(known.runtime_sources))
from newton_runtime import NewtonRuntime,add_arguments
add_arguments(parser);parser.add_argument('--reference',type=Path)
parser.add_argument('--relocation_manifest',type=Path)
add_runtime_args(parser);args=parse_runtime_args(parser)
if args.seed!=182961:raise ValueError('Only the registered development probe seed is permitted')
if args.record_video:raise ValueError('Probe excludes rendering overlays')
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
import torch
from reconstruction_state import capture,compare,digest

runtime=None
try:
    relocation=None
    if args.relocation_manifest:
        from relocated_environment import install
        relocation=install(args.relocation_manifest)
    runtime=NewtonRuntime(args);initial=runtime.reset(args.seed*1000)
    samples=[];actions=[];status='failed';reference=None
    if args.reference:
        reference=torch.load(args.reference/'trajectory.pt',map_location='cpu',weights_only=True)
        if reference['initial_state']!=initial:raise ValueError('Initial reset state mismatch')
    with torch.inference_mode():
        # Fixed prefix is independent of either future target and every actor.
        for step in range(120):
            pending=step<30
            if step==0:runtime.begin_request()
            if step==30:runtime.set_skill(3)
            if step==90:runtime.set_skill(1)
            values,hidden=runtime.propose(pending,False,step/30 if pending else 0.)
            proposal=values['proposal'].clamp(-1,1);current=values['proprio'][:,:21].clamp(-1,1)
            action=current+.15*(proposal-current) if pending else proposal
            if reference is not None:
                # Use the canonical action sequence, while retaining recomputed
                # hidden and observations for a stringent reconstruction check.
                action=reference['actions'][step].to(runtime.device)
            if step in (60,90,119):
                snapshot=capture(runtime);samples.append(dict(step=step,state=snapshot,digest=digest(snapshot)))
            terminal=runtime.step(action,hidden);actions.append(action.cpu().clone())
            if bool(terminal['drop'][0]) or bool(terminal['done'][0]):
                raise RuntimeError('Probe prefix/continuation terminated before completion; do not admit paired-state experiment')
    result=dict(initial_state=initial,actions=actions,snapshots=samples,metadata=runtime.metadata,relocation=relocation)
    torch.save(result,args.output/'trajectory.pt')
    checks=[]
    if reference is not None:
        for a,b in zip(reference['snapshots'],samples,strict=True):
            assert a['step']==b['step'];checks.append(dict(step=b['step'],**compare(a['state'],b['state'])))
    report=dict(status='completed',formal_trials_used=False,actual_model_calls=0,
        snapshots=[dict(step=x['step'],digest=x['digest']) for x in samples],
        comparisons=checks,paired_state_admitted=bool(checks) and all(c['exact_match'] for c in checks),
        scope='Fresh-process prefix reconstruction plus identical-action continuation. Not a semantic success-rate estimate.')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if checks and not report['paired_state_admitted']:raise RuntimeError('Reconstruction mismatch; inspect named fields before any paired goals')
finally:
    if runtime is not None:runtime.close()
