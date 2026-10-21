"""One controlled factorial PPO update with six independent draws and explicit weights."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/agentic'))
from recovery_controller_v3 import RecoveryController,make_optimizers,OutcomeReplay
from weighted_ppo import update
from design import ARMS,merge_draws

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--collections',nargs=6,required=True)
 p.add_argument('--slot-weights',nargs=12,type=float,required=True);p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--output',required=True);p.add_argument('--iteration',type=int,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(184100+a.iteration)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 checkpoint_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
 packs=[torch.load(f,map_location='cpu',weights_only=False) for f in a.collections]
 if any(x['checkpoint_sha256']!=checkpoint_sha for x in packs):raise RuntimeError('Rollout policy is not checkpoint policy')
 model=RecoveryController(ck['obs_dim']);model.load_state_dict(ck['model']);opts=make_optimizers(model)
 if ck.get('training_version')=='factorial_v1':
  for k,o in opts.items():o.load_state_dict(ck['optimizer'][k])
 rollout,success=merge_draws(packs,a.arm)
 stats=update(model,opts,rollout,success,a.slot_weights*6)
 # Outcome replay uses original episodes, not duplicated hard-state weights.
 original,original_success=rollout,success;replay=OutcomeReplay()
 if 'paired_outcome_replay' in ck:
  r=ck['paired_outcome_replay'];replay.train=r['train'];replay.validation=r['validation'];replay.rng.setstate(r['rng_state'])
 replay.add(original,original_success,packs[0]['seed']);stats.update(replay.fit(model,opts['outcome'],'cpu'))
 manifest=dict(ck['manifest'],version='recovery_factorial_v1',training='six draws per state, controlled correct-goal pairing and explicit training-only slot weights',arm=a.arm,factors=ARMS[a.arm])
 pack=dict(model=model.state_dict(),optimizer={k:o.state_dict() for k,o in opts.items()},obs_dim=model.obs_dim,iteration=a.iteration,
  parameters=sum(x.numel() for x in model.parameters()),manifest=manifest,training_version='factorial_v1',
  paired_outcome_replay=dict(train=replay.train,validation=replay.validation,rng_state=replay.rng.getstate()))
 torch.save(pack,out/'controller.pt')
 stats.update(iteration=a.iteration,seed=packs[0]['seed'],delay=packs[0]['delay'],arm=a.arm,slot_weights=a.slot_weights,
  nominal_training_slots=72,eligible=sum(int(x['rollout'][0]['valid'].sum()) for x in packs),
  successes=int(original_success.sum()),source_checkpoint_sha256=checkpoint_sha,
  output_checkpoint_sha256=hashlib.sha256((out/'controller.pt').read_bytes()).hexdigest(),completed=time.time())
 (out/'update.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
