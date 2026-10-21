"""Pre-registered factors and merge rules. No evaluation-driven curriculum."""
import torch

ARMS={
 'uniform_original':dict(pairing=False,weighting=False),
 'paired_only':dict(pairing=True,weighting=False),
 'weighted_only':dict(pairing=False,weighting=True),
 'paired_weighted':dict(pairing=True,weighting=True),
}
SHARED_PREFIX={'weighted_only':'uniform_original','paired_weighted':'paired_only'}


def branch_for(arm,draw_id):
 if arm not in ARMS or draw_id not in range(6):raise ValueError('Unknown factorial arm/draw')
 return draw_id if ARMS[arm]['pairing'] else 0


def merge_draws(packs,arm):
 if len(packs)!=6 or {p['draw_id'] for p in packs}!=set(range(6)):raise ValueError('Missing or duplicate stochastic draws')
 for key in ['seed','delay','checkpoint_sha256','request_cache_sha256','training_arm']:
  if len({p[key] for p in packs})!=1:raise ValueError('Unmatched '+key)
 if packs[0]['training_arm']!=arm:raise ValueError('Wrong training arm')
 for p in packs:
  if p['branch']!=branch_for(arm,p['draw_id']):raise ValueError('Incorrect goal branch')
 if len({p['noise_seed'] for p in packs})!=6:raise ValueError('Draws must have distinct registered noise seeds')
 packs=sorted(packs,key=lambda p:p['draw_id'])
 if len({len(p['rollout']) for p in packs})!=1 or not packs[0]['rollout']:raise ValueError('Rollout lengths differ')
 for p in packs:
  if not torch.equal(p['rollout'][0]['valid'],packs[0]['rollout'][0]['valid']):raise ValueError('Eligibility differs across goals/draws')
 rollout=[{k:torch.cat([p['rollout'][t][k] for p in packs],dim=0) for k in packs[0]['rollout'][t]} for t in range(len(packs[0]['rollout']))]
 return rollout,torch.cat([p['success'] for p in packs])


def weights_for(arm,registered):
 if len(registered)!=12 or any(v not in [1,2] for v in registered):raise ValueError('Expected twelve frozen slot weights')
 return registered if ARMS[arm]['weighting'] else [1]*12
