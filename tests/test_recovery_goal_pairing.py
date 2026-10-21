"""Pairing must preserve causal comparisons and on-policy PPO groups."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/agentic'))
from recovery_goal_pairing import paired_goal_ids,merge_group,hard_slot_multiplicity,switch_goal,physical_state

class PairingTests(unittest.TestCase):
 def test_six_correct_goals_and_opposite_pair(self):
  for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
   base=torch.arange(12,device=device)%6
   all_goals=torch.stack([paired_goal_ids(base,b) for b in range(6)])
   self.assertTrue(torch.equal(all_goals.sort(0).values,torch.arange(6,device=device)[:,None].expand(6,12)))
   self.assertTrue(torch.equal(all_goals[1],base^1))
   for branch in all_goals:self.assertEqual(torch.bincount(branch).tolist(),[2]*6)
 def test_reject_mixed_policy_and_keep_episode_boundaries(self):
  packs=[]
  for branch in range(6):
   rollout=[dict(obs=torch.full((2,3),branch*10+t),valid=torch.ones(2,dtype=torch.bool),done=torch.full((2,),t==2)) for t in range(3)]
   packs.append(dict(branch=branch,seed=930100,delay=6.,checkpoint_sha256='same',request_cache_sha256='same',rollout=rollout,success=torch.tensor([True,False])))
  merged,success=merge_group(packs,[2,1])
  self.assertEqual(merged[0]['obs'][:,0].tolist(),[b*10 for b in range(6) for _ in range(3)])
  self.assertTrue(merged[-1]['done'].all());self.assertFalse(merged[0]['done'].any())
  self.assertEqual(success.tolist(),[True,True,False]*6)
  packs[-1]['checkpoint_sha256']='different'
  with self.assertRaises(ValueError):merge_group(packs)
 def test_failure_curriculum_weights_all_goals_equally(self):
  rows=[dict(slot=i,goal_branch=b,eligible=i!=2,success=(i!=0 or b>1),wait_drop=False,resume_drop=False) for b in range(6) for i in range(3)]
  self.assertEqual(hard_slot_multiplicity(rows,3),[2,1,1])
  with self.assertRaises(ValueError):hard_slot_multiplicity(rows[:-1],3)
 def test_goal_switch_rejects_physical_mutation_and_proprio_leak(self):
  n=6
  def make_raw(mutate=False):
   r=SimpleNamespace(hand=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(n,21),joint_vel=torch.zeros(n,21))),
     object=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(n,13))),common_step_counter=30,
     target_face=torch.arange(n),goal_rot=torch.zeros(n,4),cfg=SimpleNamespace(fixed_target_face=None),
     goal_markers=SimpleNamespace(set_visibility=lambda v:None))
   for key in ['prev_targets','cur_targets','hand_dof_targets','actions','prev_actions']:setattr(r,key,torch.zeros(n,21))
   r.compute_student_proprio=lambda:r.hand.data.joint_pos.clone()
   def set_goal(ids):
    r.target_face[ids]=r.cfg.fixed_target_face
    if mutate:r.actions[ids]+=1
   r._reset_target_pose=set_goal
   return r
  hist=torch.zeros(n,4,5)
  for bad in [False,True]:
   r=make_raw(bad);cached=dict(physical_state(r),ih=hist,ph=hist,visual_history=hist)
   if bad:
    with self.assertRaises(RuntimeError):switch_goal(r,torch.arange(n)^1,cached,hist,hist,hist)
   else:
    audit=switch_goal(r,torch.arange(n)^1,cached,hist,hist,hist)
    self.assertEqual(audit['max_physical_change'],0);self.assertEqual(audit['physics_steps_advanced'],0)

if __name__=='__main__':unittest.main()
