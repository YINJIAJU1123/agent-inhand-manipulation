"""Goal pairing and failure curriculum without simulator state restoration.

Only training outcomes may select difficult states. A selected physical slot is
weighted equally under all six goals. Action diversity is never a reward.
"""
import hashlib
import torch

PHYSICAL_KEYS=('joint_pos','joint_vel','object_root','prev_targets','cur_targets','hand_dof_targets','actions','prev_actions')


def paired_goal_ids(base,branch):
    if branch not in range(6) or base.ndim!=1 or ((base<0)|(base>5)).any():
        raise ValueError('Expected canonical face ids and one of six branches')
    # Branch 1 is the opposing face. All six branches form a balanced mapping.
    return 2*((base//2+branch//2)%3)+(base%2+branch%2)%2


def tensor_digest(values):
    h=hashlib.sha256()
    for key,value in sorted(values.items()):
        t=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def physical_state(raw):
    return dict(joint_pos=raw.hand.data.joint_pos.clone(),joint_vel=raw.hand.data.joint_vel.clone(),
        object_root=raw.object.data.root_state_w.clone(),**{k:getattr(raw,k).clone() for k in PHYSICAL_KEYS[3:]})


def switch_goal(raw,goals,cached,ih,ph,vh):
    before=physical_state(raw);counter=int(raw.common_step_counter)
    prop=raw.compute_student_proprio().clone()
    for face in range(6):
        raw.cfg.fixed_target_face=face
        raw._reset_target_pose((goals==face).nonzero().flatten())
    raw.cfg.fixed_target_face=None;raw.goal_markers.set_visibility(False)
    after=physical_state(raw)
    errors={k:float((after[k]-before[k]).abs().max()) for k in PHYSICAL_KEYS}
    prefix_errors={k:float((before[k].cpu()-cached[k]).abs().max()) for k in PHYSICAL_KEYS}
    if max(errors.values())!=0 or max(prefix_errors.values())!=0:raise RuntimeError('Goal switch changed physical state or mismatched cached prefix')
    if int(raw.common_step_counter)!=counter:raise RuntimeError('Goal switch advanced simulation')
    if not torch.equal(prop,raw.compute_student_proprio()):raise RuntimeError('Goal leaked into proprioception')
    if not torch.equal(raw.target_face,goals):raise RuntimeError('Environment goal does not match delivered goal')
    for current,key in [(ih,'ih'),(ph,'ph'),(vh,'visual_history')]:
        if not torch.equal(current.cpu(),cached[key]):raise RuntimeError('Request history changed across goal branches')
    return dict(max_physical_change=max(errors.values()),max_cached_state_error=max(prefix_errors.values()),
        physics_steps_advanced=int(raw.common_step_counter)-counter,proprio_unchanged=True,
        physical_sha256=tensor_digest(before),request_observation_sha256=tensor_digest(dict(ih=ih,ph=ph,vh=vh)),
        environment_goals=raw.target_face.cpu().tolist(),goal_quaternions=raw.goal_rot.cpu().tolist())


def validate_group(packs):
    if len(packs)!=6 or {p['branch'] for p in packs}!=set(range(6)):
        raise ValueError('Every frozen-policy update requires all six correct goals')
    for key in ('seed','delay','checkpoint_sha256','request_cache_sha256'):
        if len({p[key] for p in packs})!=1:raise ValueError('Unmatched group '+key)
    lengths={len(p['rollout']) for p in packs}
    if len(lengths)!=1 or not next(iter(lengths)):raise ValueError('Rollout horizons differ')
    initial=packs[0]['rollout'][0]['valid']
    for p in packs:
        if not torch.equal(p['rollout'][0]['valid'],initial):raise ValueError('Goal-dependent eligibility')


def merge_group(packs,slot_multiplicity=None):
    validate_group(packs);packs=sorted(packs,key=lambda p:p['branch'])
    n=len(packs[0]['success']);mult=torch.ones(n,dtype=torch.long) if slot_multiplicity is None else torch.as_tensor(slot_multiplicity,dtype=torch.long)
    if mult.shape!=(n,) or ((mult<1)|(mult>2)).any():raise ValueError('Slot multiplicity must be one or two')
    ids=torch.arange(n).repeat_interleave(mult)
    merged=[{key:torch.cat([p['rollout'][t][key][ids] for p in packs],dim=0) for key in packs[0]['rollout'][t]} for t in range(len(packs[0]['rollout']))]
    success=torch.cat([p['success'][ids] for p in packs])
    return merged,success


def hard_slot_multiplicity(rows,n=12):
    """Previous training group only, fixed before the next group is collected."""
    if len(rows)!=6*n:raise ValueError('Need complete prior six-goal training outcomes')
    result=[]
    for slot in range(n):
        cases=[r for r in rows if r['slot']==slot]
        if len(cases)!=6 or {r['goal_branch'] for r in cases}!=set(range(6)):raise ValueError('Missing training branch')
        if len({r['eligible'] for r in cases})!=1:raise ValueError('Eligibility changed with goal')
        failures=sum(r['eligible'] and not r['success'] for r in cases)
        drops=sum(r['eligible'] and (r['wait_drop'] or r['resume_drop']) for r in cases)
        result.append(2 if failures>=2 or drops else 1)
    return result
