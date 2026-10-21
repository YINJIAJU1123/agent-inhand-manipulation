"""Public command schedule; does not inspect simulation state or task labels."""
import torch


def command_state(step, cases=6):
    request_id = step // 240 + 1
    delivered_count = max(0, (step - 180) // 240 + 1)
    phase = 'initial' if delivered_count == 0 else 'inspect' if delivered_count <= 2 else 'execute'
    colors = [None] * cases if not delivered_count else [(i+2)%6 if delivered_count == 2 else i%6 for i in range(cases)]
    delivered = step >= 180 and (step-180)%240 == 0
    pending = request_id > delivered_count
    age = (step-(request_id-1)*240)/30 if pending else 6. if delivered else 0.
    return dict(phase=phase, colors=colors, request_id=request_id, delivered=delivered, pending=pending, age=age)


def physical(raw):
    return {k:v.detach().cpu().clone() for k,v in dict(joint_pos=raw.hand.data.joint_pos,
        joint_vel=raw.hand.data.joint_vel, object_root=raw.object.data.root_state_w,
        prev_targets=raw.prev_targets, cur_targets=raw.cur_targets,
        actions=raw.actions, prev_actions=raw.prev_actions).items()}


def assert_equal(actual, expected, tolerance=1e-6):
    if actual.keys() != expected.keys():
        raise ValueError('Physical fields differ')
    errors = {}
    for key, target in expected.items():
        value = actual[key]
        if value.shape != target.shape or (not torch.isfinite(value).all() or not torch.isfinite(target).all()):
            raise ValueError('Invalid state '+key)
        errors[key] = float((value-target).abs().max())
    if max(errors.values()) > tolerance:
        raise RuntimeError('Physical prefix mismatch: '+str(errors))
    return errors


def pose_errors(quaternion, colors):
    # Scoring only, never a trigger, input feature, or action selector.
    s = 2**-.5
    poses = quaternion.new_tensor([[s,0,-s,0],[s,0,s,0],[s,s,0,0],[s,-s,0,0],[1,0,0,0],[0,1,0,0]])
    return 2*torch.acos((quaternion*poses[colors]).sum(-1).abs().clamp(0,1))
