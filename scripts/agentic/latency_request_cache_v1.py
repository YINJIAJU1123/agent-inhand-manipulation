"""Shared request states for counterfactual delay evaluation (not actor inputs)."""
import torch

def capture(raw, ih, ph, visual, oldvisual, camera, eligible, manifest):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in dict(
        joint_pos=raw.hand.data.joint_pos, joint_vel=raw.hand.data.joint_vel,
        object_root=raw.object.data.root_state_w, prev_targets=raw.prev_targets,
        cur_targets=raw.cur_targets, hand_dof_targets=raw.hand_dof_targets,
        actions=raw.actions, prev_actions=raw.prev_actions,
        ih=ih, ph=ph, visual=visual, oldvisual=oldvisual, rgb=camera['rgb'],
        eligible=eligible, student_sha256=manifest['student_sha256'],
        adapter_sha256=manifest['adapter_sha256']).items()}

def restore(env, raw, state, seed, goals):
    """Reconstruct each branch from the same measured mechanical state.

    The GPU contact solver's hidden warm-start cache is not serialized. All
    branches therefore reset the environment before applying the same state.
    """
    env.reset(seed=seed)
    for face in range(6):
        raw.cfg.fixed_target_face=face
        raw._reset_target_pose((goals==face).nonzero().flatten())
    raw.cfg.fixed_target_face=None; raw.goal_markers.set_visibility(False)
    raw.hand.write_joint_state_to_sim(state['joint_pos'],state['joint_vel'])
    raw.object.write_root_state_to_sim(state['object_root'])
    for key in ('prev_targets','cur_targets','hand_dof_targets','actions','prev_actions'):
        getattr(raw,key).copy_(state[key])
    raw.hand.set_joint_position_target(raw.cur_targets[:,raw.actuated_dof_indices],joint_ids=raw.actuated_dof_indices)
    raw.scene.write_data_to_sim(); raw.sim.forward(); raw.scene.update(0.)
    raw._compute_intermediate_values(); raw._update_face_markers()
    actual=torch.cat((raw.hand.data.joint_pos,raw.hand.data.joint_vel,raw.object.data.root_state_w),-1)
    desired=torch.cat((state['joint_pos'],state['joint_vel'],state['object_root']),-1)
    error=float((actual-desired).abs().max())
    if error>1e-4:raise RuntimeError(f'Request state reconstruction error: {error}')
    return error
