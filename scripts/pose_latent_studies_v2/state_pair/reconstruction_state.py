"""Read-only evidence for paired-state reconstruction on the installed MJWarp backend.

This does not claim that a scene-pose reset restores solver history. It records
persistent solver inputs, policy/noise history and sensor outputs after a fresh
process replays the same prefix. A matching continuation must be checked too.
"""
import hashlib,json,random
import numpy as np
import torch


def capture(runtime):
    from isaaclab_newton.physics.newton_manager import NewtonManager
    import warp as wp
    solver=NewtonManager._solver
    if solver.use_mujoco_cpu or solver.enable_sleeping or solver.update_data_interval!=1:
        raise ValueError('Reconstruction audit only supports nonsleeping MJWarp with per-step state synchronization')
    arrays={};scalars={}
    def tensor(name,value):
        if value is None:return
        value=getattr(value,'torch',value)
        if isinstance(value,wp.array):value=torch.from_numpy(value.numpy().copy())
        elif not isinstance(value,torch.Tensor):raise TypeError(name+' is not an auditable array')
        value=value.detach().cpu().clone().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite state '+name)
        arrays[name]=value
    def direct(prefix,obj):
        for name,value in vars(obj).items():
            if isinstance(value,torch.Tensor):tensor(prefix+'.'+name,value)
            elif type(value) in (int,float,bool,str) or value is None:scalars[prefix+'.'+name]=value
    sensor=runtime.observe()
    for name in ('proprio','measured_pose','fit_error_m','estimated_errors','world_quaternion'):
        tensor('sensor.'+name,sensor[name])
    tensor('sensor.rgb',torch.from_numpy(sensor['rgb'].copy()))
    for name in ('hidden','latent','anchor_pose','anchor_action'):tensor('runtime.'+name,getattr(runtime,name))
    scalars.update({'runtime.steps':runtime.steps,'runtime.active_goal':runtime.active_goal,
        'env.common_step_counter':runtime.raw.common_step_counter,
        'env.sim_step_counter':runtime.raw._sim_step_counter,'sim.physics_step_count':runtime.raw.sim._physics_step_count})
    tensor('adapter.history',runtime.adapter.history)
    direct('joint_noise',runtime.adapter.joint_noise);direct('pose_noise',runtime.adapter.noise)
    for name,resource in [('state',NewtonManager.get_state_0()),('control',NewtonManager.get_control())]:
        # Public integration inputs, not uninitialized scratch/contact padding.
        fields=('joint_q','joint_qd','body_q','body_qd','body_f','joint_f') if name=='state' else ('joint_f','joint_target_pos','joint_target_vel')
        for field in fields:
            value=getattr(resource,field,None)
            if value is not None:tensor(name+'.'+field,value)
    model=NewtonManager.get_model()
    for name in ('body_mass','body_inertia','shape_material_ke','shape_material_kd',
            'shape_material_mu','shape_material_restitution','shape_material_mu_torsional',
            'shape_material_mu_rolling','joint_target_ke','joint_target_kd'):
        tensor('model.'+name,getattr(model,name))
    for name in ('qpos','qvel','qacc_warmstart','qfrc_applied','xfrc_applied','act','ctrl','time'):
        tensor('mjw.'+name,getattr(solver.mjw_data,name))
    direct('action_manager',runtime.raw.action_manager)
    direct('joint_action',runtime.raw.action_manager.get_term('joint_pos'))
    robot=runtime.raw.scene['robot'];obj=runtime.raw.scene['object']
    for label,entity in [('robot',robot),('object',obj)]:
        tensor(label+'.root_state',entity.data.root_state_w.torch)
    for name in ('joint_pos','joint_vel','joint_pos_target','joint_vel_target','joint_effort_target'):
        tensor('robot.'+name,getattr(robot.data,name))
    for name,actuator in robot.actuators.items():direct('actuator.'+name,actuator)
    tensor('env.episode_length_buf',runtime.raw.episode_length_buf)
    direct('command',runtime.command)
    tensor('rng.torch',torch.get_rng_state());tensor('rng.cuda',torch.cuda.get_rng_state(runtime.device))
    scalars['rng.python']=repr(random.getstate())
    rng=np.random.get_state();scalars['rng.numpy']=json.dumps([rng[0],rng[1].tolist(),*rng[2:]])
    return dict(format='paired_prefix_state_v1',arrays=arrays,scalars=scalars,
        scope='Explicit persistent-state audit plus sensor/policy history; admission also requires same-action continuation agreement.')


def compare(left,right):
    if set(left['arrays'])!=set(right['arrays']) or set(left['scalars'])!=set(right['scalars']):
        raise ValueError('Snapshot field sets differ')
    mismatches={}
    for name,a in left['arrays'].items():
        b=right['arrays'][name]
        if a.shape!=b.shape or a.dtype!=b.dtype:mismatches[name]={'shape_or_dtype':True}
        elif not torch.equal(a,b):
            mismatches[name]={'max_abs_difference':float((a.double()-b.double()).abs().max()) if a.numel() else 0.}
    for name,a in left['scalars'].items():
        if a!=right['scalars'][name]:mismatches[name]={'scalar_difference':True}
    return dict(exact_match=not mismatches,mismatches=mismatches,tensor_fields=len(left['arrays']),scalar_fields=len(left['scalars']))


def digest(snapshot):
    h=hashlib.sha256(json.dumps(snapshot['scalars'],sort_keys=True,allow_nan=False).encode())
    for name,value in sorted(snapshot['arrays'].items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()
