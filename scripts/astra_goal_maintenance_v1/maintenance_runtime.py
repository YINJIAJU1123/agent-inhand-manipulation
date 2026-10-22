"""Apply a validated continuation policy after the first real skill delivery."""
import importlib.util
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/astra_runtime_v2'))
# Import the compatibility wrapper under a distinct name, avoiding this module.
spec=importlib.util.spec_from_file_location('_maintenance_base_runtime',ROOT/'scripts/astra_runtime_v2/runtime.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('_maintenance_policy_model',Path(__file__).with_name('model.py'))
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
from recovery_controller_v2 import observation


def load_validated(path,result_path,device):
    import hashlib,json
    result=json.loads(Path(result_path).read_text())
    if not result.get('local_gate_passed') or result.get('status')!='completed':
        raise RuntimeError('Continuation policy did not pass the declared local comparison')
    if not result.get('intervention_verified', False):
        raise RuntimeError('Policy has not demonstrated a nonzero control intervention')
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=result['checkpoint_sha256']:
        raise RuntimeError('Validated policy checksum mismatch')
    pack=torch.load(path,map_location=device,weights_only=True)
    if pack['protocol']['version']!='goal_maintenance_v1' or pack['iteration']!=32:
        raise RuntimeError('Wrong continuation training protocol')
    from common import STUDENT,ADAPTER,CONTROLLER,sha
    for p,k in [(STUDENT,'student_sha256'),(ADAPTER,'adapter_sha256'),(CONTROLLER,'controller_sha256')]:
        if sha(p)!=pack['protocol'][k]:raise RuntimeError('Frozen control asset mismatch')
    model=policy.ModeController().to(device).eval();model.load_state_dict(pack['model']);model.requires_grad_(False)
    return model


class GoalMaintenanceRuntime(base.ControllerRuntime):
    def __init__(self,original,initial,maintenance):
        super().__init__(original,initial,False)
        self.maintenance=maintenance;self.seen_skill=torch.zeros(len(initial),dtype=torch.bool,device=initial.device)
        self.mode=torch.zeros(len(initial),dtype=torch.long,device=initial.device)
    @torch.inference_mode()
    def action(self,step,goal,prop,proposal,current,pending,known,delivered,request_age):
        delivered=torch.as_tensor(delivered,dtype=torch.bool,device=current.device)
        started=pending&~self.was_pending
        changed=torch.zeros_like(known) if self.last_goal is None else (goal!=self.last_goal).any(-1)
        baseline,diagnostic=super().action(step,goal,prop,proposal,current,pending,known,delivered,request_age)
        # A local startup skill may be known to the base controller. Only a
        # delivered high-level skill activates this learned intervention.
        self.seen_skill|=delivered&known
        eligible=known&self.seen_skill
        age=torch.as_tensor(request_age,device=current.device,dtype=current.dtype)
        age=torch.where(pending,age,self.wait_age)
        response=torch.as_tensor(diagnostic['response_age'],device=current.device,dtype=current.dtype).clamp(max=2)
        obs=observation(self.history,prop,proposal,current,self.anchor,self.previous,age,pending,response)
        refresh=eligible & (delivered|started|changed|(step%2==0))
        if refresh.any():self.mode[refresh]=self.maintenance.distribution(obs[refresh],goal[refresh]).probs.argmax(-1)
        selected=policy.command(self.mode,baseline,proposal,current)
        action=torch.where(eligible[:,None],selected,baseline)
        diagnostic.update(maintenance_active=eligible.cpu().tolist(),maintenance_mode=self.mode.cpu().tolist())
        return action,diagnostic
