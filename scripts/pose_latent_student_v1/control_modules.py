"""Deployment-only sensor features for new pose-latent maintenance modules."""
import torch
from torch import nn
from torch.nn import functional as F

FIELDS = ('proprio', 'measured_pose', 'belief', 'proposal', 'anchor_pose', 'anchor_action',
          'fit_error_m', 'known', 'pending', 'elapsed_pending_s')
TOLERANCES = (.08, .12, .16, .22, .32, .5, 1.)

def features(values):
    # No labels, simulator pose, goal identity index or teacher actions accepted.
    pieces = []
    for key in FIELDS:
        value = values[key].float()
        pieces.append(value[:, None] if value.ndim == 1 else value)
    return torch.cat(pieces, -1)

def controller_goal_input(student_latent, bank, active_goal, intervention):
    # Never mutate the policy latent or replace it in the student runtime.
    if intervention == 'none':
        return student_latent
    if intervention == 'cyclic_shift':
        return bank[(int(active_goal)+1)%len(bank)].unsqueeze(0)
    raise ValueError('Unknown actor-only goal intervention')


class SensorGoalEncoder(nn.Module):
    def __init__(self, mean, scale, bank):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('goal_center', bank.mean(0))
        self.register_buffer('goal_scale', (bank-bank.mean(0)).square().mean(0).sqrt().clamp_min(.005))
        self.sensor = nn.Sequential(nn.Linear(len(mean),192),nn.LayerNorm(192),nn.SiLU())
        self.goal = nn.Sequential(nn.Linear(bank.shape[1],64),nn.LayerNorm(64),nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(256,128),nn.SiLU())
    def forward(self, sensor, goal):
        x = ((sensor-self.mean)/self.scale).clamp(-8,8)
        g = ((goal-self.goal_center)/self.goal_scale).clamp(-8,8)
        return self.fusion(torch.cat((self.sensor(x),self.goal(g)),-1))

class CompletionEvidence(nn.Module):
    def __init__(self, mean, scale, bank):
        super().__init__()
        self.encoder = SensorGoalEncoder(mean,scale,bank)
        self.ordinal = nn.Linear(128,len(TOLERANCES))
        self.error = nn.Linear(128,1)
    def forward(self,sensor,goal):
        x = self.encoder(sensor,goal)
        return dict(ordinal=self.ordinal(x),log_error=self.error(x).squeeze(-1))

class MaintenanceActor(nn.Module):
    def __init__(self, mean, scale, bank, goal_input='active', execution_speed_mode='fixed'):
        super().__init__()
        if goal_input not in ('active', 'center'):
            raise ValueError('Unknown controller goal input')
        self.goal_input = goal_input
        if execution_speed_mode not in ('fixed','fixed_matched','learned'):
            raise ValueError('Unknown execution speed mode')
        self.execution_speed_mode = execution_speed_mode
        self.encoder = SensorGoalEncoder(mean,scale,bank)
        self.actor = nn.Linear(128,22)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        self.execution_speed_head = None
        if execution_speed_mode != 'fixed':
            # Preserve the legacy RNG stream so paired runs see the same batches.
            with torch.random.fork_rng(devices=[]):
                self.execution_speed_head = nn.Linear(128,1)
            nn.init.zeros_(self.execution_speed_head.weight)
            nn.init.constant_(self.execution_speed_head.bias, float(torch.logit(torch.tensor(.95))))
    def forward(self,sensor,goal,proposal,current,pending):
        if self.goal_input == 'center':
            goal = self.encoder.goal_center.expand_as(goal)
        encoded = self.encoder(sensor,goal)
        raw = self.actor(encoded)
        # Zero residual retains the declared baseline: 15% interpolation while
        # waiting and full student execution after delivery.
        wait_speed = torch.sigmoid(raw[:,0]*4 + torch.logit(raw.new_tensor(.15)))
        execution_speed = (0.05+0.95*torch.sigmoid(self.execution_speed_head(encoded).squeeze(-1))
            if self.execution_speed_mode == 'learned' else torch.ones_like(wait_speed))
        speed = torch.where(pending,wait_speed,execution_speed)
        if self.execution_speed_mode == 'learned':
            base = current+speed[:,None]*(proposal.clamp(-1,1)-current)
        else:
            # Keep the old fixed-speed floating-point operations unchanged.
            base = torch.where(pending[:,None],current+speed[:,None]*(proposal.clamp(-1,1)-current),proposal.clamp(-1,1))
        residual = .35*raw[:,1:].tanh()
        return dict(action=(base+residual).clamp(-1,1),speed=speed,residual=residual)


def load_maintenance_actor(path,student_sha256,bank,device):
    if path is None:raise ValueError('Maintenance checkpoint required')
    pack=torch.load(path,map_location='cpu',weights_only=True)
    if pack['manifest']['source_student_sha256']!=student_sha256:
        raise ValueError('Maintenance checkpoint student mismatch')
    torch.testing.assert_close(pack['bank'],bank.detach().cpu(),atol=0,rtol=0)
    actor=MaintenanceActor(pack['feature_mean'],pack['feature_scale'],pack['bank'],
        goal_input=pack['manifest'].get('actor_goal_input','active'),
        execution_speed_mode=pack['manifest'].get('actor_execution_speed_mode','fixed')).to(device).eval()
    actor.load_state_dict(pack['model'])
    actor.requires_grad_(False)
    return actor


class PoseCompletionCorrection(nn.Module):
    """Learn a bounded sensor-error correction around the measured orientation.

    This never changes the scored 0.16 rad threshold. Its uncertainty is
    calibrated on separate trajectory groups before selecting a stop gate.
    """
    def __init__(self,mean,scale,bank):
        super().__init__()
        self.encoder=SensorGoalEncoder(mean,scale,bank)
        self.output=nn.Sequential(nn.Linear(129,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
    def forward(self,sensor,goal,measured_angle):
        encoded=self.encoder(sensor,goal)
        raw=self.output(torch.cat((encoded,measured_angle[:,None]),-1))
        correction=.12*raw[:,0].tanh()
        sigma=.005+.25*torch.sigmoid(raw[:,1]-2.)
        return dict(mean=(measured_angle+correction).clamp(0,torch.pi),
                    sigma=sigma,correction=correction)


class CompletionGate:
    """A calibrated internal stop decision; independent task scoring stays fixed."""
    def __init__(self,path,student_sha256,bank,device,calibration=None):
        pack=torch.load(path,map_location='cpu',weights_only=True)
        if pack['manifest']['source_student_sha256']!=student_sha256:
            raise ValueError('Completion checkpoint student mismatch')
        torch.testing.assert_close(pack['bank'],bank.detach().cpu(),atol=0,rtol=0)
        self.measured_threshold=.16
        if pack['manifest']['format']=='pose_completion_correction_v1':
            if calibration is not None:
                raise ValueError('Event calibration requires an ordinal evidence checkpoint')
            if pack['beta'] is None:raise ValueError('Completion gate was not calibrated')
            self.kind='pose_correction'
            self.model=PoseCompletionCorrection(pack['feature_mean'],pack['feature_scale'],pack['bank'])
            self.beta=float(pack['beta'])
        elif pack['manifest']['format']=='pose_latent_control_modules_v1':
            self.kind='ordinal_evidence'
            self.model=CompletionEvidence(pack['feature_mean'],pack['feature_scale'],pack['bank'])
            self.threshold=pack['threshold']
            if calibration is not None:
                import hashlib,json
                from pathlib import Path
                selected=json.loads(Path(calibration).read_text())
                if selected['evidence_sha256']!=hashlib.sha256(Path(path).read_bytes()).hexdigest():
                    raise ValueError('Calibration checkpoint mismatch')
                if selected['source_student_sha256']!=student_sha256 or selected['scored_tolerance_rad']!=.16:
                    raise ValueError('Calibration contract mismatch')
                self.threshold=selected['evidence_probability_threshold']
                self.measured_threshold=selected['measured_stop_threshold_rad']
            if self.threshold is None or not 0<self.threshold<1 or not 0<self.measured_threshold<=.16:
                raise ValueError('Completion gate was not calibrated')
        else:
            raise ValueError('Unexpected completion checkpoint')
        self.model=self.model.to(device).eval()
        self.model.load_state_dict(pack['model']);self.model.requires_grad_(False)

    @torch.inference_mode()
    def estimate(self,values,goal,measured_angle):
        if self.kind=='ordinal_evidence':
            pred=self.model(features(values),goal)
            probability=pred['ordinal'][:,TOLERANCES.index(.16)].sigmoid()
            return dict(accept=probability>=self.threshold,
                estimated_error=(pred['log_error'].exp()-.02).clamp(0,torch.pi),
                uncertainty=None,stop_bound=None,probability=probability)
        pred=self.model(features(values),goal,measured_angle)
        upper=pred['mean']+self.beta*pred['sigma']
        return dict(accept=upper<=.16,estimated_error=pred['mean'],
                    uncertainty=pred['sigma'],stop_bound=upper,probability=None)
