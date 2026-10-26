"""Public-skill actor selection with the paper's fixed published assets."""
import hashlib
import json
from pathlib import Path
import torch
import adapter as _adapter
ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text())
def validate_run(directory, choice, offline=False):
    if choice != 'mixed':
        raise ValueError('The public paper comparison fixes the mixed actor')
    path=Path(directory).resolve()
    if path != ROOT/'configs/runtime_assets.json':
        raise ValueError('Use the published runtime asset manifest')
    manifest=read(path)
    for name,asset in manifest['assets'].items():
        p=ROOT/asset['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=asset['sha256']:
            raise RuntimeError('Published asset changed: '+name)
    actor=manifest['assets']['adapted_action_student']
    if actor['sha256']!='b02d80830dcd4aebeec14c7d30e8645a47dc168a6ce059165d4744f683359862':
        raise RuntimeError('Expected the preselected paper actor')
    return dict(offline_only=offline,actor_choice=choice,checkpoint_sha256=actor['sha256'],
        scope='Fixed published checkpoint identity. Training selection is inherited from the recorded paper protocol, not repeated by this loader.')

class PublicActorRuntime:
    def __init__(self,source,pack,directory,choice,mode):
        if mode not in ('original','candidate'):raise ValueError('Unknown actor arm')
        training=Path(directory).resolve()
        self.actors=_adapter.FittedActors(source,pack,training)
        self.choice,self.mode=choice,mode
        self.changed_steps=None
        self.trace=[]

    @torch.inference_mode()
    def action(self,step,original_output,batch,known,active):
        if known.dtype!=torch.bool or active.dtype!=torch.bool or known.shape!=active.shape:
            raise ValueError('Skill and active masks must be boolean per environment')
        choices=self.actors.proposals(original_output,batch)
        enabled=known & active & (self.mode=='candidate')
        selected=self.actors.select(choices,self.choice,enabled)
        changed=(selected-original_output['action']).abs().amax(-1)>1e-6
        if self.changed_steps is None:self.changed_steps=torch.zeros_like(known,dtype=torch.long)
        self.changed_steps+=changed & active
        if not torch.equal(selected[~enabled],original_output['action'][~enabled]):
            raise RuntimeError('Original startup/disabled action changed')
        if step%10==0:
            self.trace.append(dict(step=step,known=known.cpu().tolist(),active=active.cpu().tolist(),
                enabled=enabled.cpu().tolist(),changed=changed.cpu().tolist(),
                original_proposal=original_output['action'].cpu().tolist(),selected_proposal=selected.cpu().tolist()))
        return selected
