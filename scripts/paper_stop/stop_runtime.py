"""Published fixed stop heads and the original shared-only intervention."""
import hashlib
import json
from pathlib import Path
import torch
import stop_correction as heads
import stop_history as temporal
import completion as eligibility

ROOT=Path(__file__).resolve().parents[2]
EXPECTED={
    'bounded4':('1de68888cf485baaf1b49d2396581e66818ed0faabd2cd7eb3c70ce56ef7cd48',63),
    'unbounded':('5a931eb640287b2ee81af061805488f796cc57fd6c422764513f4cb9c7f0f16c',28),
}

def load_published_heads(manifest_path,device='cpu'):
    manifest_path=Path(manifest_path).resolve()
    if manifest_path!=ROOT/'configs/paper_stop_assets.json':
        raise ValueError('Use the published stop asset manifest')
    manifest=json.loads(manifest_path.read_text())
    if (manifest['version']!='paper_shared_stop_release_v1' or manifest['threshold']!=.7
            or manifest['frame_dim']!=125 or manifest['history']!=3 or manifest['cadence_control_steps']!=10
            or set(manifest['assets'])!=set(EXPECTED)):
        raise ValueError('Changed stop contract')
    loaded={};metadata={}
    for arm,(digest,epoch) in EXPECTED.items():
        asset=manifest['assets'][arm];path=ROOT/'checkpoints/runtime'/f'stop_{arm}.pt'
        if asset['path']!=str(path.relative_to(ROOT)) or asset['sha256']!=digest or asset['epoch']!=epoch:
            raise ValueError('Unknown published checkpoint identity')
        if path.stat().st_size!=asset['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Published stop checkpoint changed: '+arm)
        pack=torch.load(path,map_location='cpu',weights_only=True)
        if (pack['format']!='stop_onpolicy_learning_v1' or pack['threshold']!=.7
                or pack['bounded']!=(arm=='bounded4') or pack['epoch']!=epoch):
            raise ValueError('Stop checkpoint contract mismatch')
        model=heads.StopCorrection(arm=='bounded4').to(device)
        model.load_state_dict(pack['model'])
        if (not all(torch.isfinite(v).all() for v in model.state_dict().values())
                or not (model.scale>0).all()):
            raise ValueError('Invalid stop weights or normalization')
        loaded[arm]=model.eval().requires_grad_(False)
        metadata[arm]=dict(epoch=epoch,checkpoint_sha256=digest)
    return loaded,dict(arms=metadata,validity_only=True,performance_selection=False,
        scope='Fixed selected weights from the recorded paper study. No fitting, efficacy screen or new live validation occurs in this loader.')


class SharedStopOnly:
    """All arms compute both heads. Only regular-cadence local proposals differ.

    Caller retains original native/pending-native verification and public pose
    summaries. No method modifies evidence, actions, goals, or task scoring.
    """
    def __init__(self,models,arm):
        if arm not in ('original','bounded4','unbounded') or set(models)!= {'bounded4','unbounded'}:
            raise ValueError('Unknown shared-stop arm or incomplete model bank')
        self.arm=arm
        self.runtimes={name:temporal.TemporalStopRuntime(model) for name,model in models.items()}
        self.step=None;self.probabilities=None

    @torch.inference_mode()
    def update(self,step,feature_bank,original_probabilities):
        if original_probabilities.shape!=feature_bank.shape[:2]:
            raise ValueError('Pose probability shape mismatch')
        if not torch.isfinite(original_probabilities).all():
            raise ValueError('Invalid original pose estimate')
        if not torch.allclose(feature_bank[...,98].sigmoid(),original_probabilities,rtol=0,atol=1e-6):
            raise ValueError('Features and original probability are not from the same observation')
        predictions={name:runtime.update(step,feature_bank) for name,runtime in self.runtimes.items()}
        predictions['original']=original_probabilities.clone()
        self.step=step;self.probabilities=predictions
        return predictions[self.arm]

    def proposal(self,step,case,snapshot):
        if self.step!=step or step%10:
            raise ValueError('Local proposal requires current regular-cadence evidence')
        if not eligibility.shared_stop_eligible(snapshot):return False
        color=snapshot['goal']
        if type(color) is not int or not 0<=color<6:
            raise ValueError('Invalid publicly delivered goal')
        return bool(self.probabilities[self.arm][case,color]>=.7)
