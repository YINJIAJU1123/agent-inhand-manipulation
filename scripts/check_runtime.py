"""Verify published weights and run a CPU forward pass without model services."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'/s) for s in ('agentic','semantic_completion_v1','semantic_completion_v2','semantic_completion_v3','astra_retention_online_v1')]+[str(ROOT/'source/BrainCo_DexHand')]
import torch
from common import load_student,CONTROLLER
from recovery_controller_v3 import RecoveryController
from evidence_runtime import CachedDenseEvidence
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch

def main():
    manifest=json.loads((ROOT/'configs/runtime_assets.json').read_text())
    for name,asset in manifest['assets'].items():
        path=ROOT/asset['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=asset['sha256']:raise RuntimeError('Checksum mismatch: '+name)
    student,text,pack=load_student('cpu');n=2
    rgb=torch.zeros(n,pack['history'],pack['rgb_dim']);prop=torch.zeros(n,pack['history'],pack['proprio_dim'])
    with torch.inference_mode():
        action=student(VisualStudentBatch(rgb,text[:n],prop))['action']
        assert action.shape==(n,21) and torch.isfinite(action).all() and action.abs().max()<=1
        controller=RecoveryController(616);controller.load_state_dict(torch.load(CONTROLLER,map_location='cpu',weights_only=True)['model'])
        evidence=CachedDenseEvidence(ROOT/'checkpoints/runtime/pose_verifier.pt','cpu',n)
        image=torch.zeros(n,512,512,3,dtype=torch.uint8);depth=torch.ones(n,512,512,1)
        evidence.update(image,rgb,prop,depth)
        assert evidence.retention_features.shape==(n,6,125)
        sys.path.insert(0,str(ROOT/'scripts/astra_semantic_memory_live_v1'))
        from online_memory import OnlineMemory
        memory=OnlineMemory('cpu',n);memory.update(image,0);memory.summary(0,0)
    print(json.dumps(dict(status='passed',verified_assets=len(manifest['assets']),student_action_shape=list(action.shape),rgb_feature_dim=pack['rgb_dim'],goal_dim=pack['language_dim'],proprio_dim=pack['proprio_dim'],history=pack['history'],action_scale=pack['action_scale'],network_requests=0)))
if __name__=='__main__':main()
