"""CPU checks for published evaluator code, metric and actor selection."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/paper_runtime'),str(ROOT/'scripts/semantic_completion_v1'),str(ROOT/'scripts/agentic'),str(ROOT/'source/BrainCo_DexHand')]
import torch
from actor_runtime import PublicActorRuntime,validate_run
from common import load_student
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--research-root',type=Path);args=parser.parse_args()
    manifest=json.loads((ROOT/'configs/paper_runtime_sources.json').read_text())
    tree=ast.parse((ROOT/'scripts/paper_runtime/evaluate.py').read_text())
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(v,ast.Attribute) and v.attr=='inference_mode' for v in ast.walk(n))]
    assert len(loops)==1 and hashlib.sha256(ast.dump(loops[0],include_attributes=False).encode()).hexdigest()==manifest['control_loop_ast_sha256']
    exact=0
    for rel,digest in manifest['source_sha256'].items():
        if args.research_root:assert sha(args.research_root/rel)==digest,rel
        path=ROOT/rel
        if path.exists():assert sha(path)==digest,rel;exact+=1
    torch.set_num_threads(2)
    reference,text,pack=load_student('cpu')
    assets=ROOT/'configs/runtime_assets.json'
    validate_run(assets,'mixed',offline=True)
    original=PublicActorRuntime(reference,pack,assets,'mixed','original')
    adapted=PublicActorRuntime(reference,pack,assets,'mixed','candidate')
    batch=VisualStudentBatch(torch.zeros(6,8,1060),text[:6],torch.zeros(6,8,128))
    masks=[torch.zeros(6,dtype=torch.bool),torch.ones(6,dtype=torch.bool),torch.tensor([True,False,True,False,True,False])]
    with torch.inference_mode():
        output=reference(batch);choices=adapted.actors.proposals(output,batch)
        for i,known in enumerate(masks):
            for j,active in enumerate(masks):
                step=10*(3*i+j)
                a=adapted.action(step,output,batch,known,active)
                b=original.action(step,output,batch,known,active)
                assert torch.equal(b,output['action'])
                assert torch.equal(a,torch.where((known&active)[:,None],choices['mixed'],output['action']))
        assert all(p.requires_grad is False for m in adapted.actors.models.values() for p in m.parameters())
    print(json.dumps(dict(status='passed',exact_source_files=exact,full_control_loop_ast_equal=True,
        checkpoint_metadata=adapted.actors.metadata,mask_pairs=9,network_requests=0,
        scope='CPU code/assets/proposal check. Simulation integration is a separate gate.')))


if __name__=='__main__':main()
