"""Check fixed stop assets, source parity and recorded CPU history predictions."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/paper_stop'),str(ROOT/'scripts/paper_runtime')]
from stop_runtime import SharedStopOnly,load_published_heads

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ast_hash(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def class_node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
def loop(path):
    nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.With)
           and any(isinstance(v,ast.Attribute) and v.attr=='inference_mode' for v in ast.walk(n))]
    assert len(nodes)==1
    return nodes[0]
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--research-root',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    manifest=json.loads((ROOT/'configs/paper_stop_sources.json').read_text())
    for rel,digest in manifest['published_sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert ast_hash(loop(ROOT/'scripts/paper_stop/evaluate.py'))==manifest['control_loop_ast_sha256']
    for item in manifest['classes']:
        assert ast_hash(class_node(ROOT/item['published'],item['name']))==item['ast_sha256']
        if args.research_root:
            assert sha(args.research_root/item['research'])==item['source_sha256']
            assert ast_hash(class_node(args.research_root/item['research'],item['name']))==item['ast_sha256']
    models,meta=load_published_heads(ROOT/'configs/paper_stop_assets.json')
    assert all(sum(p.numel() for p in m.parameters())==39425 and not any(p.requires_grad for p in m.parameters()) for m in models.values())
    runtimes={a:SharedStopOnly(models,a) for a in ('original','bounded4','unbounded')}
    generator=torch.Generator().manual_seed(20261027)
    bank=torch.randn(2,6,125,generator=generator);p0=bank[...,98].sigmoid()
    for step in (0,10,30):
        bank_before=bank.clone();p_before=p0.clone()
        for arm,runtime in runtimes.items():
            runtime.update(step,bank,p0)
            assert torch.equal(runtime.probabilities['original'],p0)
            assert torch.equal(bank,bank_before) and torch.equal(p0,p_before)
            for color in range(6):
                snapshot=dict(goal=color,goal_symbol='star',decision=dict(state='execute',color=color))
                assert runtime.proposal(step,0,snapshot)==bool(runtime.probabilities[arm][0,color]>=.7)
                for invalid in ('inspect','stop','abstain'):
                    snapshot['decision']['state']=invalid;assert not runtime.proposal(step,0,snapshot)
                snapshot['decision']['state']='verify';snapshot['goal_symbol']=None
                assert not runtime.proposal(step,0,snapshot)
            if step==30:
                for history in runtime.runtimes.values():
                    assert torch.equal(history.history,bank[:,:,None,:].repeat(1,1,3,1))
    rejected=[]
    for name,fn in [
        ('off_cadence',lambda:SharedStopOnly(models,'original').update(1,bank,p0)),
        ('nonfinite',lambda:SharedStopOnly(models,'original').update(0,bank,torch.full_like(p0,float('nan')))),
        ('unsynchronized',lambda:SharedStopOnly(models,'original').update(0,bank,1-p0)),
        ('stale_proposal',lambda:runtimes['original'].proposal(40,0,snapshot)),
        ('unknown_arm',lambda:SharedStopOnly(models,'best'))]:
        try:fn()
        except (ValueError,RuntimeError):rejected.append(name)
        else:raise AssertionError(name)
    recorded=None
    if args.research_root:
        R=args.research_root
        src=R/'scripts/stop_onpolicy_live_v1/evaluate.py'
        assert sha(src)==manifest['research_evaluator_sha256']
        assert ast_hash(loop(src))==manifest['control_loop_ast_sha256']
        original=load_module('_published_stop_research_comparison',R/'scripts/stop_onpolicy_live_v1/stop_runtime.py')
        research_models,_=original.load_audited_heads(R/'outputs/stop_onpolicy_learning_v1/development_20261026')
        rel='outputs/stop_onpolicy_collection_v1/development_20261026/data/train_153101/data.pt'
        expected=json.loads((R/'outputs/stop_onpolicy_learning_v1/development_20261026/result.json').read_text())['data_sha256'][rel]
        assert sha(R/rel)==expected
        data=torch.load(R/rel,map_location='cpu',weights_only=True)
        differences=[];predictions=0;decisions=0
        for case in range(6):
            pub={a:SharedStopOnly(models,a) for a in runtimes}
            old={a:original.SharedStopOnly(research_models,a) for a in runtimes}
            ids=sorted([i for i,r in enumerate(data['rows']) if r['case']==case],key=lambda i:data['rows'][i]['step'])
            for i in ids:
                row=data['rows'][i];step=row['step'];bank=data['features'][i:i+1];p0=bank[...,98].sigmoid()
                for arm in pub:
                    got=pub[arm].update(step,bank,p0);expected_prob=old[arm].update(step,bank,p0)
                    differences.append(float((got-expected_prob).abs().max()));predictions+=got.numel()
                    assert torch.equal(got,expected_prob)
                    for color in range(6):
                        snapshot=dict(goal=color,goal_symbol='star',decision=dict(state='execute',color=color))
                        assert pub[arm].proposal(step,0,snapshot)==old[arm].proposal(step,0,snapshot)
                        decisions+=1
        recorded=dict(source_sha256=expected,source=rel,rows=len(data['rows']),cases=6,
                      probability_values=predictions,decisions=decisions,max_absolute_probability_difference=max(differences),
                      scope='Previously collected training features only. No current experiment audit data or task-success scores.')
    result=dict(status='passed',cloud_calls=0,simulation_launched=False,training_launched=False,
                control_loop_ast_equal=True,class_ast_equal=len(manifest['classes']),
                asset_admission=meta,rejected_invalid_inputs=rejected,recorded_parity=recorded,
                scope='CPU source and inference verification. Full simulator integration is a separate gate.')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
