"""CPU release check against the exact original trainer and exported data.

This checks one update and validation, not a full fit or task performance.
"""
import argparse,ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from data import load_data,SPLITS,KEYS,PARENT_SHA
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(146001)
    original=a.research_root/'scripts/teacher_onpolicy_followup_v2/train.py';published=HERE/'train_action.py'
    old,new=ast.parse(original.read_text()),ast.parse(published.read_text())
    def dump(n):return ast.dump(n,include_attributes=False)
    def function(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    for name in ('batch','validation'):assert dump(function(old,name))==dump(function(new,name))
    def epoch(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
    assert dump(epoch(old))==dump(epoch(new))
    def optimizers(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and n.body and 'model.evidence_head.requires_grad_(False)'==ast.unparse(n.body[0]))
    assert dump(optimizers(old))==dump(optimizers(new))
    spec=importlib.util.spec_from_file_location('_published_paper_training',published);trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    shards,hashes,manifest=load_data(a.data_dir)
    assert manifest['source_trainer_sha256']==sha(original)
    source,text,pack=trainer.load_student('cpu');assert sha(trainer.STUDENT)==manifest['source_student_sha256']
    parent=copy.deepcopy(source);parent.load_state_dict(torch.load(a.data_dir/'parent.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    assert sha(a.data_dir/'parent.pt')==PARENT_SHA
    actual=trainer.validation(parent,shards,text,'cpu')
    recorded=json.loads((a.research_root/'outputs/teacher_onpolicy_followup_v2/continuation_20261025_r2/fit/train/result.json').read_text())['baseline_validation']
    max_error=abs(actual['selection_loss']-recorded['selection_loss'])
    for split in ('old_validation','new_validation'):
        for seed in SPLITS[split]:
            av,bv=actual[split]['per_seed'][str(seed)],recorded[split]['per_seed'][str(seed)]
            assert av['n']==bv['n'];max_error=max(max_error,abs(av['teacher_huber']-bv['teacher_huber']))
    assert max_error<1e-6
    models={arm:copy.deepcopy(parent).requires_grad_(True) for arm in ('old_only','mixed')}
    namespace=dict(models=models,optimizers={},torch=torch,F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[optimizers(new)],type_ignores=[])),'published_optimizer_setup','exec'),namespace)
    counts={arm:sum(p.numel() for p in m.parameters() if p.requires_grad) for arm,m in models.items()};assert set(counts.values())=={943685}
    frozen={arm:{k:v.detach().clone() for k,v in m.state_dict().items() if k.startswith(('evidence_head.','stop_head.'))} for arm,m in models.items()}
    before={arm:m.action_head[0].weight.detach().clone() for arm,m in models.items()}
    inputs={name:{key:shards[SPLITS[name+'_train'][0]][key][:64] for key in KEYS} for name in ('old','new')}
    namespace.update(train=inputs,KEYS=KEYS,common=torch.arange(32),old_extra=torch.arange(32,64),new_extra=torch.arange(32,64),text=text,source=source,batch=trainer.batch,totals={arm:0. for arm in models},steps=182)
    step=next(n for n in ast.walk(epoch(new)) if isinstance(n,ast.For) and ast.unparse(n.target)=='(arm, model)' and 'second = ' in ast.unparse(n.body[0]))
    class CPU(ast.NodeTransformer):
        def visit_Constant(self,node):return ast.copy_location(ast.Constant('cpu'),node) if node.value=='cuda:0' else node
    step=CPU().visit(copy.deepcopy(step))
    for m in models.values():m.train()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[step],type_ignores=[])),'published_one_cpu_update','exec'),namespace)
    for arm,m in models.items():
        assert not torch.equal(before[arm],m.action_head[0].weight)
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in frozen[arm].items())
        assert all(torch.isfinite(p).all() for p in m.parameters())
    report=dict(status='passed',cloud_calls=0,gpu_used=False,full_fit_repeated=False,
        source_sha256=sha(original),published_trainer_sha256=sha(published),data_loader_sha256=sha(HERE/'data.py'),exporter_sha256=sha(HERE/'export_data.py'),checker_sha256=sha(Path(__file__)),
        epoch_loop_ast_equal=True,batch_and_validation_ast_equal=True,optimizer_setup_ast_equal=True,
        bundle_manifest_sha256=sha(a.data_dir/'manifest.json'),shards=len(shards),split_samples=manifest['split_samples'],tensor_export_exact=True,
        parent_checkpoint_sha256=PARENT_SHA,reference_checkpoint_sha256=sha(trainer.STUDENT),
        cpu_validation=actual,maximum_saved_validation_difference=max_error,trainable_parameters=counts,
        one_cpu_update_per_arm=True,frozen_head_parameters_unchanged=True,
        scope='Exact exported data, core-code parity, held-out teacher-loss replay and one CPU optimizer update. Not a complete training reproduction, byte-identical fitted checkpoint claim, new simulation or performance estimate.')
    (ROOT/'docs/paper_training_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','shards','split_samples','maximum_saved_validation_difference','trainable_parameters','one_cpu_update_per_arm')}))
if __name__=='__main__':main()
