"""Export only exact training tensors from an available original research checkout.

This performs the original collection-readiness checks before exporting. It
never downloads data or credentials, runs a simulator or calls a service.
"""
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import torch
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);research=a.research_root.resolve();out=a.output.resolve()
    source=research/'scripts/teacher_onpolicy_followup_v2/train.py'
    spec=importlib.util.spec_from_file_location('_paper_original_trainer',source);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    shards,original_hashes=original.load_data()
    protocol=json.loads((original.NEW.parent/'fit/train/protocol.json').read_text())
    assert protocol['source_code_sha256'][str(source.relative_to(research))]==sha(source)
    assert protocol['source_data_sha256']==original_hashes and protocol['split_seeds']==original.SPLITS
    parent=original.PARENT/'train/full.pt';assert sha(parent)==protocol['parent_checkpoint_sha256']
    out.mkdir(parents=True,exist_ok=False);records={}
    for split,seeds in original.SPLITS.items():
        for seed in seeds:
            d=shards[seed];path=out/f'seed_{seed}.pt';torch.save(d,path)
            loaded=torch.load(path,map_location='cpu',weights_only=True)
            assert set(loaded)==set(original.KEYS) and all(torch.equal(loaded[k],d[k]) for k in original.KEYS)
            original_path=(original.OLD if split.startswith('old') else original.NEW/'data')/f'seed_{seed}/data.pt'
            records[str(seed)]=dict(split=split,n=len(d['goal_id']),source_sha256=original_hashes[str(original_path.relative_to(research))],export_sha256=sha(path))
    shutil.copyfile(parent,out/'parent.pt')
    manifest=dict(format='paper_action_data_v1',status='completed',split_seeds=original.SPLITS,shards=records,
        split_samples={split:sum(len(shards[s]['goal_id']) for s in seeds) for split,seeds in original.SPLITS.items()},
        source_student_sha256=protocol['source_student_sha256'],parent_checkpoint_sha256=sha(out/'parent.pt'),
        source_trainer_sha256=sha(source),source_protocol_sha256=sha(original.NEW.parent/'fit/train/protocol.json'),
        collection_result_sha256=sha(original.NEW/'result.json'),tensor_fields=list(original.KEYS),
        transform='Exact tensor subset only. No filtering, conversion, resampling, changed labels or split reassignment.',
        scope='Local reproducibility bundle for the fixed paper action fitting. No service traces, raw images, credentials or unrelated checkpoints.',cloud_calls=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='completed',shards=len(records),split_samples=manifest['split_samples'],manifest_sha256=sha(out/'manifest.json'))))
if __name__=='__main__':main()
