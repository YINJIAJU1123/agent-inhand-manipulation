"""Strict four-field training bundle. No privileged state enters the actor."""
import hashlib,json
from pathlib import Path
import torch
KEYS=('rgb_history','proprio_history','goal_id','teacher_action')
SPLITS={'old_train':list(range(144101,144115)),'old_validation':[144115,144116],
        'new_train':list(range(146101,146107)),'new_validation':[146107,146108]}
REFERENCE_SHA='28f62f3e1f8590cd57479a89aff5d36f054b6c49d19e544a7820d72d10f8d6f7'
PARENT_SHA='b000c4bc027ea3eb1a442c1b925f66002bddc98c1375116575693cb5bf249187'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def load_data(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    assert m['format']=='paper_action_data_v1' and m['status']=='completed' and m['split_seeds']==SPLITS
    assert m['source_student_sha256']==REFERENCE_SHA and m['parent_checkpoint_sha256']==sha(directory/'parent.pt')==PARENT_SHA
    assert len(m['shards'])==24 and set(m['shards'])=={str(s) for seeds in SPLITS.values() for s in seeds}
    result={};hashes={}
    for split,seeds in SPLITS.items():
        for seed in seeds:
            name=f'seed_{seed}.pt';p=directory/name;r=m['shards'][str(seed)]
            assert r['split']==split and sha(p)==r['export_sha256']
            d=torch.load(p,map_location='cpu',weights_only=True);assert set(d)==set(KEYS)
            n=len(d['goal_id']);assert n==r['n'] and n>0
            assert d['rgb_history'].shape==(n,8,1060) and d['proprio_history'].shape==(n,8,128) and d['teacher_action'].shape==(n,21)
            assert d['goal_id'].dtype==torch.long and ((d['goal_id']>=0)&(d['goal_id']<6)).all()
            assert all(torch.isfinite(d[k]).all() for k in KEYS) and d['teacher_action'].abs().max()<=1
            result[seed]=d;hashes[name]=r['export_sha256']
    assert {split:sum(len(result[s]['goal_id']) for s in seeds) for split,seeds in SPLITS.items()}==m['split_samples']
    return result,hashes,m
