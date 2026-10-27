"""Offline continuation control versus balanced on-policy teacher distillation.

Public replay of the exact paper continuation recipe from a verified exported
data bundle. This does not collect data, call an agent, change the scorer or deploy weights.
"""
import argparse
import copy
import fcntl
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/semantic_completion_v1'), str(ROOT/'source/BrainCo_DexHand')]
from common import STUDENT, load_student, save, sha
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch

from data import KEYS, SPLITS, load_data


def batch(data, indices, text, device):
    return VisualStudentBatch(data['rgb_history'][indices].to(device).float(),
        text[data['goal_id'][indices].to(device)], data['proprio_history'][indices].to(device).float())


@torch.inference_mode()
def validation(model, shards, text, device):
    model.eval()
    values = {}
    for split in ('old_validation','new_validation'):
        per_seed = {}
        for seed in SPLITS[split]:
            d = shards[seed]
            errors = []
            for ids in torch.arange(len(d['goal_id'])).split(64):
                pred = model(batch(d,ids,text,device))['action']
                errors.append(F.smooth_l1_loss(pred,d['teacher_action'][ids].to(device),
                    beta=.1,reduction='none').mean(-1).cpu())
            errors = torch.cat(errors)
            per_seed[str(seed)] = {'teacher_huber':float(errors.mean()),'n':len(errors)}
        values[split] = {'mean':sum(r['teacher_huber'] for r in per_seed.values())/len(per_seed),
                        'per_seed':per_seed}
    values['selection_loss'] = .5*(values['old_validation']['mean']+values['new_validation']['mean'])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    args = parser.parse_args()
    # Readiness fails before creating an output or acquiring GPU memory.
    shards,hashes,manifest = load_data(args.data_dir)
    parent_path = args.data_dir/'parent.pt'
    lock = Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(146001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source,text,original_pack = load_student('cuda:0')
    assert sha(STUDENT)==manifest['source_student_sha256']
    parent_pack = torch.load(parent_path,map_location='cpu',weights_only=True)
    initial = copy.deepcopy(source)
    initial.load_state_dict(parent_pack['model'],strict=True)
    models = {arm:copy.deepcopy(initial).requires_grad_(True) for arm in ('old_only','mixed')}
    optimizers = {}
    for arm,model in models.items():
        model.evidence_head.requires_grad_(False)
        model.stop_head.requires_grad_(False)
        optimizers[arm] = torch.optim.AdamW([
            {'params':list(model.action_head.parameters()),'lr':1e-4},
            {'params':[v for k,v in model.named_parameters() if not k.startswith('action_head.') and v.requires_grad], 'lr':1e-5}
        ],weight_decay=1e-4)
    train = {name:{key:torch.cat([shards[s][key] for s in SPLITS[name+'_train']]) for key in KEYS}
             for name in ('old','new')}
    steps = math.ceil(len(train['old']['goal_id'])/64)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(format='teacher_onpolicy_fit_v1',epochs=50,steps_per_epoch=steps,
        split_seeds=SPLITS,training_seed=146001,batch_size=64,common_old_half=32,
        mixed_new_fraction=.5,head_lr=1e-4,representation_lr=1e-5,weight_decay=1e-4,
        anchor_weight=.05,teacher_huber_beta=.1,clip_grad_norm=1.,tf32=False,
        selection='Equal weight to old and on-policy validation domains, equal weight to each seed within domain. Lowest held-out teacher Huber. No rollout success selection.',
        control='Both start from the same previous full checkpoint and receive equal optimizer updates. Old-only has 64 old examples; mixed has the same first 32 old examples and 32 on-policy examples. No failure weighting.',
        parent_checkpoint_sha256=sha(parent_path),source_student_sha256=sha(STUDENT),
        source_data_sha256=hashes,collection_result_sha256=manifest['collection_result_sha256'],
        data_manifest_sha256=sha(args.data_dir/'manifest.json'),
        source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        actor_inputs=list(KEYS[:3]),teacher_label_only=True,cloud_calls=0,deployed=False)
    save(args.output/'protocol.json',protocol)
    baseline = validation(initial,shards,text,'cuda:0')
    rng = torch.Generator().manual_seed(146001)
    best = {arm:{'loss':float('inf')} for arm in models}
    history = []
    for epoch in range(1,51):
        totals = {arm:0. for arm in models}
        for model in models.values():model.train()
        for _ in range(steps):
            common = torch.randint(len(train['old']['goal_id']),(32,),generator=rng)
            old_extra = torch.randint(len(train['old']['goal_id']),(32,),generator=rng)
            new_extra = torch.randint(len(train['new']['goal_id']),(32,),generator=rng)
            for arm,model in models.items():
                second = train['new'] if arm=='mixed' else train['old']
                extra = new_extra if arm=='mixed' else old_extra
                d = {key:torch.cat((train['old'][key][common],second[key][extra])) for key in KEYS}
                b = batch(d,torch.arange(64),text,'cuda:0')
                with torch.no_grad():anchor = source(b)['action']
                pred = model(b)['action']
                loss = F.smooth_l1_loss(pred,d['teacher_action'].to('cuda:0'),beta=.1)+.05*F.mse_loss(pred,anchor)
                assert torch.isfinite(loss)
                opt = optimizers[arm]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                totals[arm] += float(loss)/steps
        metrics = {arm:validation(model,shards,text,'cuda:0') for arm,model in models.items()}
        history.append(dict(epoch=epoch,train_loss=totals,validation=metrics))
        save(args.output/'history.json',history)
        for arm,model in models.items():
            if metrics[arm]['selection_loss']<best[arm]['loss']:
                best[arm] = dict(loss=metrics[arm]['selection_loss'],epoch=epoch,metrics=metrics[arm],
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        save(args.output/'progress.json',dict(status='training',epoch=epoch,metrics=metrics))
    results = {}
    for arm,value in best.items():
        pack = copy.deepcopy(original_pack)
        pack['model'] = value['state']
        for name,tensor in pack['model'].items():
            assert torch.isfinite(tensor).all()
            if name.startswith(('evidence_head.','stop_head.')):
                assert torch.equal(tensor,original_pack['model'][name].cpu())
        pack['onpolicy_distillation'] = dict(format='teacher_onpolicy_fit_v1',arm=arm,
            selected_epoch=value['epoch'],source_student_sha256=sha(STUDENT),
            parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'])
        path = args.output/f'{arm}.pt'
        torch.save(pack,path)
        results[arm] = dict(selected_epoch=value['epoch'],validation=value['metrics'],checkpoint_sha256=sha(path))
    for name,digest in hashes.items():assert sha(args.data_dir/name)==digest
    for name,digest in protocol['source_code_sha256'].items():assert sha(ROOT/name)==digest
    save(args.output/'result.json',dict(status='completed',baseline_validation=baseline,
        results=results,source_data_sha256=hashes,cloud_calls=0,deployed=False))
    save(args.output/'progress.json',dict(status='completed'))
    lock.close()


if __name__=='__main__':main()
