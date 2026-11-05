"""RSLCube RGB/instruction -> continuous student latent, with learned readability.

The six fixed faces are a migration development task. This experiment cannot
establish open-vocabulary grounding or generalization to unseen face layouts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from goals import load_bank
from semantic import visual_inputs

AGENT_DIR = Path(__file__).resolve().parents[1] / 'agentic'
sys.path.insert(0, str(AGENT_DIR))
from latent_goal_v4 import GoalHead, hidden_features, load_backbone


def read_dataset(path, split):
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['format'] != 'rslcube_letter_visibility_proxy_v1' or manifest['split'] != split:
        raise ValueError('Unexpected grounding dataset contract')
    rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
    if any(row['split'] != split for row in rows):
        raise ValueError('Mixed data splits')
    images = h5py.File(Path(manifest['dataset']) / 'samples.h5', 'r')
    groups = [[row for row in rows if row['goal_index'] == goal and row['readable'] == visible]
              for goal in range(6) for visible in (False, True)]
    if any(not group for group in groups):
        raise ValueError('Each face requires both visible and invisible examples')
    return manifest, rows, images, groups


def score(head, features, temperature):
    result = head(features)
    result['skill_logits'] = -(result['goal_embedding'][:, None] - head.canonical[None]).square().sum(-1) / temperature
    return result


class RSLCubeLatentPredictor:
    def __init__(self, checkpoint, model_path=None):
        root = Path(checkpoint)
        pack = torch.load(root / 'goal_head.pt', map_location='cpu', weights_only=False)
        if pack['format'] != 'rslcube_pose_latent_grounding_v1':
            raise ValueError('Not an RSLCube latent checkpoint')
        self.model, self.processor = load_backbone(str(model_path or pack['model_path']), root / 'lora')
        self.model.eval()
        self.head = GoalHead(pack['hidden_size'], pack['head']['canonical'], pack['latent_space']).cuda().eval()
        self.head.load_state_dict(pack['head'])
        self.threshold = pack['readable_threshold']
        self.temperature = pack['skill_temperature']
        self.goal_manifest = pack['goal_manifest']
        self.checkpoint_sha256 = hashlib.sha256((root / 'goal_head.pt').read_bytes()).hexdigest()
        self.adapter_sha256 = hashlib.sha256((root / 'lora' / 'adapter_model.safetensors').read_bytes()).hexdigest()

    @torch.inference_mode()
    def predict(self, rgb, instruction):
        inputs, box = visual_inputs(self.processor, rgb, instruction, 'cuda:0')
        result = score(self.head, hidden_features(self.model, inputs), self.temperature)
        probability = float(result['readable_logit'].sigmoid()[0])
        return dict(goal_embedding=result['goal_embedding'][0].cpu().numpy(),
                    skill=int(result['skill_logits'][0].argmax()),
                    readable_probability=probability, accepted=probability >= self.threshold,
                    crop_box=box, image_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest())



def evaluate_grounding(model, processor, head, validation, image_sources, temperature):
    """Measure identical held-out views before and after an adaptation run."""
    model.eval()
    head.eval()
    predictions = []
    with torch.inference_mode():
        for row in validation:
            rgb = np.asarray(image_sources[row['dataset_index']]['rgb'][row['frame']])
            inputs, box = visual_inputs(processor, rgb, row['instruction'], 'cuda:0')
            result = score(head, hidden_features(model, inputs), temperature)
            predictions.append(dict(frame=row['frame'], goal=row['goal_index'], readable=row['readable'],
                dataset_index=row['dataset_index'], seed=row['seed'], env_id=row['env_id'], episode_id=row['episode_id'],
                predicted=int(result['skill_logits'][0].argmax()),
                readable_probability=float(result['readable_logit'].sigmoid()[0]),
                target_cosine=float(F.cosine_similarity(result['goal_embedding'],
                    head.canonical[row['goal_index']][None])[0]), crop_box=box))
    def metrics(rows):
        visible = [row for row in rows if row['readable']]
        accepted = [row for row in rows if row['readable_probability'] >= .5]
        return dict(examples=len(rows), visible_examples=len(visible),
            visible_goal_correct=sum(row['predicted'] == row['goal'] for row in visible),
            accepted=len(accepted),
            accepted_correct=sum(row['readable'] and row['predicted'] == row['goal'] for row in accepted),
            false_accepts_on_invisible=sum(not row['readable'] for row in accepted), threshold=.5,
            scope='weak-label offline validation; not episode success')
    return dict(metrics=metrics(predictions), metrics_by_dataset={str(index):metrics(
        [row for row in predictions if row['dataset_index']==index]) for index in range(len(image_sources))},
        predictions=predictions)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--warm_start', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--replay_train', type=Path)
    parser.add_argument('--replay_validation', type=Path)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--text_cache', type=Path, required=True)
    parser.add_argument('--original_asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--validation_per_stratum', type=int, default=10)
    parser.add_argument('--seed', type=int, default=182821)
    parser.add_argument('--skill_temperature', type=float, default=.02)
    args = parser.parse_args()
    if bool(args.replay_train) != bool(args.replay_validation):
        parser.error('Replay training and validation must be supplied together')
    if args.steps<=0 or args.learning_rate<=0:
        parser.error('Training steps and learning rate must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    train_manifest, train, train_images, groups = read_dataset(args.train, 'train')
    val_manifest, val, val_images, val_groups = read_dataset(args.validation, 'validation')
    train_sets = [(train_manifest,train,train_images,groups)]
    val_sets = [(val_manifest,val,val_images,val_groups)]
    if args.replay_train:
        train_sets.append(read_dataset(args.replay_train,'train'))
        val_sets.append(read_dataset(args.replay_validation,'validation'))
    if {row['seed'] for _,rows,_,_ in train_sets for row in rows} & {row['seed'] for _,rows,_,_ in val_sets for row in rows}:
        raise ValueError('Capture seeds overlap')
    val_rng = random.Random(args.seed + 1)
    validation = [dict(row,dataset_index=index) for index,(_,_,_,strata) in enumerate(val_sets)
        for group in strata for row in val_rng.sample(group,min(args.validation_per_stratum,len(group)))]
    canonical, _, goal_manifest = load_bank(args.text_cache, args.original_asset, 'cuda:0')
    old = torch.load(args.warm_start / 'goal_head.pt', map_location='cpu', weights_only=False)
    model, processor = load_backbone(str(args.model), args.warm_start / 'lora', training=True)
    hidden_size = model.get_base_model().config.text_config.hidden_size
    resume_grounding = old['format']=='rslcube_pose_latent_grounding_v1'
    head = GoalHead(hidden_size, canonical,old['latent_space'] if resume_grounding else 'bounded_skill_span').cuda()
    if resume_grounding:
        for key in ('asset_sha256','cache_sha256'):
            if old['goal_manifest'][key]!=goal_manifest[key]:
                raise ValueError('Cannot continue a different grounding latent bank')
        torch.testing.assert_close(old['head']['canonical'],canonical.cpu(),atol=0,rtol=0)
        head.load_state_dict(old['head'])
    else:
        # An older backend has a different latent basis. Only its shared trunk transfers.
        transferred = {key: value for key, value in old['head'].items()
                       if key.startswith(('trunk.', 'readable.'))}
        head.load_state_dict(transferred, strict=False)
    params = [param for param in model.parameters() if param.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=.01)
    source_paths = [Path(__file__), Path(__file__).with_name('semantic.py'),
                    Path(__file__).with_name('goals.py'), AGENT_DIR / 'latent_goal_v4.py',
                    AGENT_DIR / 'agent_observation_v4.py']
    manifest = dict(format='rslcube_pose_latent_grounding_v1', arguments={key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}, goal_manifest=goal_manifest,
                    train_manifest=train_manifest, validation_manifest=val_manifest,
                    replay_train_manifest=train_sets[1][0] if args.replay_train else None,
                    replay_validation_manifest=val_sets[1][0] if args.replay_validation else None,
                    train_examples=sum(len(rows) for _,rows,_,_ in train_sets), validation_examples=len(validation),
                    trainable_backbone_parameters=sum(param.numel() for param in model.parameters() if param.requires_grad),
                    trainable_head_parameters=sum(param.numel() for param in head.parameters()),
                    head_transfer='full same-bank head continuation' if resume_grounding else 'trunk and readability only; new latent basis/projection',
                    warm_start_sha256=hashlib.sha256((args.warm_start / 'goal_head.pt').read_bytes()).hexdigest(),
                    adapter_init_sha256=hashlib.sha256((args.warm_start / 'lora/adapter_model.safetensors').read_bytes()).hexdigest(),
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                    sampling='alternate dataset sources per optimizer step; uniform color and balanced visibility strata',
                    loss='readability BCE + positive goal CE + cosine alignment',
                    limitations=['fixed letter/color layout', 'geometric visibility proxy, not human ground truth',
                                 'no student action consistency in this interface warmup',
                                 'no formal agent evaluation or autonomy claim'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'sources').mkdir()
    for path in source_paths:
        (args.output / 'sources' / path.name).write_bytes(path.read_bytes())
    (args.output / 'validation_selection.json').write_text(json.dumps(validation, indent=2) + '\n')
    start = time.monotonic()
    if resume_grounding:
        before=evaluate_grounding(model,processor,head,validation,[item[2] for item in val_sets],args.skill_temperature)
        (args.output/'before_validation.json').write_text(json.dumps(before,indent=2)+'\n')
    model.train()
    head.train()
    with (args.output / 'training.jsonl').open('w') as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(args.accumulation):
                visible = (micro + (step - 1) * args.accumulation) % 2 == 0
                _,_,images,strata = train_sets[(step-1)%len(train_sets)]
                group = strata[2 * rng.randrange(6) + int(visible)]
                row = rng.choice(group)
                rgb = np.asarray(images['rgb'][row['frame']])
                inputs, _ = visual_inputs(processor, rgb, row['instruction'], 'cuda:0')
                result = score(head, hidden_features(model, inputs), args.skill_temperature)
                loss = F.binary_cross_entropy_with_logits(result['readable_logit'],
                    torch.tensor([float(visible)], device='cuda:0'))
                if visible:
                    target = torch.tensor([row['goal_index']], device='cuda:0')
                    loss = loss + F.cross_entropy(result['skill_logits'], target)
                    loss = loss + (1 - F.cosine_similarity(result['goal_embedding'], canonical[target])).mean()
                (loss / args.accumulation).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient at step {step}')
            optimizer.step()
            progress = dict(step=step, target_steps=args.steps, mean_loss=float(np.mean(losses)),
                            gradient_norm=float(norm), elapsed_seconds=time.monotonic() - start)
            log.write(json.dumps(progress) + '\n')
            log.flush()
            (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            if step == 1 or step % 10 == 0:
                print(json.dumps(progress), flush=True)
            if step % 60 == 0 or step == args.steps:
                checkpoint = args.output / f'step_{step:05d}'
                checkpoint.mkdir()
                model.save_pretrained(checkpoint / 'lora')
                torch.save(dict(format='rslcube_pose_latent_grounding_v1', head=head.state_dict(),
                    hidden_size=hidden_size, latent_space=head.latent_space,
                    model_path=str(args.model.resolve()), goal_manifest=goal_manifest,
                    skill_temperature=args.skill_temperature, readable_threshold=.5,
                    training_step=step, manifest=manifest), checkpoint / 'goal_head.pt')
    report=evaluate_grounding(model,processor,head,validation,[item[2] for item in val_sets],args.skill_temperature)
    metrics=report['metrics']
    (args.output / 'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output / 'completion.json').write_text(json.dumps(dict(completed=True,steps=args.steps,
        final_checkpoint=str(checkpoint.resolve()),validation=metrics,
        validation_by_dataset=report['metrics_by_dataset'],elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
    for _,_,images,_ in train_sets+val_sets:
        images.close()
    print(json.dumps(dict(completed=True, validation=metrics)), flush=True)


if __name__ == '__main__':
    main()
