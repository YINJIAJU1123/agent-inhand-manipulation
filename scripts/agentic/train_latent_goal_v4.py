"""QLoRA + continuous goal head; split by episode and symbol-layout permutation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/BrainCo_DexHand'))
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch
from latent_goal_v4 import GoalHead, hidden_features, load_backbone, visual_inputs


def samples(directory):
    root = Path(directory)
    rows = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
    data = []
    for row in rows:
        for symbol in range(6):
            color = row['layout'].index(symbol)
            if not row.get('label_usable', [True] * 6)[color]:
                continue
            data.append({'root': root, 'row': row, 'color': color, 'visible': bool(row['readable'][color]),
                         'instruction': f"Turn the cube so the {row['symbols'][symbol]} sticker faces upward."})
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--val-examples', type=int, default=72)
    p.add_argument('--seed', type=int, default=71)
    p.add_argument('--student', default=str(ROOT / 'checkpoints/student/viserdex_rgbd_student.pt'))
    p.add_argument('--canonical', default=str(ROOT / 'checkpoints/student/siglip2_text_adapter.pt'))
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(4)
    dest = Path(a.output); dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'goal_head.pt').exists():
        raise RuntimeError('Refusing to overwrite a trained checkpoint')
    train, val = samples(a.train), samples(a.val)
    assert {x['row']['split'] for x in train} == {'train'}
    assert {x['row']['split'] for x in val} == {'val'}
    assert not ({x['row']['seed'] for x in train} & {x['row']['seed'] for x in val})
    assert not ({tuple(x['row']['layout']) for x in train} & {tuple(x['row']['layout']) for x in val})
    positives = [x for x in train if x['visible']]
    negatives = [x for x in train if not x['visible']]
    if not positives or not negatives:
        raise ValueError('Both readable targets and insufficient-evidence examples are required')
    positive_by_color = [[x for x in positives if x['color'] == color] for color in range(6)]
    if any(not x for x in positive_by_color):
        raise ValueError('Readable training examples required for every skill')
    rng = random.Random(a.seed)
    # Fixed validation subset; never chooses training examples.
    val = rng.sample(val, min(a.val_examples, len(val)))
    canonical = torch.load(a.canonical, map_location='cuda:0', weights_only=False)['canonical_embeddings'].float()
    ckpt = torch.load(a.student, map_location='cuda:0', weights_only=False)
    student = VisualLanguageStudent(ckpt['rgb_dim'], ckpt['language_dim'], ckpt['proprio_dim'], ckpt['action_dim'], memory_mode=ckpt['memory_mode']).cuda().eval()
    student.load_state_dict(ckpt['model']); student.requires_grad_(False)
    model, processor = load_backbone(a.model, training=True)
    base = model.get_base_model()
    hidden_size = base.config.text_config.hidden_size
    head = GoalHead(hidden_size, canonical).cuda()
    params = [x for x in model.parameters() if x.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    manifest = dict(model_path=str(Path(a.model).resolve()), steps=a.steps, accumulation=a.accumulation, seed=a.seed,
                    source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['train_latent_goal_v4.py','latent_goal_v4.py','agent_observation_v4.py']},
                    canonical_sha256=hashlib.sha256(Path(a.canonical).read_bytes()).hexdigest(),
                    student_sha256=hashlib.sha256(Path(a.student).read_bytes()).hexdigest(),
                    training_examples=len(train), positive_examples=len(positives), negative_examples=len(negatives),
                    validation_examples=len(val), trainable_vlm_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                    head_parameters=sum(x.numel() for x in head.parameters()), student_frozen=True,
                    latent_space=head.latent_space, positive_sampling='uniform target color then uniform example',
                    losses='readability BCE + visible-target CE + cosine alignment + frozen-student action consistency',
                    limits='First version aligns to six known skill prototypes. This is not open-vocabulary control or a learned observation policy.',
                    train_manifest=json.loads((Path(a.train) / 'manifest.json').read_text()),
                    val_manifest=json.loads((Path(a.val) / 'manifest.json').read_text()))
    (dest / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (dest / 'sources').mkdir(exist_ok=True)
    for name in manifest['source_sha256']:
        (dest / 'sources' / name).write_bytes((Path(__file__).parent / name).read_bytes())
    start = time.monotonic()
    model.train(); head.train()
    with (dest / 'training.jsonl').open('w') as log:
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(a.accumulation):
                sample = rng.choice(rng.choice(positive_by_color) if (step * a.accumulation + micro) % 2 == 0 else negatives)
                rgb = np.asarray(Image.open(sample['root'] / sample['row']['image']).convert('RGB'))
                inputs, _ = visual_inputs(processor, rgb, sample['instruction'], 'cuda:0')
                result = head(hidden_features(model, inputs))
                visible = torch.tensor([float(sample['visible'])], device='cuda:0')
                loss = F.binary_cross_entropy_with_logits(result['readable_logit'], visible)
                if sample['visible']:
                    target = torch.tensor([sample['color']], device='cuda:0')
                    z = result['goal_embedding']
                    loss = loss + F.cross_entropy(result['skill_logits'], target)
                    loss = loss + (1 - F.cosine_similarity(z, canonical[target])).mean()
                    with np.load(sample['root'] / sample['row']['history']) as history:
                        ih = torch.from_numpy(history['rgb_history']).cuda()[None]
                        ph = torch.from_numpy(history['proprio_history']).cuda()[None]
                    with torch.no_grad():
                        reference = student(VisualStudentBatch(ih, canonical[target], ph))['action']
                    # cuDNN eval-mode GRU cannot backpropagate into the input goal.
                    # Keep the student frozen/eval and use the differentiable native GRU.
                    with torch.backends.cudnn.flags(enabled=False):
                        predicted = student(VisualStudentBatch(ih, z, ph))['action']
                    loss = loss + 2 * F.mse_loss(predicted, reference)
                (loss / a.accumulation).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite training gradient')
            optimizer.step()
            row = {'step': step + 1, 'loss': float(np.mean(losses)), 'gradient_norm': float(norm),
                   'elapsed_s': round(time.monotonic() - start, 2)}
            log.write(json.dumps(row) + '\n'); log.flush()
            if step % 10 == 0:
                print(json.dumps(row), flush=True)
    model.eval(); head.eval()
    predictions = []
    with torch.inference_mode():
        for sample in val:
            rgb = np.asarray(Image.open(sample['root'] / sample['row']['image']).convert('RGB'))
            inputs, _ = visual_inputs(processor, rgb, sample['instruction'], 'cuda:0')
            result = head(hidden_features(model, inputs))
            predictions.append({'seed': sample['row']['seed'], 'case': sample['row']['case'], 'step': sample['row']['step'],
                                'instruction': sample['instruction'], 'visible': sample['visible'], 'target': sample['color'],
                                'predicted': int(result['skill_logits'][0].argmax()),
                                'readable_probability': float(result['readable_logit'].sigmoid()[0]),
                                'target_cosine': float(F.cosine_similarity(result['goal_embedding'], canonical[sample['color']][None])[0])})
    # Fixed threshold for this first version; no optimization on test rollouts.
    threshold = .5
    valid = [x for x in predictions if x['visible']]
    accepted = [x for x in predictions if x['readable_probability'] >= threshold]
    metrics = {'examples': len(predictions), 'readable_examples': len(valid),
               'visible_target_correct': sum(x['predicted'] == x['target'] for x in valid),
               'accepted': len(accepted), 'accepted_correct': sum(x['visible'] and x['predicted'] == x['target'] for x in accepted),
               'false_accepts_on_unreadable': sum(not x['visible'] for x in accepted),
               'readable_threshold': threshold}
    (dest / 'validation.json').write_text(json.dumps({'metrics': metrics, 'predictions': predictions}, indent=2) + '\n')
    nonzero_lora_b=sum(float(x.detach().abs().sum())>0 for n,x in model.named_parameters() if 'lora_B' in n)
    if nonzero_lora_b == 0:
        raise RuntimeError('LoRA parameters did not learn')
    manifest['nonzero_trained_lora_b_tensors']=nonzero_lora_b
    (dest / 'training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    model.save_pretrained(dest / 'lora')
    torch.save({'format': 'qwen_hidden_goal_v4', 'head': head.state_dict(), 'hidden_size': hidden_size, 'latent_space': head.latent_space,
                'model_path': str(Path(a.model).resolve()), 'readable_threshold': threshold,
                'manifest': manifest}, dest / 'goal_head.pt')
    print(json.dumps({'completed': True, 'validation': metrics, 'elapsed_s': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
