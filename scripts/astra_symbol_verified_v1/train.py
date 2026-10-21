"""Matched data-only and crop-verification experiments on full trajectories."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import StickerReader, base, crop_regions, losses, crop_losses

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics_module = load_module('region_metrics', ROOT/'scripts/astra_symbol_regions_v1/train.py')
augment = load_module('glyph_augmentation', ROOT/'scripts/astra_symbol_invariant_v1/train.py').augment
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, nargs='+', required=True, help='Explicit completed rank/data directories; excludes smoke.')
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['invariant', 'verified'], required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(10623)
    paths = {s: sorted(x for d in a.data for x in d.glob(s+'_*/data.pt')) for s in ('train', 'calibration', 'audit')}
    if any(not value for value in paths.values()):
        raise ValueError('All independent data splits are required')
    seeds = {s: set() for s in paths}
    manifests = {}
    for split, files in paths.items():
        for file in files:
            m = json.loads(file.with_name('manifest.json').read_text())
            if m['split'] != split or m['seed'] in seeds[split] or m['replay_max_error'] > 1e-6:
                raise ValueError('Repeated seed, invalid split or unverified replay: '+str(file))
            seeds[split].add(m['seed'])
            if sha(file) != m['data_sha256']:
                raise ValueError('Data checksum mismatch: '+str(file))
            manifests[str(file)] = m['data_sha256']
    assert not(seeds['train'] & seeds['calibration'] or seeds['train'] & seeds['audit'] or seeds['calibration'] & seeds['audit'])
    train = metrics_module.Views(paths['train'])
    cal = metrics_module.Views(paths['calibration'])
    known = (train.labels >= 0) & (train.labels < 6)
    counts = torch.bincount(train.labels[known], minlength=6).float()
    if not (counts > 0).all():
        raise ValueError('Missing glyph class')
    weights = ((1/counts)[train.labels.clamp(0,5)]*known).sum(-1)+.05/counts.max()
    sampler = WeightedRandomSampler(weights, 2048, replacement=True, generator=torch.Generator().manual_seed(10623))
    loader = DataLoader(train, batch_size=12, sampler=sampler)
    cal_loader = DataLoader(cal, batch_size=12)
    model = (StickerReader() if a.variant == 'verified' else base.StickerReader()).to(a.device)
    initial = torch.load(a.base, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(initial['model'], strict=False)
    if unexpected or any(not key.startswith('crop_color.') for key in missing):
        raise ValueError('Unexpected initializer architecture')
    model_source = Path(__file__).with_name('model.py') if a.variant == 'verified' else ROOT/'scripts/astra_symbol_invariant_v1/model.py'
    protocol = dict(variant=a.variant, base_sha256=sha(a.base), data_sha256=manifests,
        seeds={s:sorted(x) for s,x in seeds.items()}, train_views=len(train), calibration_views=len(cal),
        epochs=a.epochs, source_sha256={'model.py':sha(model_source), 'train.py':sha(__file__),
            'losses.py':sha(Path(__file__).with_name('model.py'))}, base_model_source_sha256=sha(ROOT/'scripts/astra_symbol_regions_v1/model.py'),
        inherited_invariant_source_sha256=sha(ROOT/'scripts/astra_symbol_invariant_v1/model.py'), cloud_calls=0,
        input='Only measured native RGB enters forward. Scene geometry and layout only supervise offline training.',
        selection='Calibration macro readable accuracy selects checkpoint. Calibration chooses threshold with >=50 accepted and >=95% precision. Audit runs once after selection.',
        comparison='Identical trajectory data, initializer, sampling, full-network training and matched-crop glyph losses. Verified additionally learns crop color and rejects query-color mismatch.',
        scope='New supplied-color simulation seeds; repeated recorded Astra videos remain development diagnostics, not final task evaluation.')
    save(out/'protocol.json', protocol)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
    best, history, start = -1., [], time.time()
    torch.manual_seed(10623)
    try:
        for epoch in range(1, a.epochs+1):
            model.train()
            totals = []
            for rgb, boxes, labels in loader:
                rgb = rgb.to(a.device).float()/255.
                boxes, labels = boxes.to(a.device).clone(), labels.to(a.device)
                readable = (labels >= 0) & (labels < 6)
                if not readable.any():
                    continue
                flip = torch.rand(len(rgb), device=a.device) < .5
                rgb[flip] = rgb[flip].flip(-1)
                boxes[flip, :, 0] = 1-boxes[flip, :, 0]
                rgb = (rgb*(.85+.3*torch.rand(len(rgb),1,1,1,device=a.device))).clamp(0,1)
                output = model(rgb)
                loss, _ = losses(model, rgb, boxes, labels, output, predicted_weight=0.)
                matched_glyph, color = crop_losses(model, output, rgb, boxes, labels)
                gt = crop_regions(rgb, boxes).reshape(len(rgb),6,3,64,64)[readable]
                augmented = F.cross_entropy(model.glyph_logits(augment(gt)), labels[readable], label_smoothing=.03)
                loss = loss+.3*matched_glyph+.5*augmented+.5*color
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite learning loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
                totals.append(float(loss.detach()))
            pr, lab, ov = metrics_module.predict(model, cal_loader, a.device)
            m = metrics_module.metrics(pr, lab, ov, .7)
            row = dict(epoch=epoch, loss=sum(totals)/len(totals), calibration=m, elapsed_s=time.time()-start)
            history.append(row)
            save(out/'progress.json', dict(status='training', **row))
            save(out/'history.json', history)
            print('VERIFIED_SYMBOL_EPOCH '+json.dumps(row), flush=True)
            if m['macro_readable'] > best:
                best = m['macro_readable']
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    epoch=epoch, cloud_eligible=False, input='native_rgb_only', architecture=a.variant), out/'head.pt')
        pack = torch.load(out/'head.pt', map_location=a.device, weights_only=True)
        model.load_state_dict(pack['model'])
        pr, lab, ov = metrics_module.predict(model, cal_loader, a.device)
        curve = [metrics_module.metrics(pr,lab,ov,t) for t in [.5,.6,.7,.8,.85,.9,.95,.98,.99]]
        eligible = [m for m in curve if m['accepted'] >= 50 and (m['precision'] or 0) >= .95]
        selected = max(eligible, key=lambda m:m['accepted']) if eligible else None
        threshold = selected['threshold'] if selected else 1.1
        audit = metrics_module.Views(paths['audit'])
        pr, lab, ov = metrics_module.predict(model, DataLoader(audit,batch_size=12), a.device)
        measured = metrics_module.metrics(pr,lab,ov,threshold)
        passed = bool(selected and measured['accepted'] >= 50 and (measured['precision'] or 0) >= .95
                      and measured['macro_readable'] >= .75 and measured['readable_mean_box_iou'] >= .6)
        pack.update(threshold=threshold, offline_gate_passed=passed)
        pack['model'] = {k:v.cpu() for k,v in pack['model'].items()}
        torch.save(pack, out/'head.pt')
        result = dict(status='completed', variant=a.variant, best_epoch=pack['epoch'], calibration=selected,
            calibration_curve=curve, audit=measured, offline_gate_passed=passed,
            checkpoint_sha256=sha(out/'head.pt'), cloud_eligible=False, cloud_calls=0,
            note='Requires recorded-request shadow, fresh autonomous closed-loop validation and separately bounded real Astra evaluation.')
        save(out/'result.json', result)
        save(out/'progress.json', dict(status='completed', offline_gate_passed=passed, elapsed_s=time.time()-start))
    except BaseException:
        save(out/'failure.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
