"""Run frozen native-RGB predictions before loading scoring labels or Astra responses."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
COLORS = ('red', 'green', 'blue', 'yellow', 'magenta', 'cyan')
SYMBOLS = ('star', 'circle', 'triangle', 'cross', 'diamond', 'bars')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def summarize(rows):
    accepted = [color for row in rows for color in row['colors'] if color['accepted']]
    targets = [row for row in rows if len(row['target_candidates']) == 1]
    semantic = [row for row in rows if row['astra_semantic_tool']]
    wrong = [row for row in semantic if not row['astra_binding_correct']]
    return dict(request_frames=len(rows), accepted_bindings=len(accepted), correct_bindings=sum(c['correct'] for c in accepted),
                precision=sum(c['correct'] for c in accepted) / len(accepted) if accepted else None,
                unique_target_frames=len(targets), correct_unique_target_frames=sum(row['unique_correct_target'] for row in targets),
                astra_semantic_frames=len(semantic), astra_incorrect_binding_frames=len(wrong),
                contradicted_wrong_binding_frames=sum(row['contradicts_astra'] for row in wrong),
                unique_correct_target_on_wrong_binding_frames=sum(row['unique_correct_target'] for row in wrong),
                contradicted_correct_binding_frames=sum(row['contradicts_astra'] and row['astra_binding_correct'] for row in semantic))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-source', type=Path, default=ROOT / 'scripts/astra_symbol_regions_v1/model.py')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    if sha(args.cache / 'inputs.pt') != manifest['inputs_sha256']:
        raise ValueError('Request RGB checksum mismatch')
    model_path = args.model_source.resolve()
    training = json.loads((args.checkpoint.parent / 'protocol.json').read_text())
    if sha(model_path) != training['source_sha256']['model.py']:
        raise ValueError('Model source differs from trained architecture')
    if training.get('base_model_source_sha256') and sha(ROOT / 'scripts/astra_symbol_regions_v1/model.py') != training['base_model_source_sha256']:
        raise ValueError('Inherited model source differs from training')
    if training.get('inherited_invariant_source_sha256') and sha(ROOT / 'scripts/astra_symbol_invariant_v1/model.py') != training['inherited_invariant_source_sha256']:
        raise ValueError('Inherited glyph model source differs from training')
    result = json.loads((args.checkpoint.parent / 'result.json').read_text())
    if sha(args.checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    spec = importlib.util.spec_from_file_location('native_rgb_reader', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = module.StickerReader().eval()
    model.load_state_dict(pack['model'])
    threshold = pack['threshold']
    data = torch.load(args.cache / 'inputs.pt', map_location='cpu', weights_only=True)
    if set(data) != {'rgb'} or data['rgb'].dtype != torch.uint8 or data['rgb'].shape[1:] != (3, 512, 512):
        raise ValueError('Only measured native RGB may enter inference')
    probabilities, boxes = [], []
    started = time.time()
    with torch.inference_mode():
        for i in range(0, len(data['rgb']), 4):
            output = model(data['rgb'][i:i + 4])
            probabilities.append(output['probability'].cpu())
            boxes.append(output['boxes'].cpu())
            save(args.output / 'progress.json', dict(status='predicting', predicted=min(i + 4, len(data['rgb'])), elapsed_s=time.time() - started))
    probability, boxes = torch.cat(probabilities), torch.cat(boxes)
    if not torch.isfinite(probability).all() or not torch.isfinite(boxes).all():
        raise ValueError('Nonfinite measured-RGB predictions')
    torch.save(dict(probability=probability, boxes=boxes, threshold=threshold), args.output / 'predictions.pt')
    inference_s = time.time() - started
    # Ground-truth layout and actual Astra replies are first loaded after all predictions are saved.
    for name in ['rows', 'scoring']:
        if sha(args.cache / (name + '.json')) != manifest[name + '_sha256']:
            raise ValueError('Scoring provenance checksum mismatch')
    rows = json.loads((args.cache / 'rows.json').read_text())
    labels = json.loads((args.cache / 'scoring.json').read_text())
    if len(rows) != len(labels) or len(rows) != len(probability):
        raise ValueError('Prediction and scoring lengths differ')
    output_rows = []
    for i, (row, label) in enumerate(zip(rows, labels)):
        confidence, prediction = probability[i, :, :6].max(-1)
        accepted = confidence >= threshold
        target = SYMBOLS.index(label['target_symbol'])
        candidates = [c for c in range(6) if accepted[c] and prediction[c] == target]
        semantic = label['tool'] in ('reorient_target', 'verify_target', 'stop')
        chosen = COLORS.index(label['color']) if label['color'] in COLORS else None
        correct_astra = semantic and chosen == label['target_color'] and label['goal_symbol'] == label['target_symbol']
        contradiction = semantic and chosen is not None and bool(accepted[chosen]) and SYMBOLS[int(prediction[chosen])] != label['goal_symbol']
        output_rows.append(dict(**row, request_status=label['request_status'],
            colors=[dict(color=c, symbol=int(prediction[c]), confidence=float(confidence[c]), accepted=bool(accepted[c]), correct=int(prediction[c]) == label['layout'][c], box=boxes[i, c].tolist()) for c in range(6)],
            target_candidates=candidates, unique_correct_target=len(candidates) == 1 and candidates[0] == label['target_color'],
            astra_tool=label['tool'], astra_semantic_tool=semantic, astra_binding_correct=correct_astra,
            contradicts_astra=bool(contradiction), target_symbol=label['target_symbol'], target_color=label['target_color']))
    metrics = summarize(output_rows)
    delivered = summarize([row for row in output_rows if row['request_status'] == 'delivered'])
    # This is an additional diagnostic, not permission to deploy a below-gate model.
    candidate = bool(result['offline_gate_passed'] and metrics['accepted_bindings'] >= 50 and (metrics['precision'] or 0) >= .95 and
                     metrics['contradicted_correct_binding_frames'] == 0 and metrics['unique_correct_target_on_wrong_binding_frames'] > 0)
    final = dict(status='completed', **metrics, delivered_request_metrics=delivered, threshold=threshold,
                 offline_gate_passed=result['offline_gate_passed'], shadow_diagnostic_passed=candidate, cloud_eligible=False,
                 checkpoint_sha256=sha(args.checkpoint), input_sha256=manifest['inputs_sha256'], model_source_sha256=sha(model_path),
                 source_sha256=sha(__file__), inference_seconds=inference_s, cloud_calls=0, rows=output_rows,
                 note='All outputs computed from RGB before scoring labels and responses were loaded. Previously inspected compressed development recordings. Correction opportunities do not establish counterfactual task success. Fresh-seed validation and control integration remain required.')
    save(args.output / 'result.json', final)
    save(args.output / 'progress.json', dict(status='completed', shadow_diagnostic_passed=candidate, elapsed_s=time.time() - started))
    print(json.dumps({key: value for key, value in final.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
