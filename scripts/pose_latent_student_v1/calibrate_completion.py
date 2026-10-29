"""Calibrate first stopping events on captured development trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from control_modules import CompletionEvidence, FIELDS, TOLERANCES
from goals import load_bank
from train_completion import measured_errors
from train_modules import load_data, sha


def decision_segments(data, episode_step):
    state, segments = {}, {}
    for i in range(len(data['goal_index'])):
        env, episode, goal = (int(data[k][i]) for k in ('env_id', 'episode_id', 'goal_index'))
        key = (env, episode)
        previous = state.get(key)
        segment = 0 if previous is None else previous[1] + int(previous[0] != goal)
        state[key] = (goal, segment)
        if data['known'][i] and not data['pending'][i] and int(episode_step[i]) % 10 == 0:
            segments.setdefault((env, episode, segment, goal), []).append(i)
    return segments


def event_metrics(accepted, truth, mask, segments):
    count = correct = reachable = missed = total = 0
    false_events = []
    for key, ids in segments.items():
        if not mask[ids[0]]:
            continue
        total += 1
        goal = key[-1]
        reach = bool(truth[ids, goal].any())
        reachable += int(reach)
        choices = np.flatnonzero(accepted[ids, goal])
        if not len(choices):
            missed += int(reach)
            continue
        index = ids[int(choices[0])]
        count += 1
        ok = bool(truth[index, goal])
        correct += int(ok)
        if not ok:
            false_events.append(dict(segment=key, sample_index=index))
    return dict(segments=total, accepted=count, correct=correct, false_stops=count-correct,
                precision=correct/count if count else None, reachable_segments=reachable,
                recall=correct/reachable if reachable else None, missed_reachable=missed,
                false_events=false_events)


def choose(values, thresholds, greater, truth, mask, segments):
    candidates = []
    for threshold in thresholds:
        accepted = values >= threshold if greater else values <= threshold
        metrics = event_metrics(accepted, truth, mask, segments)
        if metrics['accepted'] >= 20 and metrics['precision'] >= .99:
            candidates.append((metrics['correct'], metrics['precision'], float(threshold), metrics))
    # Equal calibration outcomes use the more permissive gate for earlier stopping.
    return max(candidates, key=lambda x: (x[0], x[1], -x[2] if greater else x[2])) if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'text_cache', 'original_asset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = 'cuda:0'
    manifest, data = load_data(args.data/'validation')
    pack = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if pack['manifest']['source_student_sha256'] != manifest['checkpoint_sha256']:
        raise ValueError('Completion checkpoint student mismatch')
    bank, rotations, _ = load_bank(args.text_cache, args.original_asset, device)
    torch.testing.assert_close(pack['bank'], bank.cpu(), atol=0, rtol=0)
    model = CompletionEvidence(pack['feature_mean'], pack['feature_scale'], pack['bank']).to(device).eval()
    model.load_state_dict(pack['model'])
    model.requires_grad_(False)
    measured, _ = measured_errors(manifest, data, rotations)
    elapsed_offset = sum(data[k].shape[-1] if data[k].ndim > 1 else 1
                         for k in FIELDS[:FIELDS.index('elapsed_pending_s')])
    sensor = data['sensor'].clone()
    sensor[~data['pending'], elapsed_offset] = 0.
    predictions = []
    with torch.inference_mode():
        for ids in torch.arange(len(sensor)).split(256):
            x = sensor[ids].to(device)
            n = len(ids)
            logits = model(x[:, None].expand(-1, 6, -1).reshape(n*6, -1),
                           bank[None].expand(n, -1, -1).reshape(n*6, -1))['ordinal']
            predictions.append(logits[:, TOLERANCES.index(.16)].sigmoid().reshape(n, 6).cpu())
    probability = torch.cat(predictions).numpy()
    truth = (data['angles_rad'] <= .16).numpy()
    episode_step = torch.cat([torch.load(f, map_location='cpu', weights_only=True)['episode_step']
                             for f in sorted((args.data/'validation').glob('samples_*.pt'))])
    segments = decision_segments(data, episode_step)
    calibration = np.array([int.from_bytes(hashlib.sha256(
        f"{manifest['seed']}:{int(env)}:{int(episode)}".encode()).digest()[:4], 'little') % 2 == 0
        for env, episode in zip(data['env_id'], data['episode_id'])])
    report = ~calibration
    measured = measured.numpy()
    selected_pose = choose(measured, torch.linspace(.06, .16, 201).tolist(), False,
                           truth, calibration, segments)
    selected_evidence = choose(probability, torch.linspace(.5, .9999, 200).tolist(), True,
                               truth, calibration, segments)
    results = {}
    for name, values, selected, greater in [('measured_pose', measured, selected_pose, False),
                                           ('ordinal_evidence', probability, selected_evidence, True)]:
        if selected is None:
            results[name] = dict(calibrated=False, reason='No threshold reached 20 accepted calibration events at 99% precision')
            continue
        accepted = values >= selected[2] if greater else values <= selected[2]
        results[name] = dict(calibrated=True, threshold=selected[2], calibration=selected[3],
                             report=event_metrics(accepted, truth, report, segments))
    analysis = dict(scope='First-event replay on fixed development captures, not a closed-loop intervention.',
        decision_rule='first known nonpending 3Hz stop within each continuous same-goal segment',
        threshold_tiebreak='most permissive threshold among equal calibration correct counts and precision',
        scored_tolerance_rad=.16, minimum_calibration_accepts=20, calibration_precision_target=.99,
        dense_states=len(sensor), segments_with_decisions=len(segments), validation_seed=manifest['seed'],
        split='SHA256 seed/env/episode partition, identical to module fitting', runtime_idle_elapsed_zero=True,
        evidence_sha256=sha(args.checkpoint), collection_sha256=sha(args.data/'validation/manifest.json'),
        source_sha256={f:sha(Path(__file__).with_name(f)) for f in
            ('calibrate_completion.py', 'train_completion.py', 'train_modules.py', 'control_modules.py')},
        formal_trials_used=False, final_method_admitted=False, results=results)
    analysis_file = args.output/'first_stop_event_analysis.json'
    analysis_file.write_text(json.dumps(analysis, indent=2)+'\n')
    if selected_pose is not None and selected_evidence is not None:
        selected = dict(format='pose_latent_first_event_calibration_v1',
            evidence_sha256=sha(args.checkpoint), source_student_sha256=manifest['checkpoint_sha256'],
            scored_tolerance_rad=.16, measured_stop_threshold_rad=selected_pose[2],
            evidence_probability_threshold=selected_evidence[2], calibration_source_sha256=sha(analysis_file),
            calibration_source=str(analysis_file), formal_trials_used=False,
            report_limit='Correlated development segments; independent closed-loop comparison required.')
        (args.output/'first_stop_calibration.json').write_text(json.dumps(selected, indent=2)+'\n')
    print(json.dumps(analysis), flush=True)


if __name__ == '__main__':
    main()
