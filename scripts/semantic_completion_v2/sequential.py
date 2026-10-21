"""Trajectory-level stop calibration and differentiable first-stop supervision."""
import math
import torch
import torch.nn.functional as F

THRESHOLD_GRID = (.5, .6, .7, .8, .85, .9, .95, .975, .99, .995, .999)


def trajectories(rows):
    groups = {}
    for j, row in enumerate(rows):
        key = (row['seed'], row['case'], row.get('segment', 0), row['skill'])
        groups.setdefault(key, []).append(j)
    return [(key, sorted(ids, key=lambda j: rows[j]['step'])) for key, ids in groups.items()]


def first_stop_loss(logits, ready, gate_probability=.85):
    """Negative log likelihood of a correct first stop, or no stop if never ready.

    Labels supervise training only. The fixed shift makes the soft hazard
    centered at the baseline runtime gate. No independent-frame assumption is
    needed to use this differentiable decision objective; it is not a calibrated
    model of physical-event probabilities.
    """
    shifted = logits - math.log(gate_probability / (1 - gate_probability))
    log_continue = F.logsigmoid(-shifted)
    if not ready.any():
        return -log_continue.sum()
    log_survival = torch.cat((logits.new_zeros(1), log_continue.cumsum(0)[:-1]))
    log_first_stop = log_survival + F.logsigmoid(shifted)
    return -torch.logsumexp(log_first_stop[ready], dim=0)


def score(probabilities, data, threshold):
    episodes = []
    for key, ids in trajectories(data['rows']):
        color = key[-1]
        proposed = [] if threshold >= 1.0 else [j for j in ids if float(probabilities[j, color]) >= threshold]
        first = proposed[0] if proposed else None
        e = float(data['error'][first, color]) if first is not None else None
        episodes.append(dict(seed=key[0], case=key[1], segment=key[2], goal=color,
            sampled_reach=bool((data['error'][ids, color] <= .16).any()),
            first_step=data['rows'][first]['step'] if first is not None else None,
            first_error=e, correct=first is not None and e <= .16))
    accepted = sum(r['first_step'] is not None for r in episodes)
    correct = sum(r['correct'] for r in episodes)
    reached = sum(r['sampled_reach'] for r in episodes)
    return dict(episodes=len(episodes), accepted=accepted, correct=correct,
        false_first_stops=accepted-correct, precision=correct/accepted if accepted else None,
        reached=reached, reached_recall=correct/reached if reached else None, details=episodes)


def calibrate(probabilities, data):
    curve = [dict(threshold=t, **score(probabilities, data, t)) for t in THRESHOLD_GRID]
    eligible = [r for r in curve if r['accepted'] >= 3 and r['precision'] >= .9]
    selected = max(eligible, key=lambda r: (r['correct'], r['precision'], -r['threshold'])) if eligible else None
    return selected, curve
