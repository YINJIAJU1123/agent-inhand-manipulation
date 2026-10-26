"""CPU admission of the published action component. No simulator or API calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/semantic_completion_v1'), str(ROOT/'scripts/agentic'), str(ROOT/'source/BrainCo_DexHand')]
import torch
from common import load_student
from adapted_actor import AdaptedActionProposal
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--observations', type=Path, help='Optional local student-data shard. No labels are used.')
    args = p.parse_args()
    torch.set_num_threads(2)
    source, text, pack = load_student('cpu')
    if args.observations:
        d = torch.load(args.observations, map_location='cpu', weights_only=True)
        rgb, prop, goals = d['rgb_history'][:6].float(), d['proprio_history'][:6].float(), text[d['goal_id'][:6].long()]
        input_scope = 'First six recorded observations from a caller-provided shard. Not a performance estimate.'
    else:
        rng = torch.Generator().manual_seed(20261026)
        rgb = torch.randn(6, pack['history'], pack['rgb_dim'], generator=rng) * .01
        prop = torch.randn(6, pack['history'], pack['proprio_dim'], generator=rng) * .01
        goals = text[:6]
        input_scope = 'Synthetic finite-valued interface checks only. Not empirical control performance.'
    batch = VisualStudentBatch(rgb, goals, prop)
    actor = AdaptedActionProposal(source, pack, ROOT/'checkpoints/student/adapted_action_student.pt')
    with torch.inference_mode():
        original = source(batch)
        saved = {k: v.clone() for k, v in original.items()}
        candidate = actor.model(batch)['action']
        masks = [torch.zeros(6, dtype=torch.bool), torch.ones(6, dtype=torch.bool),
                 torch.tensor([True, False, True, False, True, False])]
        for known in masks:
            for active in masks:
                for enabled in (False, True):
                    action = actor.action(original, batch, known, active, enabled=enabled)
                    use = known & active & enabled
                    assert torch.equal(action[use], candidate[use])
                    assert torch.equal(action[~use], original['action'][~use])
                    assert all(torch.equal(original[k], v) for k, v in saved.items())
        assert all(torch.equal(source.state_dict()[k], v.cpu()) for k, v in pack['model'].items())
        for invalid in (torch.ones(6), torch.ones(5, dtype=torch.bool)):
            try: actor.action(original, batch, invalid, masks[1])
            except ValueError: pass
            else: raise AssertionError('Invalid mask accepted')
        changed = int(((candidate-original['action']).abs().amax(-1) > 1e-6).sum())
    print(json.dumps(dict(status='passed', checkpoint_sha256=actor.checkpoint_sha256,
        mask_cases=18, rejected_invalid_masks=2, action_shape=list(candidate.shape),
        observations_with_changed_proposal=changed, reference_outputs_unchanged=True,
        reference_parameters_unchanged=True, input_scope=input_scope, network_requests=0)))


if __name__ == '__main__':
    main()
