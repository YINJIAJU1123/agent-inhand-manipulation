"""Run independent v4 simulator processes and aggregate paired development tests."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--latent-checkpoint', required=True)
    p.add_argument('--modes', nargs='+', choices=['latent', 'oracle', 'astra'], default=['latent', 'oracle'])
    p.add_argument('--record-video', action='store_true')
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    protocol = json.loads((ROOT / 'configs/agentic_two_track_v4.json').read_text())
    dest = Path(a.output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='4')
    deps = [str(ROOT / 'outputs' / d) for d in ['vlm_dependencies', 'learning_dependencies', 'video_dependencies']]
    env['PYTHONPATH'] = ':'.join(deps + [env.get('PYTHONPATH', '')])
    runs = []
    for scenario in protocol['test_scenarios']:
        for mode in a.modes:
            seed = scenario['seed']
            output = dest / f'{seed}_{mode}'
            if (output / 'report.json').exists():
                raise RuntimeError(f'Refusing to overwrite completed evaluation: {output}')
            if mode == 'astra' and not os.environ.get('OPENAI_API_KEY'):
                raise RuntimeError('OPENAI_API_KEY missing; real Astra evaluation requires credentials')
            command = [sys.executable, str(ROOT / 'scripts/agentic/evaluate_two_agent_v4.py'),
                       '--headless', '--output', str(output), '--seeds', str(seed), '--modes', mode,
                       '--template-set', scenario['template_set'], '--instruction-offset', str(scenario['instruction_offset']),
                       '--max-steps', str(protocol['max_steps']), '--budget', str(protocol['call_budget']),
                       '--latent-checkpoint', str(Path(a.latent_checkpoint).resolve()),
                       '--astra-model', protocol['astra_model']]
            if a.record_video and seed == protocol['test_scenarios'][0]['seed']:
                command.append('--record-video')
            with (dest / f'{seed}_{mode}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=a.timeout, check=True)
            report = json.loads((output / 'report.json').read_text())
            initial = json.loads((output / f'initial_{seed}_{mode}.json').read_text())
            runs.append({'mode': mode, 'seed': seed, 'report': report, 'initial': initial})
            episodes = report['episodes']
            print(json.dumps({'completed': mode, 'seed': seed, 'episodes': len(episodes),
                              'success': sum(x['success'] for x in episodes)}), flush=True)
    pairing = {}
    for scenario in protocol['test_scenarios']:
        group = [r for r in runs if r['seed'] == scenario['seed']]
        pairing[str(scenario['seed'])] = all(r['initial'] == group[0]['initial'] for r in group)
    summary = {'benchmark': protocol['benchmark'], 'initial_states_exactly_paired': pairing,
               'limits': protocol['limits'], 'modes': {}}
    for mode in a.modes:
        episodes = [e for run in runs if run['mode'] == mode for e in run['report']['episodes']]
        summary['modes'][mode] = {'episodes': len(episodes), 'success': sum(e['success'] for e in episodes),
                                  'grounding_correct': sum(e['grounding_correct'] for e in episodes),
                                  'drops': sum(e['drop'] for e in episodes), 'calls': sum(e['calls'] for e in episodes),
                                  'stop_claimed': sum(e['stop_claimed'] for e in episodes),
                                  'false_stop': sum(e['false_stop'] for e in episodes)}
    summary['astra_status'] = 'evaluated' if 'astra' in a.modes else 'not_run'
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if not all(pairing.values()):
        raise RuntimeError('Initial states differ; do not report this as a paired comparison')


if __name__ == '__main__':
    main()
