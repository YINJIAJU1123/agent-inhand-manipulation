"""Extend fixed request-scoped retention to commanded observation colors."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/astra_retention_online_v1'))
from retention_runtime import RetentionRuntime


class ObservationHold:
    def __init__(self, cases, mode, threshold=.7, horizon=180):
        if mode not in ('none', 'execute', 'all'):
            raise ValueError('Unknown observation hold mode')
        self.mode = mode
        self.core = RetentionRuntime(cases, mode='none' if mode == 'none' else 'hold', threshold=threshold, horizon=horizon)

    @property
    def events(self):
        return self.core.events

    def apply(self, **inputs):
        states = list(inputs['states'])
        if self.mode == 'all':
            # The retained color is the public commanded observation, not an
            # assertion of semantic grounding or a task-completion decision.
            inputs['states'] = ['execute' if state == 'inspect' else state for state in states]
        previous = len(self.core.events)
        result = self.core.apply(**inputs)
        for event in self.core.events[previous:]:
            event['commanded_state'] = states[event['case']]
        return result
