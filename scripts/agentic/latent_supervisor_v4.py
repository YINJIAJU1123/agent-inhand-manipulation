"""Periodic supervisor for the learned-latent baseline; scheduling is not learned."""
from __future__ import annotations
import time
import numpy as np
from visual_agent import Decision, image_evidence
from semantic_agent_v3 import ObservationFeedback


class LatentAgent:
    def __init__(self, predictor, instruction, budget=6):
        self.predictor, self.instruction, self.budget = predictor, instruction, budget
        self.calls = 0
        self.goal = self.goal_symbol = self.goal_direction = None
        self.goal_embedding = None
        self.trace, self.feedback_trace, self.transitions = [], [], []
        self.monitor = ObservationFeedback()
        self.last_call = -90
        self.visited = [0] * 6
        self.inspect = None
        self.inspect_since = -90
        self.last_decision = Decision(None, 'initial', '')
        self.declared_success = False
        self.recovery_used = False

    def tick(self, rgb, proprio, step):
        feedback = self.monitor.update(rgb, proprio, step)
        self.feedback_trace.append(dict(step=int(step), **feedback))
        if step - self.last_call >= 90 and self.calls < self.budget:
            start = time.monotonic()
            result = self.predictor.predict(rgb, self.instruction)
            self.calls += 1
            self.last_call = step
            if result['accepted']:
                self.goal = result['skill']
                self.goal_embedding = np.asarray(result['goal_embedding'], dtype=np.float32)
                self.goal_direction = 'up'
            self.trace.append(dict(step=int(step), operation='learned_latent', instruction=self.instruction,
                                   accepted_goal=self.goal, elapsed_s=time.monotonic() - start,
                                   **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}))
        if self.goal is not None:
            decision = Decision(self.goal, 'execute', 'execute_predicted_continuous_goal_tensor')
        else:
            if self.inspect is None or step - self.inspect_since >= 90:
                _, mass = image_evidence(rgb)
                self.inspect = min(range(6), key=lambda i: (self.visited[i], -float(mass[i])))
                self.visited[self.inspect] += 1
                self.inspect_since = step
            decision = Decision(self.inspect, 'inspect', 'periodic_visible_unvisited_skill')
        if vars(decision) != vars(self.last_decision):
            self.transitions.append({'step': int(step), 'from': vars(self.last_decision), 'to': vars(decision)})
        self.last_decision = decision
        return decision
