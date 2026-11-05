"""Language-conditioned visual supervisor; no simulator state enters this module.

The VLM returns a symbolic goal and visible evidence, never an action embedding.
The deterministic router selects a frozen skill from that evidence. Feedback is
computed exclusively from RGB and measured proprioception. No success oracle.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import time
import urllib.request
import hashlib
import numpy as np

from visual_agent import COLORS, Decision, VLMClient, image_evidence

SYMBOLS = ('star', 'circle', 'triangle', 'cross', 'diamond', 'bars')


class SemanticVLMClient(VLMClient):
    def parse_instruction(self, instruction):
        prompt = """Convert the user's requested goal into JSON. Read only what the user requests.
Supported objects of reference: star (five-pointed symbol), circle (round symbol), triangle (three corners), cross (plus sign), diamond (rhombus), bars (two parallel bars).
Supported destination: facing up, on top, or facing the ceiling, all represented as up.
Return unknown for ambiguous, negated, or unsupported goals. Do not assume a default symbol or direction.
JSON only: {"goal_symbol":"star|circle|triangle|cross|diamond|bars|unknown","goal_direction":"up|unknown"}
User instruction: """ + json.dumps(instruction, ensure_ascii=False)
        request = {'images': [], 'prompt': prompt, 'max_tokens': 80}
        start = time.monotonic()
        req = urllib.request.Request(self.url, data=json.dumps(request).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=120) as f:
            raw = json.load(f)['text']
        parsed = {}
        decoder = json.JSONDecoder()
        for i, char in enumerate(raw):
            if char == '{':
                try:
                    value = decoder.raw_decode(raw[i:])[0]
                    if isinstance(value, dict): parsed = value; break
                except json.JSONDecodeError:
                    pass
        symbol = parsed.get('goal_symbol', 'unknown')
        direction = parsed.get('goal_direction', 'unknown')
        return {'goal_symbol':symbol if symbol in SYMBOLS else 'unknown',
                'goal_direction':direction if direction == 'up' else 'unknown',
                'raw':raw, 'parsed':parsed, 'prompt':prompt, 'observations':[],
                'elapsed_s':time.monotonic()-start,
                'image_sha256':None, 'input_text_sha256':hashlib.sha256(instruction.encode()).hexdigest()}

    def ground(self, rgb, instruction, feedback=None, goal_symbol=None):
        if goal_symbol not in SYMBOLS:
            raise ValueError('Visual grounding requires a goal previously parsed from language')
        prompt = '''Inspect the two images (one view and its enlarged crop).
The object is a robot-held cube. SMALL colored square stickers contain black geometric symbols outlined in white. Ignore large face backgrounds and printed letters.
Symbols: star=five pointed tips; circle=round outline; triangle=three corners; cross=plus sign; diamond=four-corner rhombus; bars=two separate parallel rectangles.
List at most THREE clearly visible small stickers. Do not infer hidden marks. Use unknown for an unclear symbol or color. No particular target is guaranteed visible.
Do not reinterpret the requested goal based on what happens to be visible. Your output only describes visible evidence.
JSON only, no explanation: {"stickers":[{"color":"red|green|blue|yellow|magenta|cyan|unknown","symbol":"star|circle|triangle|cross|diamond|bars|unknown"}]}
User instruction: ''' + json.dumps(instruction, ensure_ascii=False)
        prompt += '\nGoal previously parsed from that instruction: ' + goal_symbol + ', facing up.'
        if feedback:
            prompt += '\nMeasured execution feedback (not semantic truth): ' + json.dumps(feedback, sort_keys=True)
        result = self.ask(rgb, prompt, max_tokens=192)
        result['goal_symbol'] = goal_symbol
        result['goal_direction'] = 'up'
        result['prompt'] = prompt
        stickers = result['parsed'].get('stickers', [])
        valid = []
        for sticker in stickers if isinstance(stickers, list) else []:
            if isinstance(sticker, dict) and sticker.get('color') in COLORS and sticker.get('symbol') in SYMBOLS:
                valid.append({'color': sticker['color'], 'symbol': sticker['symbol']})
        result['observations'] = valid
        return result


class ObservationFeedback:
    """Cheap motion/visibility diagnostics, not an object pose or slip estimator."""
    def __init__(self):
        self.rgb = None
        self.proprio = None
        self.last_step = None
        self.quiet_ticks = 0

    def update(self, rgb, proprio, step):
        # Spatial subsampling is deterministic and has no semantic labels.
        small = np.asarray(rgb, dtype=np.float32)[::8, ::8] / 255.
        prop = np.asarray(proprio, dtype=np.float32)
        if not np.isfinite(prop).all():
            raise ValueError('Non-finite observed proprioception')
        image_change = None if self.rgb is None else float(np.abs(small - self.rgb).mean())
        joint_change = None if self.proprio is None else float(np.abs(prop - self.proprio).mean())
        quiet = image_change is not None and image_change < .006
        self.quiet_ticks = self.quiet_ticks + 1 if quiet else 0
        self.rgb = small.copy()
        self.proprio = prop.copy()
        self.last_step = int(step)
        return {'image_change': image_change, 'proprio_change': joint_change,
                'visually_quiet_ticks': self.quiet_ticks}


class SemanticAgent:
    """v3 modes differ in scheduling only; both share grounding, memory and skills.

periodic: asks every 90 steps and inspects each face for 90 steps.
feedback: asks on a changed/settled view, shortens stalled inspections, and
permits one bounded reobservation manoeuvre after an execution stall.
Neither mode certifies success or assumes visual stillness means completion.
"""
    def __init__(self, mode, client, instruction, budget=6):
        if mode not in ('once', 'periodic', 'feedback'):
            raise ValueError(mode)
        self.mode, self.client, self.instruction = mode, client, instruction
        if budget < 2:
            raise ValueError('Budget must include at least one language and one visual call')
        self.budget = budget
        self.calls = 0
        self.goal = None
        self.goal_symbol = None
        self.goal_direction = None
        self.trace = []
        self.feedback_trace = []
        self.transitions = []
        self.monitor = ObservationFeedback()
        self.evidence = {}
        self.visited = [0] * 6
        self.inspect_color = None
        self.inspect_since = 0
        self.last_call = -1000
        self.last_call_rgb = None
        self.execute_since = None
        self.pending_conflict = None
        self.recovery_used = False
        self.recover_until = -1
        self.recover_color = None
        self.last_decision = Decision(None, 'initial', 'not_observed')

    def _choose_inspection(self, mass):
        # Negative evidence requires distinct observations, not repeated frames.
        excluded = set()
        if self.goal_symbol:
            for color, symbols in self.evidence.items():
                if any(s != self.goal_symbol and len(frames) >= 2 for s, frames in symbols.items()) and self.goal_symbol not in symbols:
                    excluded.add(COLORS.index(color))
        options = [i for i in range(6) if i not in excluded] or list(range(6))
        return min(options, key=lambda i: (self.visited[i], -float(mass[i])))

    def _remember_and_bind(self, result, step):
        symbol, direction = result['goal_symbol'], result['goal_direction']
        if symbol not in SYMBOLS or direction != 'up':
            return 'unresolved_instruction'
        if self.goal_symbol is None:
            self.goal_symbol, self.goal_direction = symbol, direction
        elif symbol != self.goal_symbol or direction != self.goal_direction:
            return 'instruction_parse_conflict'
        frame = result['image_sha256']
        observations = result.get('observations', [])
        for item in observations:
            self.evidence.setdefault(item['color'], {}).setdefault(item['symbol'], set()).add(frame)
        candidates = {COLORS.index(x['color']) for x in observations if x['symbol'] == self.goal_symbol}
        if len(candidates) != 1:
            return 'retain_goal_without_unique_new_evidence'
        candidate = next(iter(candidates))
        if self.goal is None:
            self.goal = candidate
            self.execute_since = step
            return 'bind_visible_semantic_target'
        if candidate == self.goal:
            self.pending_conflict = None
            return 'confirm_existing_goal'
        if self.pending_conflict and self.pending_conflict[0] == candidate and self.pending_conflict[1] != frame:
            self.goal = candidate
            self.execute_since = step
            self.pending_conflict = None
            return 'revise_goal_after_distinct_evidence'
        self.pending_conflict = (candidate, frame)
        return 'conflicting_candidate_awaits_more_evidence'

    def tick(self, rgb, proprio, step):
        if self.calls == 0:
            result = self.client.parse_instruction(self.instruction)
            self.calls += 1
            binding = self._remember_and_bind(result, step)
            self.trace.append(dict(step=step, operation='parse_language', reason='initial_language_parsing',
                                   instruction=self.instruction, previous_goal=None, accepted_goal=None,
                                   binding=binding, **result))
            if self.goal_symbol is None:
                decision = Decision(None, 'abstain', 'unresolved_language_goal')
                self.transitions.append({'step':step,'from':asdict(self.last_decision),'to':asdict(decision)})
                self.last_decision = decision
                return decision
        _, mass = image_evidence(rgb)
        feedback = self.monitor.update(rgb, proprio, step)
        compact = np.asarray(rgb, dtype=np.float32)[::8, ::8] / 255.
        view_change = None if self.last_call_rgb is None else float(np.abs(compact - self.last_call_rgb).mean())
        reason = None
        age = step - self.last_call
        quiet = feedback['visually_quiet_ticks'] >= 3
        if self.last_call_rgb is None:
            reason = 'initial_visual_grounding'
        elif self.mode == 'periodic' and age >= 90:
            reason = 'periodic'
        elif self.mode == 'feedback' and age >= 40:
            if self.goal is None:
                if view_change is not None and view_change >= .025 and (quiet or age >= 70):
                    reason = 'new_observation_after_motion'
                elif age >= 110:
                    reason = 'unresolved_observation_timeout'
            elif self.pending_conflict is not None and view_change is not None and view_change >= .015:
                reason = 'resolve_goal_conflict'
            elif step - self.execute_since >= 90 and quiet and age >= 90:
                reason = 'execution_stall_reobserve'
            elif age >= 160 and view_change is not None and view_change >= .04:
                reason = 'execution_view_changed'
        event = dict(step=int(step), previous_skill=self.last_decision.color,
                     previous_state=self.last_decision.state, change_since_vlm=view_change, **feedback)
        self.feedback_trace.append(event)
        if reason and self.calls < self.budget:
            measured = {k: v for k, v in event.items() if k != 'step'}
            result = self.client.ground(rgb, self.instruction, measured, goal_symbol=self.goal_symbol)
            old_goal = self.goal
            binding = self._remember_and_bind(result, step)
            self.calls += 1
            self.last_call = step
            self.last_call_rgb = compact.copy()
            self.trace.append(dict(step=step, operation='visual_grounding', reason=reason, instruction=self.instruction,
                                   previous_goal=old_goal, accepted_goal=self.goal, binding=binding,
                                   feedback=measured, **result))
            # A diagnostic manoeuvre is bounded; it is NOT proof of recovery success.
            if reason == 'execution_stall_reobserve' and not self.recovery_used and self.goal is not None:
                target_visible = any(x['color'] == COLORS[self.goal] and x['symbol'] == self.goal_symbol
                                     for x in result.get('observations', []))
                if not target_visible:
                    options = [i for i in range(6) if i != self.goal]
                    self.recover_color = max(options, key=lambda i: float(mass[i]))
                    self.recover_until = step + 20
                    self.recovery_used = True
        if self.goal is not None:
            if step < self.recover_until:
                decision = Decision(self.recover_color, 'recover_observe', 'bounded_reobservation_20_steps')
            else:
                decision = Decision(self.goal, 'execute', 'execute_grounded_language_goal')
        elif self.mode == 'once':
            decision = Decision(None, 'abstain', 'no_grounded_goal')
        elif self.goal_symbol is None:
            # Do not act on an unsupported/unresolved language goal.
            decision = Decision(None, 'abstain', 'unresolved_language_goal')
        else:
            inspect_age = step - self.inspect_since
            switch = self.inspect_color is None or inspect_age >= 90
            if self.mode == 'feedback' and inspect_age >= 60 and quiet:
                switch = True
            if switch:
                self.inspect_color = self._choose_inspection(mass)
                self.visited[self.inspect_color] += 1
                self.inspect_since = step
            decision = Decision(self.inspect_color, 'inspect', 'seek_unobserved_semantic_evidence')
        if asdict(decision) != asdict(self.last_decision):
            self.transitions.append({'step': step, 'from': asdict(self.last_decision), 'to': asdict(decision)})
        self.last_decision = decision
        return decision
