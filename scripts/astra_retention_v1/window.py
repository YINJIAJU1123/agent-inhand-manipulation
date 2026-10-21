"""Bound a learned retention choice to the currently pending semantic goal.

This state machine never declares success and never invents a semantic binding.
It only applies a supplied, learned rate while the same goal awaits a reply.
"""
from dataclasses import dataclass

@dataclass
class RetentionWindow:
 horizon: int = 180
 goal: int | None = None
 start: int | None = None
 rate: float | None = None
 spent: bool = False

 def update(self, *, step, pending, grounded_command, goal, ready, fresh, selected_rate):
  # grounded_command means Astra issued execute/verify for this color, not GT.
  if not pending or not grounded_command or goal is None:
   self.goal=self.start=self.rate=None;self.spent=False;return None
  if self.goal is not None and self.goal!=goal:
   self.goal=self.start=self.rate=None;self.spent=False
  if self.start is not None:
   if step-self.start>=self.horizon:
    self.start=self.rate=None;self.spent=True;return None
   return self.rate
  if self.spent or not ready or not fresh:return None
  if selected_rate not in (0.,.25,1.):raise ValueError('Unsupported learned retention action')
  self.goal=goal;self.start=step;self.rate=selected_rate
  return self.rate
