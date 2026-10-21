"""Release a hold on fresh loss of evidence, once per pending request."""
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / 'astra_retention_v1/window.py'
# Import via its normal module name to preserve the established state contract.
import sys
sys.path.insert(0, str(_path.parent))
from window import RetentionWindow


class FeedbackHoldWindow(RetentionWindow):
    def update(self, *, step, pending, grounded_command, goal, ready, fresh, selected_rate):
        same_request = pending and grounded_command and goal is not None and goal == self.goal
        if same_request and self.start is not None and fresh and not ready:
            self.start = self.rate = None
            self.spent = True
            return None
        return super().update(step=step, pending=pending, grounded_command=grounded_command,
            goal=goal, ready=ready, fresh=fresh, selected_rate=selected_rate)
