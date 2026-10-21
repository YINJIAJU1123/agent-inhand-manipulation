"""Evaluator-only bookkeeping. Never imported by agent or learned models."""
from dataclasses import dataclass, asdict


@dataclass
class Score:
    reached: bool = False
    success: bool = False
    false_stop: bool = False
    drop: bool = False
    stop_claimed: bool = False
    first_reach_step: int = -1
    finish_step: int = -1
    reason: str = ""

    def update(self, step, grounded, error, dropped, stop=False):
        if self.reason:
            return True
        valid = bool(grounded and error <= .16 and not dropped)
        if valid and not self.reached:
            self.reached, self.first_reach_step = True, step
        if dropped:
            self.drop, self.reason = True, "drop"
        elif stop:
            self.stop_claimed, self.success = True, valid
            self.false_stop = not valid
            self.reason = "success" if valid else "false_stop"
        if self.reason:
            self.finish_step = step
        return bool(self.reason)

    def finish(self, step, reason="horizon"):
        if not self.reason:
            self.reason, self.finish_step = reason, step
        return asdict(self)


def service_anomaly(events, limit=120.):
    return any(e.get("status") == "error" or
               e.get("wall_latency_s",0.) > limit for e in events)
