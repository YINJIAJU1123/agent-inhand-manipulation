"""Pure-Python accounting helpers for the state-policy baseline protocol.

This module deliberately has no Isaac Lab or PyTorch dependency.  It is used by
the simulator evaluator and can therefore be tested on a workstation without
starting Kit.  The evaluator records one row per completed episode and uses
these helpers only after all terminal, pre-reset metrics have been copied to
CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeQuota:
    """Pre-allocate an equal number of episodes to each vector slot.

    A fast-success slot cannot consume trials belonging to a slower slot.  The
    evaluator uses one quota per fixed face/seed block; this class is kept
    independent from the existing Isaac-specific helper for offline testing.
    """

    episodes: int
    num_envs: int

    def __post_init__(self) -> None:
        if self.episodes <= 0 or self.num_envs <= 0:
            raise ValueError("episodes and num_envs must be positive")

    @property
    def quotas(self) -> tuple[int, ...]:
        return tuple(
            self.episodes // self.num_envs + int(i < self.episodes % self.num_envs)
            for i in range(self.num_envs)
        )

    def counts(self) -> list[int]:
        return [0] * self.num_envs


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli rate."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if trials == 0:
        return (float("nan"), float("nan"))
    # z=1.959963984540054 is the 97.5th percentile of N(0,1), avoiding a
    # scipy dependency in the evaluator and its standalone unit tests.
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else _normal_quantile(
        0.5 + confidence / 2.0
    )
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _normal_quantile(probability: float) -> float:
    """Acklam's rational approximation; sufficient for confidence reporting."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between zero and one")
    # Coefficients from Peter John Acklam's public-domain approximation.
    a = (-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if probability < plow:
        q = sqrt(-2.0 * log(probability))
        numerator = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return numerator / denominator
    if probability > phigh:
        q = sqrt(-2.0 * log(1.0 - probability))
        numerator = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -numerator / denominator
    q = probability - 0.5
    r = q * q
    numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    denominator = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return numerator / denominator


def rate_summary(values: Iterable[bool]) -> dict[str, float | int | None]:
    """Summarize a boolean episode outcome with a 95% Wilson interval."""

    outcomes = [bool(value) for value in values]
    successes = sum(outcomes)
    low, high = wilson_interval(successes, len(outcomes))
    return {
        "successes": successes,
        "trials": len(outcomes),
        "rate": successes / len(outcomes) if outcomes else None,
        "wilson_95_low": low if outcomes else None,
        "wilson_95_high": high if outcomes else None,
    }


def mean_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize a finite numeric episode metric."""

    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
    }


def aggregate_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Create protocol summaries from JSON-compatible episode records."""

    def bools(key: str) -> list[bool]:
        return [bool(record[key]) for record in records if key in record]

    def numbers(key: str) -> list[float]:
        return [float(record[key]) for record in records if key in record and record[key] is not None]

    summary: dict[str, object] = {
        "episodes": len(records),
        "instant_reach": rate_summary(bools("instant_reach")),
        "continuous_hold": rate_summary(bools("continuous_hold")),
        "held_at_end": rate_summary(bools("held_at_end")),
        "drop": rate_summary(bools("drop")),
    }
    for key in (
        "final_orientation_error_rad",
        "min_orientation_error_rad",
        "steps",
        "time_s",
        "mean_clipped_action_l2",
        "mean_action_slew_l2",
        "mean_target_delta_rad",
    ):
        summary[key] = mean_summary(numbers(key))
    return summary
