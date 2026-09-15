"""Unit tests for camera-free baseline accounting (no Isaac imports)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "rsl_rl"))

from state_baseline_protocol import (
    EpisodeQuota,
    aggregate_records,
    mean_summary,
    rate_summary,
    wilson_interval,
)


def test_quota_balances_vector_slots_without_fast_success_bias():
    quota = EpisodeQuota(10, 3)
    assert quota.quotas == (4, 3, 3)
    assert sum(quota.quotas) == 10


def test_wilson_interval_is_finite_and_contains_observed_rate():
    low, high = wilson_interval(50, 100)
    assert 0.0 < low < 0.5 < high < 1.0
    assert wilson_interval(0, 10)[0] == 0.0
    assert math.isclose(wilson_interval(10, 10)[1], 1.0)


def test_empty_summaries_are_strict_json_safe():
    assert rate_summary([])["rate"] is None
    assert mean_summary([])["mean"] is None


def test_aggregate_records_reports_hold_reach_drop_and_control_metrics():
    records = [
        {
            "instant_reach": True,
            "continuous_hold": True,
            "held_at_end": True,
            "drop": False,
            "final_orientation_error_rad": 0.04,
            "min_orientation_error_rad": 0.03,
            "steps": 100,
            "time_s": 3.0,
            "mean_clipped_action_l2": 1.2,
            "mean_action_slew_l2": 0.2,
            "mean_target_delta_rad": 0.08,
        },
        {
            "instant_reach": True,
            "continuous_hold": False,
            "held_at_end": False,
            "drop": True,
            "final_orientation_error_rad": 0.3,
            "min_orientation_error_rad": 0.1,
            "steps": 20,
            "time_s": 0.6,
            "mean_clipped_action_l2": 2.0,
            "mean_action_slew_l2": 0.8,
            "mean_target_delta_rad": 0.2,
        },
    ]
    summary = aggregate_records(records)
    assert summary["episodes"] == 2
    assert summary["continuous_hold"]["successes"] == 1
    assert summary["drop"]["successes"] == 1
    assert math.isclose(summary["steps"]["mean"], 60.0)
