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
    repeated_summary,
)


def test_repeated_summary_keeps_drop_and_horizon_censored_episodes():
    rows = [
        {"consecutive_successes": 2, "successes_by_face": [1, 1, 0, 0, 0, 0], "horizon_censored": False},
        {"consecutive_successes": 4, "successes_by_face": [0, 0, 1, 1, 1, 1], "horizon_censored": True},
        {"consecutive_successes": 0, "successes_by_face": [0]*6, "horizon_censored": False},
    ]
    summary = repeated_summary(rows)
    assert summary["episodes"] == 3
    assert summary["cs_mean"] == 2
    assert math.isclose(summary["cs_std"], math.sqrt(8/3))
    assert summary["horizon_censored_episodes"] == 1
    assert summary["successes_by_face"] == [1]*6
    assert repeated_summary([])["cs_mean"] is None


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
            "mean_raw_out_of_bounds_fraction": 0.0,
            "mean_action_slew_l2": 0.2,
            "mean_applied_target_delta_rad": 0.08,
            "mean_joint_velocity_rms_rad_s": 1.0,
            "max_joint_velocity_norm_rad_s": 3.0,
            "mean_object_angular_velocity_rad_s": 2.0,
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
            "mean_raw_out_of_bounds_fraction": 0.1,
            "mean_action_slew_l2": 0.8,
            "mean_applied_target_delta_rad": 0.2,
            "mean_joint_velocity_rms_rad_s": 2.0,
            "max_joint_velocity_norm_rad_s": 5.0,
            "mean_object_angular_velocity_rad_s": 4.0,
        },
    ]
    summary = aggregate_records(records)
    assert summary["episodes"] == 2
    assert summary["continuous_hold"]["successes"] == 1
    assert summary["drop"]["successes"] == 1
    assert math.isclose(summary["steps"]["mean"], 60.0)
