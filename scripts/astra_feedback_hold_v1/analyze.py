"""Recompute the frozen paired comparison from every nominal episode."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


MODES = ("none", "hold", "feedback")
SEEDS = ("117101", "117102")


def aggregate(reports):
    episodes = [episode for report in reports for episode in report["episodes"]]
    return dict(cases=len(episodes), drops=sum(e["drop"] for e in episodes),
        prefix_drops=sum(r["prefix_drops"] for r in reports),
        final_ready=sum(e["final_ready"] for e in episodes),
        execute_ready_fraction=sum(e["execute_ready_steps"] for e in episodes)/(len(episodes)*1140),
        inspect_ready_fraction=sum(e["inspect_ready_steps"] for e in episodes)/(len(episodes)*480))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    result = json.loads(raw)
    assert result["status"] == "completed" and result["cloud_calls"] == 0
    assert set(result["by_seed"]) == set(SEEDS)
    rows, by_seed = [], {}
    for seed in SEEDS:
        group = result["by_seed"][seed]
        assert set(group) == {"record", *MODES}, "Incomplete branch set; retain as incomplete"
        assert group["record"]["branch_step"] is not None
        for mode in MODES:
            report = group[mode]
            assert report["status"] == "completed" and report["cases"] == 6
            assert report["feedback_release"] == (mode == "feedback")
            assert report["branch_step"] == group["record"]["branch_step"]
            assert [e["case"] for e in report["episodes"]] == list(range(6))
            assert report["source_sha256"] == group["record"]["source_sha256"]
            for key in ("feedback_window_sha256", "feedback_wrapper_sha256"):
                assert report[key] == group["record"][key]
            calculated = aggregate([report])
            for key, value in calculated.items():
                assert math.isclose(value, report[key], rel_tol=0, abs_tol=1e-12), (seed, mode, key)
        by_seed[seed] = {mode: aggregate([group[mode]]) for mode in MODES}
        for case in range(6):
            rows.append(dict(seed=seed, case=case,
                modes={mode: group[mode]["episodes"][case] for mode in MODES}))
    totals = {mode: aggregate([result["by_seed"][s][mode] for s in SEEDS]) for mode in MODES}
    for mode in MODES:
        for key, value in totals[mode].items():
            assert math.isclose(value, result["by_mode"][mode][key], rel_tol=0, abs_tol=1e-12)
    contrasts = {}
    for reference in ("none", "hold"):
        pairs = [(r["modes"]["feedback"], r["modes"][reference]) for r in rows]
        contrasts[reference] = dict(
            final_ready_wins=sum(a["final_ready"] and not b["final_ready"] for a, b in pairs),
            final_ready_losses=sum(b["final_ready"] and not a["final_ready"] for a, b in pairs),
            drops_prevented=sum(b["drop"] and not a["drop"] for a, b in pairs),
            new_drops=sum(a["drop"] and not b["drop"] for a, b in pairs),
            occupancy_wins=sum(a["execute_ready_steps"] > b["execute_ready_steps"] for a, b in pairs),
            occupancy_losses=sum(a["execute_ready_steps"] < b["execute_ready_steps"] for a, b in pairs),
            occupancy_difference=totals["feedback"]["execute_ready_fraction"] - totals[reference]["execute_ready_fraction"])
    f = totals["feedback"]
    gate = all(f["execute_ready_fraction"] >= totals[m]["execute_ready_fraction"] + .05
        and f["drops"] <= totals[m]["drops"] and f["final_ready"] >= totals[m]["final_ready"] for m in ("none", "hold"))
    assert gate == result["local_gate_passed"]
    drop_phases = {}
    for mode in MODES:
        episodes = [r["modes"][mode] for r in rows]
        dropped = [e for e in episodes if e["drop"]]
        drop_phases[mode] = dict(phases=dict(Counter(e["drop_phase"] for e in dropped)),
            during_request_wait=sum(e["pending_drop"] for e in dropped),
            outside_request_wait=sum(not e["pending_drop"] for e in dropped))
    audit = dict(status="completed", all_nominal_cases=12, by_mode=totals, by_seed=by_seed,
        paired_feedback_vs=contrasts, drop_phases=drop_phases, local_gate_passed=gate, cases=rows,
        cloud_calls=0, deployed=False, source_result_sha256=hashlib.sha256(raw).hexdigest(),
        scope="Two seeds of fixed supplied colors. Shared prefixes are checked by the frozen cloud driver. No language binding or autonomous stop. Cases within a seed share a batched simulation; no large-sample significance or Astra success claim.")
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("by_mode", "by_seed", "paired_feedback_vs", "drop_phases", "local_gate_passed")}))


if __name__ == "__main__":
    main()
