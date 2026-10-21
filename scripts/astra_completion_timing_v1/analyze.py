"""Audit recorded request timing against measured and evaluator-only evidence.

This reads completed development trajectories. It never changes actions,
scores, thresholds, or the prepared live experiment, and makes no API calls.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


COLORS = ("red", "green", "blue", "yellow", "magenta", "cyan")
GROUNDED_TOOLS = ("reorient_target", "verify_target", "stop")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def analyze(report_path, replay_dir):
    report = json.loads(report_path.read_text())
    manifest = json.loads((replay_dir / "manifest.json").read_text())
    audit = json.loads((replay_dir / "replay_audit.json").read_text())
    assert manifest["status"] == "completed" and audit["maximum"] <= 1e-6
    assert Path(manifest["source"]).resolve() == report_path.parent.resolve()
    assert manifest["actions_sha256"] == digest(report_path.parent / "actions.pt")
    data = torch.load(replay_dir / "data.pt", map_location="cpu", weights_only=False, mmap=True)
    episodes = []
    for ep in report["episodes"]:
        color = ep["target_color"]
        indices = [i for i, row in enumerate(data["rows"]) if row["case"] == ep["case"]]
        sampled = []
        for index in indices:
            row = data["rows"][index]
            sampled.append(dict(step=row["step"], error=float(data["error"][index, color]),
                probability=row["probability"][color],
                correct_binding=(row["claimed_goal"] == color and row["claimed_symbol"] == ep["target_symbol"]),
                state=row["state"]))
        requests = []
        for trace in ep["trace"]:
            if "tool" not in trace:
                continue
            feedback = json.loads(trace.get("prompt", "{}")).get("feedback", {})
            verifier = feedback.get("local_pose_verifier", {})
            probabilities = verifier.get("completion_probability_by_color")
            threshold = verifier.get("threshold")
            args = trace["arguments"]
            selected = COLORS.index(args["color"]) if args.get("color") in COLORS else None
            event = next((event for event in ep["async_events"]
                          if event.get("request_step") == trace["step"]), {})
            delivered = event.get("delivery_step")
            during = [s for s in sampled if delivered is not None and trace["step"] <= s["step"] < delivered]
            request_sample = next((s for s in sampled if s["step"] == trace["step"]), None)
            # Use the first later sample, explicitly recording its offset.
            after = next((s for s in sampled if delivered is not None and s["step"] >= delivered), None)
            requests.append(dict(request_step=trace["step"], delivery_step=delivered,
                tool=trace["tool"], selected_color=selected, symbol=args.get("goal_symbol"),
                binding_correct=(selected == color and args.get("goal_symbol") == ep["target_symbol"])
                    if trace["tool"] in GROUNDED_TOOLS else None,
                actual_prompt_target_probability=probabilities[color] if probabilities else None,
                actual_prompt_selected_probability=probabilities[selected] if probabilities and selected is not None else None,
                actual_prompt_threshold=threshold,
                request_pose_error=request_sample["error"] if request_sample else None,
                first_post_reply_sample=after,
                post_reply_sample_offset_steps=after["step"] - delivered if after else None,
                wait_sample_count=len(during),
                wait_correct_pose_samples=sum(s["error"] <= .16 for s in during),
                wait_supported_correct_pose_samples=sum(s["error"] <= .16 and s["probability"] >= .7 for s in during),
                evidence=args.get("evidence")))
        bound = [r for r in requests if r["binding_correct"] is not None]
        ready = [s for s in sampled if s["correct_binding"] and s["state"] in ("execute", "verify", "stop") and s["error"] <= .16]
        episodes.append(dict(batch=report_path.parent.name, case=ep["case"], reason=ep["reason"],
            reached=ep["reached"], first_reach_step=ep["first_reach_step"], finish_step=ep["finish_step"],
            target_color=color, target_symbol=ep["target_symbol"],
            last_explicit_binding_correct=bound[-1]["binding_correct"] if bound else None,
            stop_proposals=sum(r["tool"] == "stop" for r in requests),
            sampled_grounded_ready=len(ready),
            sampled_grounded_ready_supported=sum(s["probability"] >= .7 for s in ready),
            requests=requests))
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = sorted(args.source.glob("eval_*collision_clearance/report.json"))
    assert len(reports) == 3, "Expected all three frozen collision-clearance batches"
    rows = []
    provenance = {}
    for report in reports:
        replay = args.replays / report.parent.name
        rows.extend(analyze(report, replay))
        for path in (report, replay / "manifest.json", replay / "replay_audit.json", replay / "data.pt"):
            provenance[str(path)] = digest(path)
    assert len(rows) == 18 and len({(r["batch"], r["case"]) for r in rows}) == 18
    abstains = [r for r in rows if r["reason"] == "abstain"]
    result = dict(status="completed", cloud_calls=0,
        scope="Previously inspected development episodes; descriptive timing audit, no intervention or new task scores.",
        sampling="Replay evidence every 10 physics steps. Prompt probabilities are the recorded actual inputs. Post-reply samples may be later than delivery and do not prove the state at delivery. Replay labels are evaluator-only.",
        thresholds="Original pose threshold .16 rad and measured probability .7. No threshold search.",
        summary=dict(episodes=len(rows), reasons=dict(Counter(r["reason"] for r in rows)),
            abstain_cases=len(abstains),
            abstains_with_correct_last_binding=sum(r["last_explicit_binding_correct"] is True for r in abstains),
            abstains_without_any_grounded_binding=sum(r["last_explicit_binding_correct"] is None for r in abstains),
            abstains_ever_reached=sum(r["reached"] for r in abstains),
            abstains_with_stop_proposal=sum(r["stop_proposals"] > 0 for r in abstains),
            false_stops_with_correct_last_binding=sum(r["reason"] == "false_stop" and r["last_explicit_binding_correct"] is True for r in rows)),
        episodes=rows, provenance=provenance, source_sha256=digest(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))
    for row in rows:
        if row["reason"] in ("abstain", "false_stop"):
            print(json.dumps({k: v for k, v in row.items() if k != "requests"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
