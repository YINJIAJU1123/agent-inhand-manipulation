"""Probe the frozen VLM service without physics or changing benchmark outcomes.

Uses existing development validation views, including their weak geometric
readability labels. Results are input-sensitivity diagnostics, not new held-out
task success estimates. The service receives only image and instruction.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import time
from collections import Counter

import h5py
import numpy as np
from PIL import Image
import requests


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18794")
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=183401)
    parser.add_argument("--expected_head", required=True)
    parser.add_argument("--expected_adapter", required=True)
    args = parser.parse_args()
    if not args.url.startswith("http://127.0.0.1:"):
        raise ValueError("Only the local model service is supported")
    if args.cases < 6:
        raise ValueError("At least six cases are required")
    args.output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    health_response = session.get(args.url + "/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    assert health["backend"] == "rslcube" and health["frame_size"] == 120
    assert health["checkpoint_sha256"] == args.expected_head
    assert health["adapter_sha256"] == args.expected_adapter
    rows = [json.loads(line) for line in args.labels.read_text().splitlines()]
    assert all(row["split"] == "validation" for row in rows)
    assert not {183101, 183102} & {row["seed"] for row in rows}
    rng = random.Random(args.seed)
    positives = {goal: [r for r in rows if r["goal_index"] == goal and r["readable"]]
                 for goal in range(6)}
    negatives = {goal: [r for r in rows if r["goal_index"] == goal and not r["readable"]]
                 for goal in range(6)}
    for group in positives.values():
        rng.shuffle(group)
    selected, used_frames = [], set()
    for case in range(args.cases):
        goal = case % 6
        original = next(r for r in positives[goal] if r["frame"] not in used_frames)
        used_frames.add(original["frame"])
        occluded = rng.choice([r for r in negatives[goal] if r["frame"] != original["frame"]])
        changed = rng.choice([r for r in rows if r["frame"] == original["frame"]
                              and r["goal_index"] != goal])
        selected.append(dict(case=case, original=original, target_occluded=occluded,
                             changed_instruction=changed))
    manifest = dict(format="frozen_vlm_input_diagnostic_v1", health=health,
        cases=args.cases, seed=args.seed, labels_sha256=digest(args.labels),
        images_sha256=digest(args.images), source_sha256=digest(Path(__file__)),
        scope="Existing development-validation views. No new episode-success estimate.",
        label_limit="Readability is an unaudited geometric proxy, not human ground truth.",
        conditions=["original", "target_occluded_view", "uniform_gray", "changed_instruction"],
        selected=selected)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with h5py.File(args.images, "r") as images, (args.output / "predictions.jsonl").open("x") as out:
        for case in selected:
            original = case["original"]
            frame = np.asarray(images["rgb"][original["frame"]], dtype=np.uint8)
            probes = [
                ("original", frame, original, True),
                ("target_occluded_view", np.asarray(images["rgb"][case["target_occluded"]["frame"]], dtype=np.uint8), case["target_occluded"], False),
                ("uniform_gray", np.full_like(frame, 127), original, False),
                ("changed_instruction", frame, case["changed_instruction"], case["changed_instruction"]["readable"]),
            ]
            for condition, rgb, target, expected_readable in probes:
                assert rgb.shape == (120, 120, 3)
                payload = dict(image=encode(rgb), instruction=target["instruction"])
                assert set(payload) == {"image", "instruction"}
                start = time.monotonic()
                response = session.post(args.url + "/predict", json=payload, timeout=(10, 180))
                response.raise_for_status()
                prediction = response.json()
                vector = np.asarray(prediction["goal_embedding"])
                assert vector.shape == (768,) and np.isfinite(vector).all()
                assert prediction["image_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
                record = dict(case=case["case"], condition=condition, frame=target["frame"],
                    goal=target["goal_index"], instruction=target["instruction"],
                    proxy_readable=bool(expected_readable), prediction=prediction,
                    request_elapsed_s=time.monotonic() - start)
                records.append(record)
                out.write(json.dumps(record) + "\n")
                out.flush()
            if (case["case"] + 1) % 10 == 0:
                print(json.dumps(dict(completed_cases=case["case"] + 1, requests=len(records))), flush=True)
    summaries = {}
    for condition in manifest["conditions"]:
        group = [r for r in records if r["condition"] == condition]
        summaries[condition] = dict(cases=len(group),
            predicted_goal_matches_fixed_layout=sum(r["prediction"]["skill"] == r["goal"] for r in group),
            accepted=sum(r["prediction"]["accepted"] for r in group),
            accepted_correct_goal=sum(r["prediction"]["accepted"] and r["prediction"]["skill"] == r["goal"] for r in group),
            accepted_proxy_invisible=sum(r["prediction"]["accepted"] and not r["proxy_readable"] for r in group),
            median_readability=float(np.median([r["prediction"]["readable_probability"] for r in group])),
            median_inference_s=float(np.median([r["prediction"]["inference_s"] for r in group])))
    grouped = {(r["case"], r["condition"]): r for r in records}
    changes = {}
    for condition in manifest["conditions"][1:]:
        distances = []
        for case in selected:
            original = grouped[case["case"], "original"]["prediction"]
            variant = grouped[case["case"], condition]["prediction"]
            a, b = np.array(original["goal_embedding"]), np.array(variant["goal_embedding"])
            distances.append(float(np.linalg.norm(a-b)))
        changes[condition] = dict(median_latent_l2_from_original=float(np.median(distances)),
            changed_skill=sum(grouped[c["case"], "original"]["prediction"]["skill"] !=
                              grouped[c["case"], condition]["prediction"]["skill"] for c in selected))
    result = dict(completed=True, cases=args.cases, requests=len(records),
        task_rollouts=0, benchmark_modified=False, conditions=summaries, paired_changes=changes,
        source_seeds=sorted({r["seed"] for r in rows}),
        original_goal_counts=dict(Counter(c["original"]["goal_index"] for c in selected)))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
