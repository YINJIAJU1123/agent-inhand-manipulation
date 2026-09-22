"""Process isolation for multi-block Isaac Sim evaluation (no simulator imports)."""

import json
from pathlib import Path
import subprocess
import sys

from state_baseline_protocol import aggregate_records, repeated_summary


def merge_block_reports(reports, seeds, faces, quota):
    expected = {(seed, face) for seed in seeds for face in faces}
    seen = set()
    records = []
    for block in reports:
        key = (block["seeds"][0], block["faces"][0])
        if key not in expected or key in seen or not block["complete"]:
            raise ValueError("Unexpected, duplicate, or incomplete evaluation block")
        if len(block["records"]) != quota:
            raise ValueError("Evaluation block has an incorrect episode count")
        for field in ("checkpoint_sha256", "task", "protocol", "hold_tolerance_rad", "hold_time_s"):
            if block[field] != reports[0][field]:
                raise ValueError(f"Evaluation blocks disagree on {field}")
        seen.add(key)
        records.extend(block["records"])
    if seen != expected:
        raise ValueError("Missing evaluation blocks")
    report = dict(reports[0])
    report.update(seeds=seeds, faces=faces, records=records, complete=True,
                  summary=aggregate_records(records),
                  per_face={str(f): aggregate_records([r for r in records if r["face"] == f]) for f in faces},
                  per_seed={str(s): aggregate_records([r for r in records if r["seed"] == s]) for s in seeds},
                  per_object={name: aggregate_records([r for r in records if r.get("object_id") == name])
                              for name in sorted({r["object_id"] for r in records if "object_id" in r})})
    if report["protocol"] == "repeated_reorientation_v1":
        report["per_face"] = {}
        report["repeated_summary"] = repeated_summary(records)
        report["repeated_per_seed"] = {str(s): repeated_summary([r for r in records if r["seed"] == s]) for s in seeds}
    # Keep every block's configuration, rather than presenting the first
    # fixed face/seed as if it were the configuration of the whole report.
    report["block_manifests"] = [{k: v for k, v in b.items() if k not in
                                   {"records", "summary", "per_face", "per_seed", "per_object"}}
                                  for b in reports]
    report.pop("env_cfg", None)
    return report


def run_isolated_blocks(args, argv):
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    faces = [-1] if args.mode == "repeated" else [int(f) for f in args.faces.split(",") if f.strip()]
    if not seeds or not faces or len(set(seeds)) != len(seeds) or len(set(faces)) != len(faces):
        raise ValueError("Seeds and faces must be nonempty and unique")
    if args.mode == "hold" and any(f < 0 or f > 5 for f in faces):
        raise ValueError("Face IDs must be in [0, 5]")
    if len(seeds) * len(faces) == 1:
        return False
    output = Path(args.report).expanduser().resolve()
    blocks_dir = output.parent / (output.stem + "_blocks")
    blocks_dir.mkdir(parents=True, exist_ok=True)
    forwarded = []
    tokens = iter(argv[1:])
    for token in tokens:
        if token in {"--seeds", "--faces", "--report"}:
            next(tokens)
        elif not any(token.startswith(key + "=") for key in ("--seeds", "--faces", "--report")):
            forwarded.append(token)
    reports = []
    for seed in seeds:
        for face in faces:
            path = blocks_dir / f"seed_{seed}_face_{face}.json"
            print(f"[EVAL] Isolated process seed={seed} face={face}", flush=True)
            subprocess.run([sys.executable, argv[0], *forwarded,
                            "--seeds", str(seed), "--faces", str(face), "--report", str(path)], check=True)
            reports.append(json.loads(path.read_text()))
    report = merge_block_reports(reports, seeds, faces, args.episodes_per_face)
    report["argv"] = argv
    report["block_process_isolation"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[EVAL] Completed {len(report['records'])} episodes; report: {output}", flush=True)
    return True
