"""Validate a complete state-control evaluation grid before reporting it."""
import argparse
import json
from pathlib import Path

from state_baseline_protocol import aggregate_records


def summarize(directory):
    records = []
    hashes, revisions = set(), set()
    for split in ("train", "val"):
        for seed in (1000, 1001, 1002):
            for face in range(6):
                path = directory / f"eval_{split}_seed{seed}_face{face}.json"
                report = json.loads(path.read_text())
                assert report["complete"] and len(report["records"]) == 96, path
                hashes.add(report["checkpoint_sha256"])
                revisions.add(report["git_revision"])
                assert all(r["object_split"] == split and r["seed"] == seed and r["face"] == face
                           for r in report["records"]), path
                counts = [sum(r["object_id"] == name for r in report["records"])
                          for name in report["per_object"]]
                assert len(counts) == (6 if split == "train" else 3) and len(set(counts)) == 1, path
                records.extend(report["records"])
    assert len(hashes) == len(revisions) == 1
    result = {"complete": True, "scope": "privileged state control; not visual semantic search",
              "checkpoint_sha256": hashes.pop(), "git_revision": revisions.pop(),
              "per_split": {s: aggregate_records([r for r in records if r["object_split"] == s])
                            for s in ("train", "val")},
              "per_object": {n: aggregate_records([r for r in records if r["object_id"] == n])
                             for n in sorted({r["object_id"] for r in records})}, "records": records}
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "summary.json")
    print(json.dumps(result["per_split"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
