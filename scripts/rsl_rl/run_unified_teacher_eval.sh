#!/usr/bin/env bash
set -euo pipefail

# Unified state-teacher evaluation contract.
#
# Primary: six fixed target faces x 64 episodes, random in-plane yaw,
# tolerance 0.16 rad, continuous hold 0.5 s, 10 s horizon.
# Secondary: 128 repeated-goal episodes, tolerance 0.16 rad, 30 s horizon.
#
# Example on brainco:
#   CHECKPOINT=logs/rsl_rl/.../model_750.pt \
#   bash scripts/rsl_rl/run_unified_teacher_eval.sh all

stage=${1:-all}
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/source/BrainCo_DexHand:$ROOT_DIR/scripts/rsl_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR="${TMPDIR:-/home/jiaju/tmp}"
mkdir -p "$TMPDIR/isaaclab/logs"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"

RUNTIME="${ISAAC_PYTHON:-/home/jiaju/.micromamba/envs/viserdex/bin/python}"
TASK="${EVAL_TASK:-BrainCo-Direct-Revo3-VisERDexTeacher-Cube-v0}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the teacher checkpoint path}"
REPORT_DIR="${REPORT_DIR:-outputs/unified_eval_$(basename "${CHECKPOINT%/*}")_$(basename "${CHECKPOINT%.pt}")}"
SEED="${EVAL_SEED:-101}"
EPISODES_PER_FACE="${EVAL_EPISODES_PER_FACE:-64}"
NUM_ENVS="${EVAL_NUM_ENVS:-64}"
mkdir -p "$REPORT_DIR"

run_hold() {
  for face in 0 1 2 3 4 5; do
    report="$REPORT_DIR/hold_face${face}.json"
    if [[ -s "$report" ]]; then
      echo "[SKIP] existing $report"
      continue
    fi
    echo "[RUN] hold face=$face episodes=$EPISODES_PER_FACE seed=$SEED"
    timeout -k 30s "${EVAL_TIMEOUT:-30m}" "$RUNTIME" -u scripts/rsl_rl/evaluate_state_baseline.py \
      --task "$TASK" --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
      --mode hold --faces "$face" --success-tolerance 0.16 \
      --goal-hold-time-s 0.5 --episode-length-s 10 \
      --episodes-per-face "$EPISODES_PER_FACE" --num-envs "$NUM_ENVS" \
      --seeds "$SEED" --report "$report"
  done
}

run_repeated() {
  report="$REPORT_DIR/repeated30s_tol016.json"
  if [[ -s "$report" ]]; then
    echo "[SKIP] existing $report"
    return
  fi
  echo "[RUN] repeated episodes=128 seed=$SEED"
  timeout -k 30s "${EVAL_TIMEOUT:-30m}" "$RUNTIME" -u scripts/rsl_rl/evaluate_state_baseline.py \
    --task "$TASK" --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
    --mode repeated --success-tolerance 0.16 --goal-hold-time-s 0.0 \
    --episode-length-s 30 --episodes-per-face 128 --num-envs "$NUM_ENVS" \
    --seeds "$SEED" --report "$report"
}

summarize() {
  "$RUNTIME" - "$REPORT_DIR" "$CHECKPOINT" "$TASK" <<'PY'
import hashlib
import json
import pathlib
import sys

report_dir = pathlib.Path(sys.argv[1])
checkpoint = sys.argv[2]
task = sys.argv[3]
hold = []
for path in sorted(report_dir.glob("hold_face*.json")):
    data = json.loads(path.read_text())
    rows = data.get("records", [])
    usable = [
        bool(row.get("continuous_hold", False))
        and bool(row.get("held_at_end", False))
        and not bool(row.get("drop", False))
        for row in rows
    ]
    dropped = [bool(row.get("drop", False)) for row in rows]
    reached = [bool(row.get("instant_reach", False)) and not bool(row.get("drop", False)) for row in rows]
    hold.append({
        "file": path.name,
        "face": int(path.stem.replace("hold_face", "")),
        "episodes": len(rows),
        "usable_teacher_rate": sum(usable) / len(usable) if usable else None,
        "instant_reach_rate": sum(reached) / len(reached) if reached else None,
        "drop_rate": sum(dropped) / len(dropped) if dropped else None,
        "usable_teacher_successes": sum(usable),
        "drop_count": sum(dropped),
    })

all_usable = [x for row in hold for x in [row["usable_teacher_successes"]]]
total_episodes = sum(row["episodes"] for row in hold)
total_usable = sum(row["usable_teacher_successes"] for row in hold)
total_drops = sum(row["drop_count"] for row in hold)
summary = {
    "protocol": "state_teacher_metrics_v2",
    "task": task,
    "checkpoint": checkpoint,
    "hold": {
        "episodes": total_episodes,
        "usable_teacher_rate": total_usable / total_episodes if total_episodes else None,
        "drop_rate": total_drops / total_episodes if total_episodes else None,
        "usable_teacher_successes": total_usable,
        "drop_count": total_drops,
        "per_face": hold,
    },
}
repeated_path = report_dir / "repeated30s_tol016.json"
if repeated_path.exists():
    repeated = json.loads(repeated_path.read_text())
    summary["repeated"] = repeated.get("repeated_summary", repeated.get("summary", {}))
sha = pathlib.Path(checkpoint)
if sha.exists():
    summary["checkpoint_sha256"] = hashlib.sha256(sha.read_bytes()).hexdigest()
out = report_dir / "unified_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[INFO] wrote {out}")
PY
}

case "$stage" in
  hold) run_hold; summarize ;;
  repeated) run_repeated; summarize ;;
  all) run_hold; run_repeated; summarize ;;
  summary) summarize ;;
  *) echo "usage: $0 {hold|repeated|all|summary}" >&2; exit 2 ;;
esac
