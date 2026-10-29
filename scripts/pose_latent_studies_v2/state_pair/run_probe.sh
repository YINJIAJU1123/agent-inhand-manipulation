#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}" "${STUDY_ROOT:?}" "${CUDA_VISIBLE_DEVICES:?}"
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
cd "$RUNTIME_ROOT/viserDex_revo3"
common=(--runtime_sources "$RUNTIME_ROOT/outputs/goal_study_release_20261029/sources"
 --student_checkpoint "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt"
 --checkpoint "$RUNTIME_ROOT/outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt"
 --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt"
 --original_asset "$PWD/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda"
 --pose_checkpoint "$PWD/checkpoints/pose_estimator/cnn_6202_final.pth"
 --camera_calibration "$PWD/config/cameras"
 --seed 182961 --viz none --device cuda:0
 env.sim.physics.solver_cfg.nconmax=4096 env.sim.physics.solver_cfg.njmax=1024)
if [[ -n "${RELOCATION_MANIFEST:-}" ]]; then common+=(--relocation_manifest "$RELOCATION_MANIFEST"); fi
for run in reference repeat; do
 extra=()
 if [[ "$run" == repeat ]]; then extra=(--reference "$STUDY_ROOT/reference"); fi
 "$VISERDEX_PYTHON" "$RUNTIME_ROOT/outputs/goal_study_release_20261029/sources/run_logged.py" --log "$STUDY_ROOT/$run.log" -- \
  bash scripts/run_newton.sh "$STUDY_ROOT/probe_reconstruction.py" "${common[@]}" "${extra[@]}" --output "$STUDY_ROOT/$run"
done
