#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUNTIME_ROOT:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
: "${CAPTURE_RUN:?}"
: "${OUTPUT_RUN:?}"
out="$RUNTIME_ROOT/outputs/$OUTPUT_RUN"
capture="$RUNTIME_ROOT/outputs/$CAPTURE_RUN"
python="$RUNTIME_ROOT/.venv-grounding/bin/python"
stage=waiting_for_capture
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$out/process_status.txt"' EXIT
printf '%s\n' "$stage" > "$out/current_stage.txt"
while [[ ! -f "$capture/process_status.txt" ]]; do sleep 20; done
grep -qx 'exit_code=0' "$capture/process_status.txt"
stage=labeling_with_image_signal
printf '%s\n' "$stage" > "$out/current_stage.txt"
for split in train validation; do
  "$python" "$out/sources/pose_latent_student_v1/label_grounding.py" \
    --dataset "$capture/$split/grounding" --output "$out/labels/$split" --split "$split" --require_image_signal
done
stage=checking_data
printf '%s\n' "$stage" > "$out/current_stage.txt"
"$python" - "$capture" "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
import h5py,numpy as np
capture,out=map(Path,sys.argv[1:])
report={}
for split,seed,expected in [('train',182931,4800),('validation',182932,2400)]:
    validity=json.loads((capture/(split+'.validity.json')).read_text())
    manifest=json.loads((capture/split/'grounding/manifest.json').read_text())
    labels=json.loads((out/'labels'/split/'manifest.json').read_text())
    if not validity['physical_log_valid'] or not manifest['completed'] or manifest['seed']!=seed:
        raise ValueError('Incomplete, invalid or unexpected capture: '+split)
    with h5py.File(capture/split/'grounding/samples.h5','r') as data:
        shapes={key:list(data[key].shape) for key in data}
        if any(shape[0]!=expected for shape in shapes.values()) or shapes['rgb']!=[expected,120,120,3] or shapes['hand_depth']!=[expected,120,120] or shapes['object_pose']!=[expected,7]:
            raise ValueError('Unaligned capture: '+str(shapes))
        pose=data['object_pose'][:]
        if not np.isfinite(pose).all() or not np.allclose(np.linalg.norm(pose[:,3:],axis=1),1,atol=1e-3):
            raise ValueError('Invalid pose labels')
        if data['rgb'].dtype!=np.uint8:
            raise ValueError('Unexpected RGB type')
    if labels['split']!=split or any(min(v)<=0 for v in labels['negative_positive_counts'].values()):
        raise ValueError('Missing supervision strata')
    report[split]=dict(seed=seed,shapes=shapes,counts=labels['negative_positive_counts'],
        label_manifest_sha256=hashlib.sha256((out/'labels'/split/'manifest.json').read_bytes()).hexdigest())
(out/'capture_validation.json').write_text(json.dumps(report,indent=2)+'\n')
PY
stage=waiting_for_label_audit
printf '%s\n' "$stage" > "$out/current_stage.txt"
while [[ ! -f "$out/label_audit.json" ]]; do sleep 20; done
"$python" - "$out" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
audit=json.loads((out/'label_audit.json').read_text())
checked=json.loads((out/'capture_validation.json').read_text())
if audit['accepted_for_training'] is not True:
    raise ValueError('Label audit rejected')
for split in ['train','validation']:
    if audit['label_manifest_sha256'][split]!=checked[split]['label_manifest_sha256']:
        raise ValueError('Reviewed labels changed')
PY
stage=preparing_replay
printf '%s\n' "$stage" > "$out/current_stage.txt"
for split in train validation; do
  "$python" "$out/sources/pose_latent_student_v1/label_grounding.py" \
    --dataset "$RUNTIME_ROOT/outputs/grounding_rgb_20261029/$split" \
    --output "$out/replay_labels/$split" --split "$split" --require_image_signal
done
stage=training
printf '%s\n' "$stage" > "$out/current_stage.txt"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
"$python" -u "$out/sources/pose_latent_student_v1/grounding.py" \
  --model "$RUNTIME_ROOT/models/Qwen3-VL-4B-Instruct" \
  --warm_start "$out/step_00240" \
  --train "$out/labels/train" --validation "$out/labels/validation" \
  --replay_train "$out/replay_labels/train" --replay_validation "$out/replay_labels/validation" \
  --text_cache "$RUNTIME_ROOT/VLMmanipulation/outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt" \
  --original_asset "$RUNTIME_ROOT/viserDex_revo3/source/viserdex/viserdex/assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda" \
  --output "$out/fit" --steps 240 --accumulation 2 --learning_rate .00005 --seed 182941
stage=finished
printf '%s\n' "$stage" > "$out/current_stage.txt"
