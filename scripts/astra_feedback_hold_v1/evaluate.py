"""Use the original shared-prefix evaluator with one explicit window ablation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
release = '--feedback-release' in sys.argv
if release:
    sys.argv.remove('--feedback-release')
    if '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode')+1] != 'hold':
        raise ValueError('Feedback release applies only to the hold intervention')
    spec = importlib.util.spec_from_file_location('fresh_feedback_hold_window', Path(__file__).with_name('feedback_window.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import window
    window.RetentionWindow = module.FeedbackHoldWindow
# Isaac shutdown may terminate the process before run_path returns. Save the
# intervention identity before launching while leaving the output directory new.
destination = Path(sys.argv[sys.argv.index('--output')+1])
destination.parent.mkdir(parents=True, exist_ok=True)
sidecar = destination.parent/(destination.name+'_intervention.json')
if sidecar.exists():
    raise RuntimeError('Intervention identity already exists; refuse overwrite')
sidecar.write_text(json.dumps(dict(feedback_release=release,
    feedback_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    feedback_window_sha256=hashlib.sha256(Path(__file__).with_name('feedback_window.py').read_bytes()).hexdigest()), indent=2)+'\n')
runpy.run_path(str(ROOT/'scripts/astra_control_prefix_v1/evaluate.py'), run_name='__main__')
