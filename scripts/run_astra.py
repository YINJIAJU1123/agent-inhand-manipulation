"""Current autonomous-stop Astra experiment with published frozen assets."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    command=[os.environ.get('ISAACLAB_PYTHON',sys.executable),str(ROOT/'scripts/astra_startup_live_v1/evaluate.py'),
        '--startup-mode','fixed_green','--memory-mode','memory','--arm','verify',
        '--evidence-model',str(ROOT/'checkpoints/runtime/pose_verifier.pt'),
        '--seed','110101','--headless','--device','cuda:0']+sys.argv[1:]
    raise SystemExit(subprocess.call(command,cwd=ROOT))
